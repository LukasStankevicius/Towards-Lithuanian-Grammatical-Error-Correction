from collections import defaultdict
import pandas as pd
import json
from tqdm.auto import tqdm
import difflib
import pickle
from collections import Counter
import random
from pathlib import Path
import numpy as np


class Typo:

    def __init__(self, corpus='twitter', seed=42, layout="qwerty", weight=1.0):
        self.weight = weight

        if corpus not in ['twitter', 'github']:
            raise ValueError

        self.azerty_layout = str.maketrans({"q": 'a', "w": "z", "a": "q", ";": "m", "z": "w", "m": ",", ",": ";",
                                            ".": ":"})
        self.qwertz_layout = str.maketrans({"z": "y", "y": "z"})

        stats = self.init_twitter_statistics() if corpus == 'twitter' else self.init_github_statistics(layout=layout)
        self.char_count, self.deleted_char_count, self.substitution_counts = stats[:3]
        self.transposition_counts, self.insertion_before_target_after = stats[3:]

        # filter insignificant counts:
        self.filter_significant(min_count=1000)

        substitution_p, self.substitution_matrix = self.get_substitution_p_matrix()
        self.transposition_p = self.get_transposition_p()  # {"de":0.14, }
        insert_p_after, insert_p_before, self.insert_after_m, self.insert_before_m = self.get_insertion_p()

        # columns as transformations, index as letters, values as probabilities
        self.char_transform_p = pd.DataFrame([self.get_deletion_p(), substitution_p, insert_p_after, insert_p_before],
                                             index=['deletion', 'substitution', 'insert_after',
                                                    'insert_before']).T.sort_index()
        # these two do the same thing, so:
        self.char_transform_p['insert_before'] /= 2
        self.char_transform_p['insert_after'] /= 2

        # we will save every type of mistake generated so that after generation finishes, this attribute can show
        # the work done
        self.mistakes_generated = defaultdict(int)
        random.seed(seed)
        np.random.seed(seed)
    #         self.char_transform_p.plot(kind='bar', stacked=True)

    def filter_significant(self, min_count=1000):
        """
        the idea is to leave only characters or combinations of significant frequency
        :return:
        """
        # use only more frequent characters
        mask = self.char_count['occurance_count all'] >= min_count
        self.char_count = self.char_count.loc[mask]
        idx = self.char_count.index

        self.deleted_char_count = self.deleted_char_count.loc[idx.intersection(self.deleted_char_count.index)]
        self.substitution_counts = self.substitution_counts.loc[self.substitution_counts.isin(idx).all(axis=1)]
        self.transposition_counts = self.transposition_counts[self.transposition_counts['count all'] >= min_count]

        mask2 = self.insertion_before_target_after.isin(idx).all(axis=1)
        self.insertion_before_target_after = self.insertion_before_target_after.loc[mask2]

    def init_github_statistics(self, typo_corpus_file='github-typo-corpus.v1.0.0.jsonl', lower=True, lang='eng',
                               layout=None):
        name = f'github_init_stats_{layout}.pickle'
        if Path(name).exists():
            with open(name, 'rb') as f:
                print(f"loading precomputed: {name}")
                data = pickle.load(f)
                return data

        rezult = []
        with open(typo_corpus_file, encoding='utf-8') as f:
            for line in tqdm(f):
                a = json.loads(line)
                for edit in a['edits']:
                    rezult.append((edit['src']['text'], edit['src']['lang'], edit['tgt']['text'], edit['tgt']['lang']))
        df = pd.DataFrame(rezult, columns=['src_text', 'src_lang', 'tgt_text', 'tgt_lang'])
        if lang is not None:
            mask = df['src_lang'] == lang
            df = df.loc[mask]

        if lower:
            print("lowering text")
            df['src_text'] = df['src_text'].str.lower()
            df['tgt_text'] = df['tgt_text'].str.lower()
        
        if layout == 'qwertz':
            print("applying qwertz keyboard layout")
            df['src_text'] = df['src_text'].str.translate(self.qwertz_layout)
            df['tgt_text'] = df['tgt_text'].str.translate(self.qwertz_layout)
        elif layout == 'azerty':
            print("applying azerty keyboard layout")
            df['src_text'] = df['src_text'].str.translate(self.azerty_layout)
            df['tgt_text'] = df['tgt_text'].str.translate(self.azerty_layout)
        elif layout == "qwerty":
            pass
        else:
            raise ValueError

        char_count = pd.DataFrame(
            Counter([letter for row in df['tgt_text'].dropna().tolist() for letter in row]).most_common(),
            columns=['char', 'occurance_count all']).set_index('char')

        print("This may take several minutes, please be patient")
        df['difflib'] = df.apply(lambda x: "".join(difflib.ndiff(x['tgt_text'], x['src_text'])), axis=1)
        print('Being patient finished')

        # deletion
        df['deleted'] = df['difflib'].str.findall(r"\s\s.\s\s.-\s(.)\s\s.")
        mask = df['deleted'].apply(len) > 0
        print(f"we have {mask.sum()} rows with deletion")
        deleted_char_count = pd.DataFrame(
            Counter([letter for row in df.loc[mask, 'deleted'].tolist() for letter in row]).most_common(),
            columns=['char', 'delete_count']).set_index('char')
        # substitution statistics
        df['substitution'] = df['difflib'].str.findall(r"-\s(.)\+\s(.)")  # letter to replace, replacement letter
        mask = df['substitution'].apply(len) > 0
        print(f"we have {mask.sum()} rows with substitution")
        substitution_counts = pd.DataFrame([pair for row in df.loc[mask, 'substitution'].tolist() for pair in row],
                                           columns=['char_original', 'char_typo'])
        # transposition
        df['transposed'] = df['difflib'].str.findall(r"\+\s(.)\s\s(.)-\s\1")
        mask = df['transposed'].apply(len) > 0
        print(f"we have {mask.sum()} rows with transposition")
        two_chars_transposition_counts = pd.DataFrame(
            Counter(
                ["".join(i)[::-1] for row in df.loc[mask, 'transposed'].tolist() for i in row]).most_common(),
            columns=['cc', 'count transposition']).set_index('cc')
        two_chars_total_count = pd.DataFrame(Counter(
            [word[i: i + 2] for word in df['tgt_text'].tolist() for i in range(len(word) - 1) if
             len(word) > 1]).most_common(),
                                             columns=['cc', 'count all']).set_index('cc')
        transposition_counts = two_chars_transposition_counts.join(two_chars_total_count)
        # insertion
        df['inserted'] = df['difflib'].str.findall(r"\s\s(.)\+\s(.)\s\s(.)\s\s.")
        mask = df['inserted'].apply(len) > 0
        print(f"we have {mask.sum()} rows with inserted")
        insertion_before_target_after = pd.DataFrame([pair for rw in df.loc[mask, 'inserted'].tolist() for pair in rw],
                                                     columns=['char_before', 'char', 'char_next'])

        data = char_count, deleted_char_count, substitution_counts, transposition_counts, insertion_before_target_after
        with open(name, 'wb') as f:
            print(f'dumping precomputed to {name}')
            pickle.dump(data, f)

        return data

    @staticmethod
    def init_twitter_statistics(typo_corpus_file='typo-corpus-r1.txt',
                                original_column="context of original (freq)"):
        # THIS METHOD IS NOT MAINTAINED
        # I moved my focus to github as it was larger. Combining both is not possible due to different normalization
        df = pd.read_csv(typo_corpus_file, sep='\t', names=["typo", "original", "operation", "operation symbol",
                                                            "context of typo (freq)", "context of original (freq)"])

        df["context of original (freq)"] = df["context of original (freq)"].str.split("(").str.get(
            0).str.replace("_", " ")

        char_count = pd.DataFrame(
            Counter([a for aa in df[original_column].dropna().tolist() for a in aa]).most_common(),
            columns=['char', 'occurance_count all']).set_index('char')

        # deletion statistics
        df0 = df.loc[df['operation'] == "RM"].copy()
        df0['char'] = df0['operation symbol'].str.findall(r"\((.*?)\)").str.get(0)
        deleted_char_count = pd.DataFrame(Counter(df0['char'].tolist()).most_common(),
                                          columns=['char', 'delete_count']).set_index('char')
        # substitution statistics
        df1 = df.loc[df['operation'] == "R2"].copy()
        # filter space from this
        df1 = df1.loc[~((df1[['char_typo', 'char_original']] == "_").any(axis=1))]

        df1['char_typo'] = df1['operation symbol'].str.findall(r"\[(.*?)\]").str.get(0).str.split(":").str.get(0)
        df1['char_original'] = df1['operation symbol'].str.findall(r"\[(.*?)\]").str.get(0).str.split(":").str.get(1)
        substitution_counts = df1[['char_original', 'char_typo']]

        # transposition

        df2 = df.loc[df['operation'] == "R1"].copy()
        df2['consecutive_characters'] = df2['operation symbol'].str.findall(r"\[(.*?)\]").str.get(0).str.replace(
            "/", "").str[::-1]

        two_chars_transposition_counts = pd.DataFrame(Counter(df2['consecutive_characters'].tolist()).most_common(),
                                                      columns=['cc', 'count transposition']).set_index('cc')

        two_chars_total_count = pd.DataFrame(Counter(
            [word[i: i + 2] for word in df2[original_column].tolist() for i in range(len(word) - 1) if
             len(word) > 1]).most_common(),
                                             columns=['cc', 'count all']).set_index('cc')
        transposition_counts = two_chars_transposition_counts.join(two_chars_total_count)

        # insertion
        df3 = df.loc[df['operation'] == "IN"].copy()
        df3['original'].fillna("", inplace=True)  # ten kur yra nan, ten yra tiesiog įterptas nereikalingas tarpas
        df3['char'] = df3['operation symbol'].str.findall(r"<(.*?)>").str.get(0)
        df3['char_before'] = df3['operation symbol'].str.findall(r".(?=<)").str.get(0)
        df3['char_next'] = df3['operation symbol'].str.findall(r"(?<=>).").str.get(0)

        insertion_before_target_after = df3[['char_before', 'char', 'char_next']]

        return char_count, deleted_char_count, substitution_counts, transposition_counts, insertion_before_target_after


    def get_insertion_p(self):
        df3 = self.insertion_before_target_after.copy()
        insertion_before = pd.DataFrame(Counter(df3['char_next'].dropna().tolist()).most_common(),
                                        columns=['char', 'insertion_before_this_char']).set_index('char')
        insertion_next = pd.DataFrame(Counter(df3['char_before'].dropna().tolist()).most_common(),
                                      columns=['char', 'insertion_after_this_char']).set_index('char')
        df1_sub = self.char_count.join(insertion_before).join(insertion_next)

        d_after = (df1_sub['insertion_after_this_char'] / df1_sub['occurance_count all']).fillna(0).to_dict()

        d_before = (df1_sub['insertion_before_this_char'] / df1_sub['occurance_count all']).fillna(0).to_dict()

        # now given a character, what probabilities of characters to insert are?
        # before

        df1_2 = pd.DataFrame(Counter((df3['char_next'] + df3['char']).tolist()).most_common(),
                             columns=['c->c', 'count'])
        df1_2['char_next'] = df1_2['c->c'].str.get(0)
        df1_2['char_inserted_before'] = df1_2['c->c'].str.get(1)

        df1_2 = df1_2.sort_values(['char_next', 'char_inserted_before'])
        df1_2 = df1_2.set_index(['char_next', 'char_inserted_before'])['count'].unstack('char_inserted_before')
        matrix_to_insert_before = df1_2.div(df1_2.sum(axis=1), axis=0).fillna(0)

        # inserting character after the given character:
        df1_2 = pd.DataFrame(Counter((df3['char_before'] + df3['char']).tolist()).most_common(),
                             columns=['c->c', 'count'])
        df1_2['char_before'] = df1_2['c->c'].str.get(0)
        df1_2['char_inserted_after'] = df1_2['c->c'].str.get(1)

        df1_2 = df1_2.sort_values(['char_before', 'char_inserted_after'])
        df1_2 = df1_2.set_index(['char_before', 'char_inserted_after'])['count'].unstack('char_inserted_after')
        matrix_to_insert_after = df1_2.div(df1_2.sum(axis=1), axis=0).fillna(0)

        return d_after, d_before, matrix_to_insert_after, matrix_to_insert_before

    def get_transposition_p(self):
        p2 = self.transposition_counts.copy()
        d = (p2['count transposition'] / p2['count all']).to_dict()
        return d

    def get_substitution_p_matrix(self):
        char_substitution_counts = pd.DataFrame(
            Counter(self.substitution_counts['char_original'].tolist()).most_common(),
            columns=['char', 'substitute_count']).set_index('char')
        df1_2 = pd.DataFrame(
            Counter((self.substitution_counts['char_original'] + self.substitution_counts['char_typo']).tolist()
                    ).most_common(),
            columns=['c->c', 'count'])

        cc_df = self.char_count.join(char_substitution_counts)
        d = (cc_df['substitute_count'] / (cc_df['occurance_count all'])).fillna(0).to_dict()

        # create a matrix for each letter destinations
        df1_2['char_original'] = df1_2['c->c'].str.get(0)
        df1_2['char_typo'] = df1_2['c->c'].str.get(1)
        df1_2 = df1_2.sort_values(['char_original', 'char_typo']).set_index(['char_original', 'char_typo'])[
            'count'].unstack('char_typo')
        matrix = df1_2.div(df1_2.sum(axis=1), axis=0).fillna(0)
        return d, matrix

    def get_deletion_p(self):
        cc_df = self.char_count.join(self.deleted_char_count)
        d = (cc_df['delete_count'] / (cc_df['occurance_count all'])).fillna(0).to_dict()
        return d

    def generate_errors(self, text):
        output_text = ""
        skip = False
        n = len(text)
        for i, char in enumerate(text):
            is_upper = char.isupper()
            if char.lower() not in self.char_transform_p.index:
                output_text += char
                continue
            if skip:
                skip = False
                continue
            # write all operations and respective probabilities for the current character
            transformations = ['deletion', 'substitution', 'insert_after', 'insert_before']
            probabilities = (self.char_transform_p.loc[char.lower(), transformations] * self.weight).tolist()
            # if there is at least one in a future token to swap with, transposition is possible
            if i <= n - 2:
                if text[i:i + 2].lower() in self.transposition_p:
                    probabilities.append(self.transposition_p[text[i:i + 2].lower()] * self.weight)
                    transformations.append('transposition')
            if random.random() < sum(probabilities):
                operation = random.choices(population=transformations, weights=probabilities, k=1)[0]
                self.mistakes_generated[operation] += 1
            else:
                # operation = 'nothing'
                output_text += char
                self.mistakes_generated["nothing"] += 1
                continue
            char = char.lower()
            if operation == 'deletion':
                continue
            elif operation == 'substitution':
                new_char = np.random.choice(self.substitution_matrix.loc[char].index.values, 1,
                                            p=self.substitution_matrix.loc[char].tolist())[0]
                output_text += new_char if not is_upper else new_char.upper()
            elif operation == 'insert_after':
                new_char = np.random.choice(self.insert_after_m.loc[char].index.values, 1,
                                            p=self.insert_after_m.loc[char].tolist())[0]
                output_text += (char + new_char) if not is_upper else (char + new_char).upper()
            elif operation == 'insert_before':
                new_char = np.random.choice(self.insert_before_m.loc[char].index.values, 1,
                                            p=self.insert_before_m.loc[char].tolist())[0]
                output_text += (new_char + char) if not is_upper else (new_char + char).upper()
            elif operation == 'transposition':
                output_text += text[i + 1] + (char if not is_upper else char.upper())
                skip = True
        return output_text
