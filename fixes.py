import re

from tqdm.auto import tqdm


class Fix:
    def __init__(self, pat_b, pat_g, repl, flags=re.UNICODE):
        self.pat_b, self.pat_g, self.repl, self.flags = pat_b, pat_g, repl, flags

    def count(self, series):
        s_bad = series.str.count(self.pat_b, flags=self.flags)
        s_good = series.str.count(self.pat_g, flags=self.flags)
        return s_bad, s_good

    def replace(self, series):
        return series.str.replace(self.pat_b, self.repl, flags=self.flags, regex=True)


class DeleteSpaceBeforePunctuation(Fix):
    # iskyrus pries daugtaski
    def __init__(self):
        super().__init__(pat_b='(?<![\s–:,;])\s+(?=[;\.,:\)](?!\.\.))', pat_g='(?<![\s–:])[;\.,:\)(?!\.\.)]', repl="")
        self.to_avoid = ['\.com', '\.lt', '\.uk', '\.org', '\.net',
                         '\.jpg', '\.png', '\.tiff', '\.consumers', '\.unite', '\.xxx']

    def count(self, series):
        s_bad, s_good = series.copy(), series.copy()
        s_bad[:], s_good[:] = 0, 0
        mask_avoid = series.str.contains("|".join(self.to_avoid), regex=True)
        u1, u2 = super().count(series.loc[~mask_avoid])
        s_bad.loc[~mask_avoid] = u1
        s_good[~mask_avoid] = u2
        return s_bad.astype(int), s_good.astype(int)

    def replace(self, series):
        sr = series.copy(deep=True)
        mask = ~sr.str.contains("|".join(self.to_avoid), regex=True)
        sr.loc[mask] = super().replace(sr.loc[mask])
        return sr


class AddSpaceAfterPoint(Fix):
    def __init__(self):
        # 5 pelnytais taškais – 60:52. M.A.M.A.
        super().__init__(pat_b='\.(?![\.\s\d,\-)“\/:\]\>])(?=.)', pat_g='\.\s(?![\.\d,\)“\/:])(?=.)', repl=". ")
        self.to_avoid = ['M\.A\.M\.A\.', 'www', 'http:\/\/', '\.com', '\.lt', '\.uk', '\.org', '\.net',
                         '\.jpg', '\.png', '\.tiff', '\.consumers', '\.unite', '\.xxx', 'T\.Ė\.T\.Ė.']

    def count(self, series):
        s_bad, s_good = series.copy(), series.copy()
        s_bad[:], s_good[:] = 0, 0
        mask_avoid = series.str.contains("|".join(self.to_avoid), regex=True)
        u1, u2 = super().count(series.loc[~mask_avoid])
        s_bad.loc[~mask_avoid] = u1
        s_good[~mask_avoid] = u2
        return s_bad.astype(int), s_good.astype(int)

    def replace(self, series):
        sr = series.copy(deep=True)
        mask = ~sr.str.contains("|".join(self.to_avoid), regex=True)
        sr.loc[mask] = super().replace(sr.loc[mask])
        return sr


class AddSpaceBefore_m_d(Fix):
    def __init__(self):
        super().__init__(pat_b='[\d]{1,4}(?=[md]\.)', pat_g='[\d]{1,4}\s(?=[md]\.)', repl=lambda x: x[0] + " ")


class NormalizeKabutes(Fix):
    def __init__(self):
        super().__init__(pat_b="""[„“"”].+?[“"”]""", pat_g="„.+?“", repl=lambda x: "„" + x[0][1:-1] + "“")
        self.initial_repl = {'‘‘': '“', ',,': '„', "''": '"'}

    def count(self, series):
        s_good = series.str.count(self.pat_g, flags=self.flags)
        s_bad = series.copy()
        for k, v in self.initial_repl.items():
            s_bad = s_bad.str.replace(k, v, flags=re.UNICODE)
        s_bad = s_bad.str.count(self.pat_b, flags=self.flags) - s_good
        return s_bad, s_good

    def replace(self, series):
        for k, v in self.initial_repl.items():
            series = series.str.replace(k, v, flags=re.UNICODE)
        return super().replace(series)


class RemoveDublicatedFirstLetter(Fix):
    def __init__(self):
        # let's do not use M, C, X, I as these can repeat as roman numerals
        # there is also very rare cases of artistic cases: uuuliavojo - padaryti, kad tik dvi tokios pacios, nes kai trys - tada ir būna pagražinimai
        # kad isvengti mm, ppm?
        letters = 'aąbcčdeęėfghiįyjklnorsštuųūvzž'  # isimame p ir m, del ppm ir mm, komendantas pplk. N. Sulikas
        e = r'(?!\b)'
        super().__init__(
            pat_b=f'(?<=\s)(?P<raide>[{letters}])(?=(?P=raide)){e}|(?<=\A)(?P<raid>[{letters}])(?=(?P=raid)){e}',
            pat_g=f'(?<=\s)(?P<raide>[{letters}])(?!(?P=raide)){e}|(?<=\A)(?P<raid>[{letters}])(?!(?P=raid)){e}',
                         repl="")


class RemoveDublicatedWords(Fix):
    def __init__(self):
        words = ['į', 'kad', 'nei', 'su', 'o', 'iš', 'kaip', 'bet', 'prie', 'apie', 'už', 'kai', 'dėl', 'po', 'per']
        super().__init__(pat_b=f'\s(?P<zodis>({"|".join(words)}))(?=\s(?P=zodis)\s)', repl='',
                         pat_g=f'(?<=\s)({"|".join(words)})(?=\s)')

    def count(self, series):
        s_bad = series.str.count(self.pat_b, flags=self.flags)  # poros
        s_good = series.str.count(self.pat_g, flags=self.flags)  # pavieniai
        return s_bad, s_good - s_bad*2


def other_fixes(series):
    # very frequent: ' iir '
    dd = {'…': '...', '¬': '', '—': '–', '\u2028': ""}
    for k, v in dd.items():
        series = series.str.replace(k, v, regex=False)
    return series


fixes_list = [
    # ('fix_kabutes', NormalizeKabutes),
    ('add_space_before_m_d', AddSpaceBefore_m_d),
    ('add_space_after_point', AddSpaceAfterPoint),
    ('delete_space_before_punctuation', DeleteSpaceBeforePunctuation),
    ('remove_dublicated_first_letter', RemoveDublicatedFirstLetter),
    # ('remove_dublicated_words', RemoveDublicatedWords)
]

# ù, ó  paslėp-tas


def count_stats_of_fixes(df):
    # turns series into a dataframe with bunch of additional statistics columns
    sr = df[['website', 'text']].copy()
    for name, class_ in tqdm(fixes_list):
        sr[f"{name}-bad"], sr[f"{name}-good"] = class_().count(df['text'].copy())
    return sr


def do_fixes(series):
    for _, class_ in fixes_list:
        series = class_().replace(series)
    series = other_fixes(series)
    return series
