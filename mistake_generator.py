from random import random, choices
import re
from tqdm.auto import tqdm


class Mistake:
    def __init__(self, pat, frac=0.5, flags=None):
        self.pat, self.frac = pat, frac
        self.flags = (re.UNICODE | re.IGNORECASE) if flags is None else flags

    def corrupt_match(self, match): return match.group(0)

    def analyze(self, series): return series.str.count(self.pat, flags=re.UNICODE | re.IGNORECASE)

    def corrupt(self, series):
        return series.str.replace(pat=self.pat, regex=True, flags=self.flags,
                                  repl=lambda x: x.group(0) if (random() > self.frac) else self.corrupt_match(x))


class Suduslejimas(Mistake):
    def __init__(self, frac=0.1):
        super().__init__(pat=r"(?<=\w)[bdgzž][ptksš]", frac=frac)
        suduslejimas_dict = {"b": "p", 'd': "t", "g": "k", "z": "s", "ž": "š"}
        self.suduslejimas_dict = {**suduslejimas_dict, **{k.upper(): v.upper() for k, v in suduslejimas_dict.items()}}

    def corrupt_match(self, match):
        return self.suduslejimas_dict[match.group(0)[0]] + match.group(0)[1:]


class Suskardejimas(Mistake):
    def __init__(self, frac=0.05):
        super().__init__(pat=r"(?<=\w)[ptksš][bdgzž]", frac=frac)
        suskard_dict = {"p": "b", "t": 'd', "k": "g", "s": "z", "š": "ž"}
        self.suskard_dict = {**suskard_dict, **{k.upper(): v.upper() for k, v in suskard_dict.items()}}

    def corrupt_match(self, match):
        return self.suskard_dict[match.group(0)[0]] + match.group(0)[1:]


class Geminata2(Mistake):
    # two similar consonants
    def __init__(self, frac=0.05):
        super().__init__(pat=r"([cčsšzž])(?!\1)[cčsšzž]", frac=frac)

    def corrupt_match(self, match):
        return match.group(0)[1]


class Geminata(Mistake):
    # two same consonants
    def __init__(self, frac=0.05):
        super().__init__(pat=r'(([bcčdfghjklmnprsštvzž])\s*\2)+', frac=frac)

    def corrupt_match(self, match):
        if random() > 0.5:
            return match.group(0)[1:]
        else:
            return match.group(0)[:-1]


def add_delete_spaces(series, frac=0.01):
    # deleting spaces
    series = series.str.replace(r"\s", lambda x: x[0] if random() > frac else "")
    # inserting spaces
    series = series.str.replace(r"\B", lambda x: x[0] if random() > frac else " ")
    return series


def swapcase(series, frac):
    pat = r'(?<!\A)(?<![a-ząčęėįšųūž][\.!?]\s)\b[a-ząčęėįšųūžA-ZĄČĘĖĮŠŲŪŽ]{1}(?=[a-ząčęėįšųūž]*\b)'
    series = series.str.replace(pat=pat, regex=True, repl=lambda x: x[0] if random() > frac else x[0].swapcase())
    return series


GROUPS = [
    (r"[,\.–]{0,1} ", [' ', ', ', '. ', '– '], [79695056, 9876726, 5125941, 1347515]),
    (r"[\.,;:\–\-?!\(\)\[\]\<\>/]", [',', '.', '–', ')', '(', '-', ':', '?', '!', ';', '/', '[', ']', '>', '<'],
[10072919, 7976435, 1453095, 665253, 655651, 546698, 519928, 300962, 106333, 105526, 90778, 34295, 34283, 5759, 4457]),
    ('u{0,1}ou{0,1}', ['o', 'uo', 'ou', 'uou'], [33058916, 3355463, 41509, 34]),
    ('ia|e', ['ia', 'e'], [6733731, 35509427]),
    ('[scz]', ['s', 'c', 'z'], [47349069, 2645328, 1646823]),
    ('[ščž]', ['š', 'č', 'ž'], [7002598, 2619317, 5044500]),
    ('[eęė]', ['e', 'ę', 'ė'], [35509427, 1336170, 9781460]),
    ('[iįy]', ['į', 'y', 'i'], [3490952, 8347510, 82431807]),
    ('[uųū]', ['ū', 'ų', 'u'], [2795974, 7826828, 28978236]),
    ('[aą]', ['a', 'ą'], [68291558, 4471872]),
    ('[cč]', ['c', 'č'], [2645328, 2619317]),
    ('[zž]', ['z', 'ž'], [1646823, 5044500]),
    ('[td]', ['t', 'd'], [35864854, 14822144]),
    ('[kg]', ['k', 'g'], [26461947, 10626341]),
    ('[pb]', ['p', 'b'], [16187509, 8148725]),
    (r"""‘‘|,,|[„“"”]|''""", ['"', '”', ',,', '‘‘', "''"], [436378, 46847, 11777, 817, 87]),
]


def ff(original_, values_, weights_):
    output = choices(values_, weights=weights_)[0]
    if original_.isupper():
        output = output.upper()
    return output


def generate_mistakes(series, frac=0.1):
    for i in tqdm([Suduslejimas(frac=frac), Suskardejimas(frac=frac), Geminata2(frac=frac), Geminata(frac=frac)]):
        series = i.corrupt(series)
    series = swapcase(series, frac=frac)
    for pat, values, weights in tqdm(GROUPS):
        series = series.str.replace(pat=pat, regex=True, flags=re.IGNORECASE,
                                    repl=lambda x: x.group(0) if (random() > frac) else ff(x[0], values, weights))
    series = add_delete_spaces(series, frac=frac)
    return series


