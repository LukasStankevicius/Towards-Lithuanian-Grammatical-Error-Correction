import re
import logging

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


def by_fraction_lithuanian(series, lengths):
    # ìĩá
    liet_regex = '[aąbcčdeęėfghiįyjklmnoprsštuųūvzž\.,\(\);:–\-?!„“\[\]\<\>\/\d\s€₤$%wx]'
    fraction = series.str.count(liet_regex, re.UNICODE | re.IGNORECASE) / lengths
    # >= 0.98
    return fraction


def get_fraction_of_letters_to_non_letters(series):
    n_letters = series.str.count('\w', re.UNICODE | re.IGNORECASE)
    n_non_letters = series.str.count('[^\w]', re.UNICODE | re.IGNORECASE)
    # 1<= <= 12
    return n_letters / n_non_letters


def get_fraction_of_words_to_non_words(series):
    n_letters = series.str.count('\w+', re.UNICODE | re.IGNORECASE)
    n_non_letters = series.str.count('[^\w]+', re.UNICODE | re.IGNORECASE)
    # <= 0.9
    return n_letters / n_non_letters


def get_fraction_of_spaces_to_non_spaces(series):
    #  > 0.3
    # 'K o m e r c i n ė r i z i k a – tai rizika,
    # < 0.03:
    # Išsamesnė informacija:http://www.mita.lt/lt/naujienos/7bp/skatinamas_baltijos_valstybiu
    # >=0.03 < 0.05:
    # 'Šis tekstas paimtas iš Vatikano radijo puslapio (http://lt.radiov

    n_spaces = series.str.count('\s', re.UNICODE | re.IGNORECASE)
    n_non_spaces = series.str.count('[^\s]', re.UNICODE | re.IGNORECASE)
    # > 0.05 would remove 0.0005770932231525629
    return n_spaces / n_non_spaces


def my_filter(df, min_characters=20, min_lithuanian_fraction=0.98, min_fraction_of_spaces_to_non_spaces=0.02):
    n0 = len(df)
    logger.info(f'We start with {n0} rows')
    lengths = df['text'].apply(len)
    mask = lengths >= min_characters

    df = df.loc[mask]
    n1 = len(df)
    logger.info(f'Filtering by length removed {n0-n1} rows')

    lit_frac = by_fraction_lithuanian(df['text'], lengths)
    mask = lit_frac >= min_lithuanian_fraction
    df = df.loc[mask]
    n2 = len(df)
    logger.info(f'Filtering by how lithuanian removed {n1-n2} rows more')

    space_frac = get_fraction_of_spaces_to_non_spaces(df['text'])
    mask = space_frac >= min_fraction_of_spaces_to_non_spaces
    df = df.loc[mask]
    n3 = len(df)
    logger.info(f'Filtering by fraction of spaces to non spaces removed {n2-n3} rows even more')

    logger.info(f'Now we are left with {n3} rows. From initial only  {n3*100/n0:2.2f} % remains.')
    return df[['text']]

