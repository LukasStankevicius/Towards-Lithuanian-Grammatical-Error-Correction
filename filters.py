import re
import pandas as pd
from matplotlib import pyplot as plt, patches
import logging
from fixes import fixes_list

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


def plot_inspect(series):
    # limits
    l = [
        [[0, 1], [1, 10**6]],
        [[0.8, 1], [1, 10**6]],
        [[0.95, 1], [1, 10**6]]
    ]
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 4))

    for i, ax in enumerate(axes):
        mask = (series >= l[i][0][0]) & (series <= l[i][0][1])
        series.loc[mask].hist(bins=100, log=True, ax=ax)
        if i != 2:
            ax.add_patch(patches.Rectangle(
                (l[i+1][0][0], l[i+1][1][0]), l[i+1][0][1] - l[i+1][0][0], l[i+1][1][1] - l[i+1][1][0],
                linewidth=1, edgecolor='r', alpha=0.1, zorder=2))
        ax.set_xlim(l[i][0]), ax.set_ylim(l[i][1])
    plt.show()


def get_large_groups(df):
    #  df should contain website, author, year fields
    #  The first, make everything not null:
    df['author'].where(~df['author'].isnull(), '', inplace=True)
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['date'].where(~df['date'].isnull(), pd.Timestamp('1921-12-19 09:45:00+0000', tz='UTC'), inplace=True)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.year
    #  Secondly, determine categories with at least 10000 samples
    names = [f"{name}-bad" for name, _ in fixes_list] + [f"{name}-good" for name, _ in fixes_list]
    agg = df.groupby(['website'])[names].sum()  # , 'date', 'author'
    # agg = agg.loc[agg.sum(axis=1) > 0]
    return agg


def check_examples_of_errors(df_, website):
    # df_ has columns 'website', 'text' and 12 columns for error stats
    # print random errors for each found category
    names = [(f"{name}-bad", f"{name}-good" ) for name, _ in fixes_list]
    tmp = df_.loc[df_['website'] == website]
    for name_bad, name_good in names:
        print("#"*80, '\n'+name_bad)
        mask = tmp[name_bad] > 1
        n_b = mask.sum()
        print(f"total of {n_b} rows with this error")
        mask_g = tmp[name_bad] > 1
        n_g = mask_g.sum()
        print(f"total of {n_g} rows without this error")
        print(f'fraction of error is {tmp[name_bad].sum()/(tmp[name_bad].sum() + tmp[name_good].sum())}')
        print('\n'+'-'*80)

        print(("\n"+"-"*80+"\n").join(tmp.loc[mask, 'text'].sample(min(n_b, 10)).tolist()))


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


if __name__ == "__main__":
    filename = '2022-02-14-11-58.pickle'
    df = pd.read_pickle(filename)

    n0 = len(df)
    logger.info(f'We start with {n0} rows')
    df['len'] = df['text'].apply(len)
    mask = df['len'] >= 20

    df = df.loc[mask]
    n1 = len(df)
    logger.info(f'Filtering by length removed {n0-n1} rows')

    lit_frac = by_fraction_lithuanian(df['text'], df['len'])
    mask = lit_frac >= 0.98
    df = df.loc[mask]
    n2 = len(df)
    logger.info(f'Filtering by how lithuanian removed {n1-n2} rows more')

    space_frac = get_fraction_of_spaces_to_non_spaces(df['text'])
    mask = (space_frac >= 0.02)
    df = df.loc[mask]
    n3 = len(df)
    logger.info(f'Filtering by fraction of spaces to non spaces removed {n2-n3} rows even more')

    logger.info(f'Now we are left with {n3} rows. From initial only  {n3*100/n0::2.2f} % remains.')
    df.to_pickle("filtered_" + filename)



