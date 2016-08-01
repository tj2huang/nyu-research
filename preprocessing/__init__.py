from functools import lru_cache

import re
import os
import pickle

import pandas as pd
import datetime

alc_keywords = '''drink drinker drinks drinking drank wine champgne alcohol alcoholics alcoholism beer beers bottle bottles
    pint pints cocktail cocktails bar brewery lounge pub liquor booze vodka tequila gin ciroc margarita margaritas shot shots
    ale whiskey lager tipsy drunk sober wasted pregame pregaming'''


def re_filter(text):
    """
    Regular expression matching
    :param text: string
    :return: bool if any alcohol keywords are in the text
    """
    expression = alc_keywords.split().join('|')
    alcohol_filter = re.compile(expression)
    return bool(alcohol_filter.match(text))


def prefilter(text):
    """
    Keyword matching
    :param text:
    :return:
    """
    keywords = set(alc_keywords.split())
    return bool(set(text.split()).intersection(keywords))


def sample(in_folder, out_folder, block_size=100000, fraction=0.01):
    """
    Samples from csvs
    :param in_folder: directory of csvs
    :param out_folder:
    :param block_size: tweets per csv in output
    :param fraction: sampling fraction
    """
    tweets = pd.DataFrame()
    file_num = 0
    for file in os.listdir(in_folder):
        try:
            _df = pd.read_csv(in_folder + '/'  + file, encoding='utf8', engine='python')
            _df_sample = _df.sample(int(len(_df)*fraction))
            _df['id'].apply(str)
            tweets = pd.concat([tweets, _df_sample])

            if len(tweets) > block_size:
                tweets[:block_size].to_csv(out_folder + '/random_' + str(file_num) +'.csv', encoding='utf8')
                tweets = tweets[block_size:]
                file_num += 1
        except Exception as e:
            print (file)
            print (e)
    tweets.to_csv(out_folder + '/tweets_' + str(file_num) +'.csv')


def block(in_folder, out_folder, block_size=100000):
    """
    Combines csvs into equally sized chunks
    :param in_folder: directory of unevenly sized csvs
    :param out_folder:
    :param block_size:
    """
    tweets = pd.DataFrame()
    file_num = 0
    for file in os.listdir(in_folder):
        _df = pd.read_csv(in_folder + '/'  + file)[['id', 'created_at', 'utc_offset', 'text']]
        tweets = pd.concat([tweets, _df])
        while len(tweets) > block_size:
            tweets[:block_size].to_csv(out_folder + '/tweets_' + str(file_num) +'.csv', encoding='utf8')
            tweets = tweets[block_size:]
            file_num += 1
    tweets.to_csv(out_folder + '/tweets_' + str(file_num) +'.csv')


def make_batch(pickled_df, size, output_dir):
    """
    Creates csv to be labeled on AMT
    pickled_df: unlabeled data with 'random_number' column
    size: # of items batch to be created
    output_dir: where the outputs are saved
    """
    # Load data
    X = pickle.load(open(pickled_df, 'rb'))

    # select size subset
    batch = X.sort_values(by='random_number').head(size)

    # process for turk
    batch["text"] = batch.text.str.encode("utf-8")
    date = str(datetime.date.today())
    time = datetime.datetime.now().time()
    key = batch.iloc[0].random_number

    # persist
    description = "/c_amt {} {}h{}m {}".format(date, time.hour, time.minute, key)
    if not os.path.exists(output_dir + description):
        os.makedirs(output_dir + description)
    batch[["_id", "text", "random_number"]].to_csv(output_dir + description +  description + '.csv')
    pickle.dump(batch, open(output_dir + description + description +".p", 'wb'))
    X = X.drop(batch.index)
    pickle.dump(X, open(pickled_df, 'wb'))


class TurkResults2Label:
    """
    TurkResults2Label
    ~~~~~~~~~~~~~~~~~

    Usage:
        TurkResults2Label.parse_to_labels(string_labe: str)
    """

    first = re.compile("First")
    alch = re.compile("Alcohol Consumption")

    drinking_level = {
        "First Person - Alcohol": 0,
        "First Person - Alcohol::Casual Drinking": 0,
        "First Person - Alcohol::Looking to drink": 1,
        "First Person - Alcohol::Reflecting on drinking": 2,
        "First Person - Alcohol::Heavy Drinking": 0
    }  # Heavy Drinking is collasped into "drinking" since the label is unsuccessful

    related = {
        "Alcohol Related::Discussion": 0,
        "Alcohol Related::Promotional Content": 1
    }

    @classmethod
    @lru_cache(20)
    def parse_to_labels(cls, string_label):
        """
        :param string_label: amazon turk classification result
        :return: dictionary of classifications
        """
        label = {}
        if string_label == "Not Alcohol Related":
            label["alcohol"] = 0
            return label
        else:
            label["alcohol"] = 1

        if cls.alch.match(string_label) and not cls.first.match(string_label):
            return label

        if cls.first.match(string_label):
            label["first_person"] = 1
            label["first_person_level"] = cls.drinking_level[string_label]
            return label
        else:
            label["first_person"] = 0
            label["alcohol_related"] = cls.related[string_label]
        label["raw"] = string_label
        return label
