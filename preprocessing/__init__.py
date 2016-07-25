from functools import lru_cache

import re
import os
import pickle

import pandas as pd
import datetime

def make_batch(pickled_df, size, output_dir):
    """
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
