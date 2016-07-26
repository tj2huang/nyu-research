__author__ = 'JasonLiu'

from functools import lru_cache
from operator import itemgetter

import re


class LabelGetter:
    alcohol = "alcohol"
    first_person = "first_person"
    first_person_level = "first_person_level"

    def __init__(self, X):
        self.X = X

    def get_flatlabels(self):
        labels = self.X["labels"]
        return self.X, labels.apply(self._flatten)

    def get_alcohol(self):
        """
        :return: X, y
        """
        return self._get_labels(self.alcohol)

    def get_first_person(self):
        """
        :return: X, y
        """
        return self._get_labels(self.first_person)

    def get_first_person_label(self):
        """
        :return: X, y
        """
        X, y = self._get_labels(self.first_person_level)
        y[y == 3] = 0 # Fixed the problem of heavy labels
        return X, y

    def _flatten(self, label_dict):
        """
        :param label_dict:
        :return:
        """
        if self.first_person_level in label_dict:
            return label_dict[self.first_person_level] + 2
        else:
            return label_dict[self.alcohol]

    def _get_labels(self, label_name, column_name='labels'):
        """
        :param label_name:
        :return: X, y with label_name
        """
        labels = self.X[column_name]
        if label_name == "flat":
            return self.get_flatlabels()
        has = labels.apply(self._contains(label_name))
        return self.X[has], labels[has].apply(itemgetter(label_name))

    @classmethod
    def _contains(cls, a):
        def c(b): return a in b

        return c

    def __repr__(self):
        return self.X.__repr__()

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
