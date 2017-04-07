import sys
import getopt
import pickle
import os
from multiprocessing import Pool

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import twokenize.twokenize as tokenizer

from pipelines.helpers import ItemGetter


def predict(args):
    """
    Makes predictions and stores them in csv files
    :param args: [clf, out_dir, file] as a list as an argument for pool function
    :param clf: Prediction Transformer object
    :param out_dir: directory of output to store prediction csvs
    :param file: csv file to make predictions on
    :return:
    """
    clf = args[0]
    out_dir = args[1]
    file = args[2]
    try:
        print(file)
        df = pd.read_csv(file, engine='python').dropna()
        predicted = clf(df)
        predicted.to_csv(out_dir + '/' + file.split('/')[-1][:-4] + 'predict.csv')
    except Exception as e:
        print(file)
        print(e)


def train(tob_data, fp_data, fpl_data):
    def make_classifier():
        clf = Pipeline([
            ("getter", ItemGetter("text")),
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression())])

        clf_params = {
            'clf__C': 200,
            'clf__dual': False,
            'clf__max_iter': 100,
            'clf__multi_class': 'ovr',
            'clf__penalty': 'l2',
            'tfidf__tokenizer':tokenizer.tokenize,
            'tfidf__ngram_range':(1, 3),
            'tfidf__max_features':200000
        }

        clf.set_params(**clf_params)
        return clf

    def fit_clf(clf, X):
        X_tweets = list(X['text'])
        X_labels = list(X['labels'])
        X_pairs = []
        for p in zip(X_labels, X_tweets):
            X_pairs.append(p)
        X_all_training = pd.DataFrame([e[1] for e in X_pairs], columns=['text'])
        y_all_training = [e[0] for e in X_pairs]
        clf.fit(X_all_training, y_all_training)
        return clf

    clf_tob = make_classifier()
    clf_fp = make_classifier()
    clf_fpl = make_classifier()

    X_tob = pd.read_csv(tob_data, names = ['labels', 'text'])
    clf_tob = fit_clf(clf_tob, X_tob)

    X_fp = pd.read_csv(fp_data, names = ['labels', 'text'])
    clf_fp = fit_clf(clf_fp, X_fp)

    X_fpl = pd.read_csv(fpl_data, names = ['labels', 'text'])
    clf_fpl = fit_clf(clf_fpl, X_fpl)

    clf = PredictionTransformer(clf_tob, clf_fp, clf_fpl)
    return clf


class PredictionTransformer:
    cols = [
        'predict_tob',
        'predict_fp',
        'predict_fp|tob',
    ]

    def __init__(self, clf_tob, clf_fp, clf_fpl):
        self.clf_tob = clf_tob
        self.clf_fp = clf_fp
        self.clf_fpl = clf_fpl

    def __call__(self, df, thres=0.2):
        self.df = df

        for col in self.cols:
            self.df[col] = 0

        self.thres = thres

        self._make_tob_predictions()
        self._make_firstperson_predictions()
        self._make_firstpersonlevel_predictions()

        return self.df

    def _make_tob_predictions(self):
        predictions_tob = self.clf_tob.predict_proba(self.df)
        self.df["predict_tob"] = predictions_tob[:,1]

    def _make_firstperson_predictions(self):
        filter_tob = self.df.predict_tob > self.thres

        # predict only on subset of the data, makes things way faster
        predict_fp = self.clf_tob.predict_proba(self.df[filter_tob])
        self.df.loc[filter_tob, "predict_fp|tob"] = predict_fp[:,1]

        # compute a marginal using the product rule
        self.df["predict_fp"] = self.df["predict_tob"] * self.df["predict_fp|tob"]

    def _make_firstpersonlevel_predictions(self):
        filter_tob = self.df.predict_tob > self.thres

        # predict only on subset of the data, makes things way faster
        predict_fpl = self.clf_fpl.predict_proba(self.df[filter_tob])

        # convert it to a named dataframe
        predict_fpl = pd.DataFrame(
            predict_fpl,
                 columns=[
                "predict_present|fp",
                "predict_not_present|fp"],
            index=self.df[filter_tob].index)

        marginal_firstperson = self.df[filter_tob]["predict_fp"]

        # for each conditional level generate a marginal
        for col in predict_fpl.columns:
            col_marginal = col.split("|")[0]
            predict_fpl[col_marginal] = predict_fpl[col] * marginal_firstperson

        self.df = self.df.join(predict_fpl).fillna(0)

def main(argv):
    # help option
    try:
        opts, args = getopt.getopt(argv, 'h')
        for opt, arg in opts:
            if opt == '-h':
                print('''args: 3 pickle files folder_of_data number_of_cores:
                clf_alc.p clf_fpa.p clf_fpl.p tweets_folder 8''')
                sys.exit(2)
    except getopt.GetoptError:
        print('-h for help')
        sys.exit(2)

    # run prediction
    infolder, out_dir, cores = tuple(argv)
    cores = int(cores)

    # hard coded classifier location, TODO: add as parameter
    dir_path = os.path.dirname(os.path.realpath(__file__))
    clf_alc = pickle.load(open(dir_path + '/classifiers/tob.p', 'rb'))
    clf_fpa = pickle.load(open(dir_path + '/classifiers/tob_fp.p', 'rb'))
    clf_fpl = pickle.load(open(dir_path + '/classifiers/tob_fpl.p', 'rb'))
    clf = PredictionTransformer(clf_alc, clf_fpa, clf_fpl)

    # clf = train(tob_data=dir_path + '/training_data/tob.csv', fp_data=dir_path + '/training_data/fp.csv',
    #             fpl_data=dir_path + '/training_data/fpl.csv')

    print('Training done')
    # parallel
    p = Pool(cores)
    dirs = [(clf, out_dir, infolder + '/' + f) for f in os.listdir(infolder)[1:]]
    p.map(predict, dirs)

if __name__ == "__main__":
    #main(sys.argv[1:])
    #main(('/Volumes/TOSHIBA EXT/Sept_March_2015/csv with offset/septtransfer/sept blocked', '/Users/tom/Documents/hpc-test/sept-out', 2))
    main(('/Volumes/TOSHIBA EXT/Sept_March_2015/csv with offset/june-small-offset', '/Users/tom/Documents/hpc-test/out', 2))
