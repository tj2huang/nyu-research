import sys
import getopt
import pickle
import os
from multiprocessing import Pool

import pandas as pd
import numpy as np

from classification.prediction import PredictionTransformer
from hpc import postprocessing

alc_keywords = ['drink','drinker','drinks','drinking','drank','wine','champgne','alcohol','alcoholics',
                'alcoholism','beer','beers','bottle','bottles','pint','pints','cocktail','cocktails','bar','brewery',
                'lounge','pub','liquor','booze','vodka','tequila','gin','ciroc','margarita','margaritas','ale',
                'whiskey','lager','tipsy','drunk','sober','wasted','pregame','pregaming', 'johnnie walker', 'smirnoff',
                'hennessy', 'jack daniel\'s', 'bacardi', 'absolut', 'captain morgan', 'chivas regal', 'grey goose',
                'crown royal', 'ballantine\'s', 'baileys', 'jagermeister']

def prefilter(text):
    """
    Keyword matching
    :param text:
    :return:
    """
    text = str(text).lower()
    keywords = set(alc_keywords)
    return bool(set(text.split()).intersection(keywords))


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
        df['alc_key'] = df.text.apply(prefilter)
        alc = df[df.alc_key]
        not_alc = df[np.invert(df.alc_key)]
        predicted = clf(alc, thres=0.5)
        merged = not_alc.merge(predicted, how='outer')
        merged.to_csv(out_dir + '/' + file.split('/')[-1][:-4] + 'predict.csv')
    except Exception as e:
        print(file)
        print(e)


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
    clf_alc = pickle.load(open(dir_path + '/classifiers/clf_alc_simple.p', 'rb'))
    clf_fpa = pickle.load(open(dir_path + '/classifiers/clf_fpa_simple.p', 'rb'))
    clf_fpl = pickle.load(open(dir_path + '/classifiers/clf_fpl_simple.p', 'rb'))
    clf = PredictionTransformer(clf_alc, clf_fpa, clf_fpl)

    # parallel
    p = Pool(cores)
    dirs = [(clf, out_dir, infolder + '/' + f) for f in os.listdir(infolder)]
    p.map(predict, dirs)

if __name__ == "__main__":
    # main(sys.argv[1:])
    main(('C:/Users/Tom/Documents/nyu-test/alc-run/sept/in', 'C:/Users/Tom/Documents/nyu-test/alc-run/sept/out', 2))
    postprocessing.main(('C:/Users/Tom/Documents/nyu-test/alc-run/sept/out', 'C:/Users/Tom/Documents/nyu-test/alc-run/sept/summary-keyword', 2))
