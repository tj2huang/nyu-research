import sys
import getopt
import pickle
import os
from multiprocessing import Pool

import pandas as pd

from classification.prediction import PredictionTransformer


def predict(args):
    clf = args[0]
    out_dir = args[1]
    file = args[2]
    # print(file)
    try:
        df = pd.read_csv(file).dropna()
        predicted = clf(df)
        predicted.to_csv(out_dir + '/' + file.split('/')[-1][:-4] + 'predict.csv')
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
    s_clf_alc, s_clf_fpa, s_clf_fpl, folder, out_dir, cores = tuple(argv)
    cores = int(cores)
    clf_alc = pickle.load(open(s_clf_alc, 'rb'))
    clf_fpa = pickle.load(open(s_clf_fpa, 'rb'))
    clf_fpl = pickle.load(open(s_clf_fpl, 'rb'))
    clf = PredictionTransformer(clf_alc, clf_fpa, clf_fpl)

    # parallel
    p = Pool(cores)
    dirs = [(clf, out_dir, folder + '/' + f) for f in os.listdir(folder)]
    p.map(predict, dirs)

if __name__ == "__main__":
    main(sys.argv[1:])
