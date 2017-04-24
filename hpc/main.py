import sys
import getopt
import pickle
import os
from multiprocessing import Pool

import pandas as pd

from classification.prediction import PredictionTransformer
from hpc import postprocessing


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
        predicted = clf(df, thres=0.995)
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
    main(('C:/Users/Tom/Documents/nyu-test/alc-run/june/in', 'C:/Users/Tom/Documents/nyu-test/alc-run/june/out', 2))
    postprocessing.main(('C:/Users/Tom/Documents/nyu-test/alc-run/june/out', 'C:/Users/Tom/Documents/nyu-test/alc-run/june/summary', 2))
