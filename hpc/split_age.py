import sys
import getopt
import os
from multiprocessing import Pool
import pickle

from age.age import setup, predict


def predict_wrap(args):
    clf = args[0]
    feature_list = args[1]
    out_dir = args[2]
    in_file = args[3]
    print(in_file)
    try:
        predict(clf, feature_list, in_file, out_dir)
    except Exception as e:
        print(e)


def main(argv):
    # help option
    try:
        opts, args = getopt.getopt(argv, 'h')
        for opt, arg in opts:
            if opt == '-h':
                print('''args: folder_of_data out_dir number_of_cores:
                clf_alc.p clf_fpa.p clf_fpl.p tweets_folder 8''')
                sys.exit(2)
    except getopt.GetoptError:
        print('-h for help')
        sys.exit(2)

    # run prediction
    in_folder, out_dir, cores = tuple(argv)
    cores = int(cores)

    # clf, feature_list = setup()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    clf = pickle.load(open(dir_path+'/age/age_svm.p', 'rb'))
    feature_list = pickle.load(open(dir_path+'/age/feature_list.p', 'rb'))

    # parallel
    print('trained')
    p = Pool(cores)
    p_args = [(clf, feature_list, out_dir, in_folder + '/' + f) for f in os.listdir(in_folder)]
    p.map(predict_wrap, p_args)

if __name__ == "__main__":
    main(sys.argv[1:])
