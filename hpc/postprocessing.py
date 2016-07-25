import sys
import getopt
import os
from multiprocessing import Pool

import pandas as pd

# sys.path.append('C:\\users\\tom work\\pycharmprojects\\nyu-twipsy')
# sys.path.append('C:\\users\\tom work\\documents\\twitterstream\\nyu-twipsy')


def adjust_timestamp(df):
    """
    df:pd.DataFrame with datetime index and utc_offset columns
    """
    us = df[(df.utc_offset <= -14400) & (df.utc_offset >= -28800)]
    us.index = us.index + us.utc_offset.apply(lambda x: pd.to_timedelta(x, unit='s'))
    return us


def fpl_summary(args):
    in_dir, out_dir, file_name = args
    try:
        df = pd.read_csv(in_dir, engine='python')
        df.index = pd.to_datetime(df.created_at)
        df = adjust_timestamp(df)
        df['casual'] = df.predict_present > 0.6
        df['looking'] = df.predict_future > 0.6
        df['reflecting'] = df.predict_past > 0.6
        df['alc'] = df.predict_alc > 0.75
        df['fpa'] = df.predict_fpa > 0.75

        df_casual = df.casual.groupby([df.index.day, df.index.hour]).agg({'casual': sum, 'total': len})
        df_looking = df.looking.groupby([df.index.day, df.index.hour]).agg({'looking': sum, 'total': len})
        df_reflecting = df.reflecting.groupby([df.index.day, df.index.hour]).agg({'reflecting': sum, 'total': len})
        df_alc = df.alc.groupby([df.index.day, df.index.hour]).agg({'alc': sum, 'total': len})
        df_fpa = df.fpa.groupby([df.index.day, df.index.hour]).agg({'fpa': sum, 'total': len})

        df_casual.to_csv(out_dir + '/casual_' + file_name)
        df_looking.to_csv(out_dir + '/looking_' + file_name)
        df_reflecting.to_csv(out_dir + '/reflecting_' + file_name)
        df_alc.to_csv(out_dir + '/alc_' + file_name)
        df_fpa.to_csv(out_dir + '/fpa_' + file_name)

    except Exception as e:
        print(e)


def main(argv):
    # help option
    try:
        opts, args = getopt.getopt(argv, 'h')
        for opt, arg in opts:
            if opt == '-h':
                print('''args: infolder_of_data out_dir number_of_cores:
                tweets_folder output_folder 8''')
                sys.exit(2)
    except getopt.GetoptError:
        print('-h for help')
        sys.exit(2)

    # run prediction
    # print(argv)
    in_folder, out_dir, cores = tuple(argv)
    cores = int(cores)
    # print(os.getcwd())
    # os.chdir(cur_dir)

    # parallel
    p = Pool(cores)
    args = [(in_folder + '/' + f, out_dir, 'tweets_' + str(i) + '.csv') for i, f in enumerate(os.listdir(in_folder))]
    p.map(fpl_summary, args)
    # for args in dirs:
    #     predict(args)

if __name__ == "__main__":
    main(sys.argv[1:])
