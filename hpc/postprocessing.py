import sys
import getopt
import os
from multiprocessing import Pool

import pandas as pd

from hpc.preprocessing import timezone

# sys.path.append('C:\\users\\tom work\\pycharmprojects\\nyu-twipsy')
# sys.path.append('C:\\users\\tom work\\documents\\twitterstream\\nyu-twipsy')


def adjust_timestamp(df):
    """
    df: pd.DataFrame with datetime index and utc_offset columns
    """
    us = df[(df.utc_offset <= -14400) & (df.utc_offset >= -28800)]
    us.index = us.index + us.utc_offset.apply(lambda x: pd.to_timedelta(x, unit='s'))
    return us


def fpl_summary(args):
    """
     Generates a csv with total tweet counts by (hour, day), for each class
    :param args: [in_dir, out_dir, file_name] as a list for a pool parameter
    :param data_path: path to csv
    :param out_dir: output directory for file name
    :param file_name: unique name for output file
    :return:
    """
    data_path, out_dir, file_name = args
    try:
        df = pd.read_csv(data_path, engine='python')
        df.index = pd.to_datetime(df.created_at, errors='coerce')

        # remove rows with invalid datetimes
        df = df[pd.notnull(df.index)]

        # create local_time column, requires "place" column
        df = timezone.convert2local(df)
        df.index = df.local_time

        # set prediction thresholds
        df['casual'] = (df.predict_present > 0.6) & (df.predict_alc > 0.99)
        df['looking'] = (df.predict_future > 0.6) & (df.predict_alc > 0.99)
        df['reflecting'] = (df.predict_past > 0.6) & (df.predict_alc > 0.99)
        df['alc'] = df.predict_alc > 0.99
        df['fpa'] = (df.predict_fpa > 0.75) & (df.predict_alc > 0.99)

        # new index
        df['hour'] = df.index.hour
        df['day'] = df.index.day

        # aggregate counts by day and hour
        df_summary = df[['casual', 'looking', 'reflecting', 'alc', 'fpa', 'day', 'hour']].groupby(['day', 'hour']).agg([sum, len])

        # output one summary file per column per input data file
        df_summary.casual.to_csv(out_dir + '/casual_' + file_name)
        df_summary.looking.to_csv(out_dir + '/looking_' + file_name)
        df_summary.reflecting.to_csv(out_dir + '/reflecting_' + file_name)
        df_summary.alc.to_csv(out_dir + '/alc_' + file_name)
        df_summary.fpa.to_csv(out_dir + '/fpa_' + file_name)

    except Exception as e:
        print(e)


def all_summary(in_folder, out_dir):
    """
    For each class, aggregates the (hour, day) counts across all temporary csvs into one summary file.
    :param in_folder: directory containing partial summaries of each file from fpl_summary function
    :param out_dir: directory of final summary (not ending in '/')
    :return:
    """

    casual = pd.DataFrame()
    looking = pd.DataFrame()
    reflecting = pd.DataFrame()
    alc = pd.DataFrame()
    fpa = pd.DataFrame()

    # append contents of each file into the correct dataframe
    for file in os.listdir(in_folder):
        try:
            df = pd.read_csv(in_folder + '/' + file, index_col=[0, 1])

            prefix = file[:3]
            if prefix == 'cas':
                casual = pd.concat([casual, df])
            elif prefix == 'loo':
                looking = pd.concat([looking, df])
            elif prefix == 'ref':
                reflecting = pd.concat([reflecting, df])
            elif prefix == 'alc':
                alc = pd.concat([alc, df])
            elif prefix == 'fpa':
                fpa = pd.concat([fpa, df])

        except Exception as e:
            print(e)

    # group by day, hour, then sum and output to a corresponding csv
    def group(df, out_name):
        df.groupby(level=[0, 1]).agg(sum).to_csv(out_dir + '/' + out_name + '.csv')

    group(casual, 'casual')
    group(looking, 'looking')
    group(reflecting, 'reflecting')
    group(alc, 'alc')
    group(fpa, 'fpa')
    group(casual, 'total')


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

    data_folder, out_dir, cores = tuple(argv)
    cores = int(cores)

    # parallel
    p = Pool(cores)

    # temporary per-data-file aggregation
    agg_dir = out_dir + '/sum_by_file'
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)
    args = [(data_folder + '/' + f, agg_dir, 'agg_' + str(i) + '.csv') for i, f in enumerate(os.listdir(data_folder))]
    p.map(fpl_summary, args)

    # combine all the aggreagations into a single file
    summary_dir = out_dir + '/summary'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    all_summary(agg_dir, summary_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
