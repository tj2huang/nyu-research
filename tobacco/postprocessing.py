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
    print(file_name)
    try:
        df = pd.read_csv(data_path, engine='python')
        df.index = pd.to_datetime(df.created_at, errors='coerce')

        # remove rows with invalid datetimes
        df = df[pd.notnull(df.index)]

        # create local_time column, requires "place" column
        # df = timezone.convert2local(df)
        # df.index = df.local_time
        df = adjust_timestamp(df)

        # set prediction thresholds
        df['present'] = (df.predict_present > 0.5)
        df['tob'] = df.predict_tob > 0.5
        df['fp'] = (df.predict_fp > 0.5)
        df['cur'] = df.predict_cur > 0.5
        df['com'] = df.predict_com > 0.5
        df['psh'] = ((df.predict_present > 0.5) & df.shisha).apply(int)

        # new index
        df['hour'] = df.index.hour
        df['day'] = df.index.day

        # aggregate counts by day and hour
        df_summary = df[['present', 'tob', 'fp', 'cur', 'com', 'psh', 'day', 'hour']].groupby(['day', 'hour']).agg([sum, len])

        # output one summary file per column per input data file
        df_summary.present.to_csv(out_dir + '/present_' + file_name)
        df_summary.tob.to_csv(out_dir + '/tob_' + file_name)
        df_summary.fp.to_csv(out_dir + '/fp_' + file_name)
        df_summary.cur.to_csv(out_dir + '/cur_' + file_name)
        df_summary.com.to_csv(out_dir + '/com_' + file_name)
        df_summary.psh.to_csv(out_dir + '/psh_' + file_name)

    except Exception as e:
        print(e)


def all_summary(in_folder, out_dir):
    """
    For each class, aggregates the (hour, day) counts across all temporary csvs into one summary file.
    :param in_folder: directory containing partial summaries of each file from fpl_summary function
    :param out_dir: directory of final summary (not ending in '/')
    :return:
    """

    present = pd.DataFrame()
    tob = pd.DataFrame()
    fp = pd.DataFrame()
    cur = pd.DataFrame()
    com = pd.DataFrame()
    psh = pd.DataFrame()

    # append contents of each file into the correct dataframe
    for file in os.listdir(in_folder):
        try:
            df = pd.read_csv(in_folder + '/' + file, index_col=[0, 1])

            prefix = file[:2]
            if prefix == 'pr':
                present = pd.concat([present, df])
            elif prefix == 'to':
                tob = pd.concat([tob, df])
            elif prefix == 'fp':
                fp = pd.concat([fp, df])
            elif prefix == 'cu':
                cur = pd.concat([cur, df])
            elif prefix == 'co':
                com = pd.concat([com, df])
            elif prefix == 'ps':
                psh = pd.concat([psh, df])

        except Exception as e:
            print(e)

    # group by day, hour, then sum and output to a corresponding csv
    def group(df, out_name):
        df.groupby(level=[0, 1]).agg(sum).to_csv(out_dir + '/' + out_name + '.csv')

    group(present, 'present')
    group(tob, 'tob')
    group(fp, 'fp')
    group(present, 'total')
    group(cur, 'current')
    group(com, 'combined')
    group(psh, 'shisha')


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
    # main(sys.argv[1:])
    main(
        ('C:/Users/Tom/Documents/nyu-test/tob-run/june/out', 'C:/Users/Tom/Documents/nyu-test/tob-run/june/summary', 2))

