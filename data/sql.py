import mysql.connector
import pandas as pd

from __private import dc_sql_connect

TWEETS_TABLE = 'uploaded_tweets'
PREDICTION_TABLE = 'predictions'
LABEL_TABLE = 'labels'


class DataAccess:
    def __init__(self):
        self.con = mysql.connector.connect(**dc_sql_connect)
        self.cur = self.con.cursor()
        self.cur.execute('SET NAMES utf8mb4')

    def insert_from_df(self, df):
        """
        Inserts all the rows of the dataframe into database without labels or prediction
        :param df: pd.DataFrame
        """
        _columns = ['text', 'id', 'created_at', 'lat', 'lon']
        query_vars = dict()
        for index, row in df.iterrows():
            for key in _columns:
                query_vars[key] = str(row[key]) if key in df else 'NULL'
            try:
                self.cur.execute('''INSERT INTO {} (text, tweet_id, date, lat, lon)
                 VALUES (%s,%s,%s,%s,%s)'''.format(TWEETS_TABLE), params=tuple([query_vars[key] for key in _columns]))
                self.con.commit()

            except mysql.connector.Error as e:
                print(e)
                self.con.rollback()

    def insert_fast(self, df):
        _columns = ['text', 'id', 'created_at', 'lat', 'lon']
        query_vars = dict()
        data = list()
        for index, row in df.iterrows():
            for key in _columns:
                query_vars[key] = str(row[key]) if key in df else ''
            data.append(tuple([query_vars[key] for key in _columns]))
        while data:
            try:
                insert_data = data[:500]
                data = data[500:]
                self.cur.executemany('''INSERT INTO {} (text, tweet_id, date, lat, lon)
                     VALUES (%s,%s,%s,%s,%s)'''.format(TWEETS_TABLE), insert_data)
                self.con.commit()

            except mysql.connector.Error as e:
                print(e)
                self.con.rollback()

    def read_as_df(self, start_date, end_date):
        """

        :param start_date: pd.datetime
        :param end_date: pd.datetime
        :param sample:
        :return:
        """
        self.cur.execute('''SELECT {0}.tweet_id, {0}.text, {0}.date, {0}.lat, {0}.lon, {0}.utc_offset,
        {1}.Alcohol, {1}.FirstPerson, {1}.Casual, {1}.Looking, {1}.Reflecting
        FROM {0} INNER JOIN {1} ON {0}.tweet_id = {1}.tweet_id WHERE date BETWEEN %s and %s
                         '''.format(TWEETS_TABLE, PREDICTION_TABLE), (str(start_date), str(end_date)))
        results = self.cur.fetchall()
        df = pd.DataFrame(results, columns=['id', 'text', 'created_at', 'lat', 'lon', 'utc_offset', 'predict_alc',
                                            'predict_fpa', 'predict_casual', 'predict_looking', 'predict_reflecting'])
        # print(df.shape)
        return df

    def save_prediction(self, df, tweet_id='tweet_id', alc='alcohol', fpa='first person', casual='casual',
                        looking='looking', reflecting='reflecting', description=''):
        """
        Adds predicted probabilities to the prediction table
        :param df: with columns ['tweet_id', 'alcohol','firstperson', 'casual', 'looking', 'reflecting'] containing
        their probability between 0 and 1 inclusive
        :param tweet_id: col name
        :param alc: col name
        :param fpa: col name
        :param casual: col name
        :param looking: col name
        :param reflecting: col name
        :param description: string description of where this prediction is from
        (e.g. logreg or whatever, maybe not needed)
        """
        _columns = [alc, fpa, casual, looking, reflecting]
        data = list()
        for index, row in df.iterrows():
            data.append((row[tweet_id],) + tuple([row[key] for key in _columns]) + (description,))
        while data:
            insert_data = data[:1000]
            data = data[1000:]
            try:
                self.cur.executemany('''INSERT INTO {} (tweet_id, Alcohol, FirstPerson,
                 Casual, Looking, Reflecting, description) VALUES (%s,%s,%s,%s,%s,%s,%s)'''.format(PREDICTION_TABLE),
                                 insert_data)
                self.con.commit()

            except mysql.connector.Error as e:
                print(e)
                self.con.rollback()

    # TODO: make this work directly with turk results
    def save_labels(self, df, tweet_id_col='tweet_id', label_col='labels'):
        """

        :param df:
        :param tweet_id_col:
        :param label_col: dict of labels
        :return:
        """

        for index, row in df.iterrows():
            try:
                self.cur.execute('''INSERT INTO {} (tweet_id, label) VALUES (%s,%s)'''.format(LABEL_TABLE),
                                 (row[tweet_id_col], row[label_col]))
                self.con.commit()

            except mysql.connector.Error as e:
                print(e)
                self.con.rollback()

    def update_from_df(self, df):
        _columns = ['lat', 'lon', 'utc_offset', 'id']
        query_vars = dict()
        data = list()
        for index, row in df.iterrows():
            for key in _columns:
                query_vars[key] = str(row[key]) if key in df else ''
            data.append(tuple([query_vars[key] for key in _columns]))
        while data:
            try:
                print('new batch')
                insert_data = data[:1000]
                data = data[1000:]
                self.cur.executemany('''UPDATE {} lat=%s, lon=%s, utc_offset=%s
                         WHERE tweet_id=%s'''.format(TWEETS_TABLE), insert_data)
                self.con.commit()

            except mysql.connector.Error as e:
                print(e)
                self.con.rollback()