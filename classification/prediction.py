import pandas as pd

class PredictionTransformer:

    cols = [
        'predict_alc',
        'predict_fpa',
        'predict_fpa|alc',
    ]

    def __init__(self, clf_alc, clf_fpa, clf_fpl):
        self.clf_alc = clf_alc
        self.clf_fpa = clf_fpa
        self.clf_fpl = clf_fpl

    def __call__(self, df, thres=0.75):
        self.df = df

        for col in self.cols:
            self.df[col] = 0

        self.thres = thres

        self._make_alcohol_predictions()
        self._make_firstperson_predictions()
        self._make_firstpersonlevel_predictions()

        return self.df


    def _make_alcohol_predictions(self):
        predictions_alc = self.clf_alc.predict_proba(self.df)
        self.df["predict_alc"] = predictions_alc[:,1]

    def _make_firstperson_predictions(self):
        filter_alc = self.df.predict_alc > self.thres

        # predict only on subset of the data, makes things way faster
        predict_fpa = self.clf_fpa.predict_proba(self.df[filter_alc])
        self.df.loc[filter_alc, "predict_fpa|alc"] = predict_fpa[:,1]

        # compute a marginal using the product rule
        self.df["predict_fpa"] = self.df["predict_alc"] * self.df["predict_fpa|alc"]

    def _make_firstpersonlevel_predictions(self):
        filter_alc = self.df.predict_alc > self.thres

        # predict only on subset of the data, makes things way faster
        predict_fpl = self.clf_fpl.predict_proba(self.df[filter_alc])

        # convert it to a named dataframe
        predict_fpl = pd.DataFrame(
            predict_fpl,
                 columns=[
                "predict_present|fpa",
                "predict_future|fpa",
                "predict_past|fpa"],
            index=self.df[filter_alc].index)

        marginal_firstperson = self.df[filter_alc]["predict_fpa"]

        # for each conditional level generate a marginal
        for col in predict_fpl.columns:
            col_marginal = col.split("|")[0]
            predict_fpl[col_marginal] = predict_fpl[col] * marginal_firstperson

        self.df = self.df.join(predict_fpl).fillna(0)