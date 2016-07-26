from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer

# from pipelines.helpers import ItemGetter, ExplodingRecordJoiner
# from pipelines.user import UserAgeMonths, UserEgoVectorizer
# from pipelines.time import Timestamp2DatetimeIndex, DatetimeIndexAttr
# from pipelines.text import Gensim

__author__ = 'JasonLiu'


class AlcoholPipeline:
    def __init__(self):
        """
        :param time_features: default(["dayofweek", "hour", "hourofweek"])
        :param global_features: default(["text", "time", "user", "age"])
        :param lsi: if true, includes the TruncatedSVD() piece
        """
        self.global_features = {"text"}

    @property
    def _exploder(self):
        return ExplodingRecordJoiner(
            user=[
                'created_at',
                'favourites_count',
                'followers_count',
                'friends_count',
                'statuses_count',
                'verified'
            ]
        )

    def feature_textpipe(self):
        """
        :return: Pipeline(ItemGetter -> TfidfVectorizer)
        """
        textpipe = [
            ("getter", ItemGetter("text")),
            ("tfidf", TfidfVectorizer()),
        ]
        return Pipeline(textpipe)

    def pipeline(self, clf):
        """
        :param clf: sklearn.classifer
        :return: Pipeline([
            ("exploder", exploder),
            ("features", features),
        ])
        """
        pipeline = [
            #("exploder", self._exploder),
            ("features", self.feature_textpipe()),
            ("scaler", Normalizer()),
            ("clf", clf)
        ]
        return Pipeline(pipeline)
