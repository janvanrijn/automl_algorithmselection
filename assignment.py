import pandas as pd

import sklearn.tree
import sklearn.ensemble

import typing


class AbsractMetaLearningModel(object):

    def fit(self, df_features: pd.DataFrame, df_performance: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the internal model on it.

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param df_performance: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        raise NotImplementedError("Abstract Method, please subclass")

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        raise NotImplementedError("Abstract Method, please subclass")

    def score(self, y: pd.DataFrame, y_hat: typing.List[str]) -> typing.List[str]:
        """
        Scores the how well the by the meta-model selected classifiers would have performed.

        :param y: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)

        :param y_hat: pd.DataFrame
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best

        :return: List[float]
        The performance of the by the meta-model selected classifiers (indicated in y_hat) per task
        """
        result = []
        for idx, classifier in enumerate(y_hat):
            result.append(y[classifier][idx])
        return result


class MetaLearningBestOnAverage(AbsractMetaLearningModel):

    def __init__(self, _):
        """
        Baseline method, that determines during fit time which method performs best on average. This method is selected
        always at test time.
        """
        self.best_model_name = None

    def fit(self, df_features: pd.DataFrame, df_performance: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the internal model on it.

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param df_performance: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        raise NotImplementedError('Please implement')

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        raise NotImplementedError('Please implement')


class MetaLearningRegressionBased(AbsractMetaLearningModel):

    def __init__(self, expected_models):
        """
        Baseline method, that determines during fit time which method performs best on average. This method is selected
        always at test time.
        """
        self.models = {
            model: sklearn.ensemble.RandomForestRegressor(random_state=0) for model in expected_models
        }

    def fit(self, df_features: pd.DataFrame, df_performance: pd.DataFrame) -> None:
        """
        Takes an input (meta-features) and trains the internal model on it.

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :param df_performance: pd.DataFrame
        a data frame with size (N, D), where all N rows represent a base-dataset, and all D columns represent the
        performance of a (base-)model on that specific dataset (predictive accuracy)
        """
        raise NotImplementedError('Please implement')

    def predict(self, df_features: pd.DataFrame) -> typing.List[str]:
        """
        Predicts for a set of datasets (expressed in meta-features) the performance per (base-)model

        :param df_features: pd.DataFrame
        a data frame with size (N, M), where all N rows represent a base-dataset, and all M columns represent a
        meta-feature, calculated over that specific dataset

        :return: List[str]
        A list of length N, where each item in the list represents the name of the (base-)model that is predicted to
        perform best
        """
        raise NotImplementedError('Please implement')
