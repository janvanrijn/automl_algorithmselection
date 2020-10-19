import pandas as pd
import unittest

from assignment import MetaLearningBestOnAverage, MetaLearningRegressionBased


class TestMetaModels(unittest.TestCase):

    def setUp(self) -> None:
        self.meta_features = [
            'MajorityClassSize',
            'MaxNominalAttDistinctValues',
            'MinorityClassSize',
            'NumberOfClasses',
            'NumberOfFeatures',
            'NumberOfInstances',
            'NumberOfInstancesWithMissingValues',
            'NumberOfMissingValues',
            'NumberOfNumericFeatures',
            'NumberOfSymbolicFeatures'
        ]
        self.classifier_features = [
            'AdaBoostClassifier(DecisionTreeClassifier)',
            'BernoulliNB',
            'DecisionTreeClassifier',
            'ExtraTreesClassifier',
            'GradientBoostingClassifier',
            'KNeighborsClassifier',
            'MLPClassifier',
            'RandomForestClassifier',
            'SGDClassifier',
            'SVC'
        ]

    def assertArrayAlmostEqual(self, results, fixture):
        for r, f in zip(results, fixture):
            self.assertAlmostEqual(r, f)

    def test_models(self):
        df_meta = pd.read_csv('metadata.csv')
        df_classifiers = df_meta[self.classifier_features]
        df_metaf = df_meta[self.meta_features]

        test_size = 10
        train_features = df_metaf[test_size:]
        train_results = df_classifiers[test_size:]
        test_features = df_metaf[:test_size].reset_index()
        del test_features['index']
        test_results = df_classifiers[:test_size].reset_index()
        del test_results['index']

        baseline = MetaLearningBestOnAverage(self.classifier_features)
        baseline.fit(train_features, train_results)
        predictions = baseline.predict(test_features)
        score_baseline = baseline.score(test_results, predictions)
        fixture_baseline = [0.9953066332916144, 0.9538, 0.9616, 0.976, 0.828, 0.9713876967095852, 0.9645, 0.747,
                            0.8045, 0.5369993211133739]
        self.assertArrayAlmostEqual(score_baseline, fixture_baseline)

        regression_based = MetaLearningRegressionBased(self.classifier_features)
        regression_based.fit(train_features, train_results)
        predictions = regression_based.predict(test_features)
        score_regression = regression_based.score(test_results, predictions)
        fixture_regresion = [0.9953066332916144, 0.9487, 0.7504000000000001, 0.9765, 0.828, 0.9713876967095852, 0.9645, 0.747, 0.8045, 0.5668703326544466]

        self.assertArrayAlmostEqual(score_regression, fixture_regresion)
