import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import fbeta_score, roc_auc_score, make_scorer, RocCurveDisplay
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.model_selection import RepeatedKFold, cross_validate
import matplotlib.pyplot as plt


def _f05_score(precision: np.array, recall: np.array) -> np.array:
    """Computes F05 Score for given precision and recall"""
    return ((1 + 0.5**2) * (precision * recall)) / (0.5**2 * precision + recall)


class TestClassifier(BaseEstimator):
    """
    A custom class that allows us to perform same validation on different models
    """

    def __init__(self, classifier, y_test):
        self._classifier = classifier
        self._y_test = y_test

    def name(self):
        """
        Returns name of the passed classifier
        """
        return type(self._classifier).__name__

    # fit model
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train
        self._classifier.fit(self.X_train, self.y_train)
        return self

    def _predict(self, x_test):
        """
        Returns predictions made by the passed classifier
        """
        return self._classifier.predict(x_test)

    def _predict_proba(self, x_test):
        """
        Returns probability predictions made by the passed classifier
        """
        return self._classifier.predict_proba(x_test)

    def make_predictions(self, x_test) -> None:
        """
        Kind of a wrapper for 'predict' and 'predict_proba' methods
        """
        self._predictions = self._predict(x_test)
        self._prob_predictions = self._predict_proba(x_test)[:, 1]

    def _roc_auc_score(self) -> float:
        """
        Calculates ROC AUC
        """
        return roc_auc_score(self._y_test, self._prob_predictions)

    # Don't know if this method makes sense
    # I tried to get the best threshold maximizing F-0.5 score
    # We can use this threshold to make predictions on a new data later
    def _optimal_threshold(self):
        """
        Calculates an optimal threshold maximizing F-0.5 score
        """
        # calculate F-0.5 using precision and recall
        precision, recall, thresholds = pr_curve(self._y_test, self._prob_predictions)
        f05score = _f05_score(precision, recall)
        ix = np.nanargmax(f05score)  # index of max element
        # list containing the best threshold and corresponding F-0.5 score
        return [thresholds[ix], f05score[ix]]

    def test_results(self):
        """
        Returns results of all tests we do
        """
        return self._roc_auc_score(), self._optimal_threshold()

    def test_with_cv(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs cross validation on 2 metrics F-0.5 and AP
        """
        f05_scorer = make_scorer(fbeta_score, beta=0.5)
        cv = RepeatedKFold(n_repeats=5, n_splits=10, random_state=1)
        scores = cross_validate(
            self._classifier,
            self.X_train,
            self.y_train,
            scoring={"f05": f05_scorer, "ap": "average_precision"},
            cv=cv,
        )
        return np.mean(scores["test_f05"]), np.mean(scores["test_ap"])

    def plot_results(self) -> None:
        """
        Plots Confusion matrix, ROC and PR curves of passed classifier
        """
        # will use it for PR curve 'no skill' line
        no_skill = len(self._y_test[self._y_test == 1]) / len(self._y_test)

        # create figure and axes, set labels
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            "Confusion matrix, ROC and PR Curve of {}".format(self.name()), fontsize=18
        )
        ax1.set_title("Confusion Matrix", fontsize=14)
        ax2.set_title("ROC Curve", fontsize=14)
        ax3.set_title("PR Curve", fontsize=14)

        # plot all graphs
        ConfusionMatrixDisplay.from_predictions(self._y_test, self._predictions, ax=ax1)
        RocCurveDisplay.from_predictions(self._y_test, self._prob_predictions, ax=ax2)
        ax2.plot([0, 1], [0, 1], linestyle="--", label="No skill")
        PrecisionRecallDisplay.from_predictions(
            self._y_test, self._prob_predictions, ax=ax3
        )
        ax3.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No skill")

        plt.legend()
        plt.show()
