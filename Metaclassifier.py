from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from scipy.stats import ttest_rel
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn import datasets, naive_bayes
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from scipy.stats import ttest_ind
from sklearn.base import ClassifierMixin, BaseEstimator


class SamplingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, base_preprocesing=None):
        self.base_estimator = base_estimator
        self.base_preprocesing = base_preprocesing

    def fit(self, X, y):
        if self.base_preprocesing != None:
            preproc = clone(self.base_preprocesing)
            X_new, y_new = preproc.fit_resample(X, y)
            self.clf = clone(self.base_estimator)
            self.clf.fit(X_new, y_new)
            return self
        else:
            self.clf = clone(self.base_estimator)
            self.clf.fit(X, y)
            return self

    def predict(self, X):
        prediction = self.clf.predict(X)
        return prediction
