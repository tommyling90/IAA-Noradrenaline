"""SVM linéaire — modèle de référence (Levy et al. 2020)"""
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def get_model(C=1.0, class_weight=None):
    base = LinearSVC(C=C, class_weight=class_weight, max_iter=2000)
    return CalibratedClassifierCV(base, cv=3, ensemble=False)