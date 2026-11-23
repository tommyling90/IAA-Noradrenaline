"""SVM RBF après PCA — frontière non linéaire dans espace réduit"""
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


def get_model(n_components=20, C=1.0, gamma="scale", class_weight=None):
    pca  = PCA(n_components=n_components)
    svc  = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight)
    pipe = Pipeline([("pca", pca), ("svc", svc)])
    return CalibratedClassifierCV(pipe, cv=3, ensemble=False)