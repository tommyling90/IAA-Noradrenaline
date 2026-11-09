from sklearn.svm import SVR

def build():
    return SVR(kernel="rbf", cache_size=500), {
        "regressor__C": [0.1, 1.0, 10.0, 100.0],
        "regressor__gamma": ["scale", 0.001, 0.01, 0.1],
    }