from sklearn.linear_model import Ridge

def build():
    return Ridge(), {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]}