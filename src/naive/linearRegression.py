"""
Implementation of OLS with numpy
"""

import numpy as np

class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        
    def fit(self, l: float = 0) -> None:
        self.l = l
        
        #remove nulls
        if self.y.ndim == 1:
            self.y = np.expand_dims(self.y, axis = 1)

        data = np.hstack((self.X, self.y))
        data_clean = data[~np.isnan(data).any(axis = 1)]
        self.X_train = data_clean[:, :-1]
        self.y_train = data_clean[:, -1]
        
        #get coefficients

        X_2 = self.X_train.T @ self.X_train
        X_y = self.X_train.T @ self.y_train
        l_i = self.l * np.eye(X_2.shape[0])

        self.coefficients = np.linalg.inv(X_2 + l_i) @ X_y
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(self.coefficients, np.ndarray):
            raise Exception("Need to fit the model first")

        pred = X @ self.coefficients
        return pred
    
    def rmse(self, X: np.ndarray, actual_y: np.ndarray) -> float:
        pred = self.predict(X)
        diff = (actual_y - pred) ** 2
        rmse = np.sqrt(diff.mean())
        return rmse

if __name__ == "__main__":
    SEED = 9999
    np.random.seed(SEED)
    X = np.random.randint(-20, 100, size=(20, 5))
    y = np.random.randint(10, 80, size=20)

    ols = LinearRegression(X, y)
    ols.fit()
    print(ols.coefficients)