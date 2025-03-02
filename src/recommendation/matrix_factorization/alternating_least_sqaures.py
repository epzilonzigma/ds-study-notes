import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class ALS:
    """
    Alternating least square matrix factorization implementation base class.
    """
    def __init__(
            self, 
            n_features: int, 
            user_column_header: str = "User",
            item_column_header: str = "Item",
            rating_column_header: str = "Rating",
            user_regularization_param: float = 0.1,
            item_regularization_param: float = 0.1,
            max_iter: Optional[int] = 10,
            seed: Optional[int] = None
        ) -> None:
        self.feat_count = n_features
        self.lambda_1 = user_regularization_param
        self.lambda_2 = item_regularization_param

        self.user_col_header = user_column_header
        self.item_col_header = item_column_header
        self.rating_column_header = rating_column_header
        
        self.max_iter = max_iter
        if isinstance(seed, int):
            self.seed = seed
            np.random.seed = self.seed


    def _initialize(self):
        """
        Initialize user and item factor matrices
        """
        self.user_count, self.item_count = self.R.shape
        self.U_init = np.random.uniform(
            0,
            1/np.sqrt(self.feat_count),
            size = (self.user_count, self.feat_count)
        )

    def _wrangle_into_user_item_matrix(self):
        pass
    
    def _get_coefficients(self, X: np.ndarray, y: np.ndarray, l: float):
        if y.ndim == 1:
            y = np.expand_dims(y, axis = 1)
        dataset = np.hstack([X, y])
        dataset_cleaned = dataset[~np.isnan(dataset).any(axis = 1)]
        X_train = dataset_cleaned[:, :-1]
        y_train = dataset_cleaned[:, -1]

        X_2 = X_train.T @ X_train
        X_y = X_train.T @ y_train
        l_i = l * np.identity(X_2.shape[0])

        coefficients = np.linalg.inv(X_2 + l_i) @ X_y
        return coefficients
    
    def _rmse(
        self, 
        actual_matrix: np.ndarray, 
        prediction_matrix: np.ndarray
    ) -> float:
        """
        Calculate between to matrices
        """

        if actual_matrix.shape != prediction_matrix.shape:
            raise ValueError(
                f"Different matrix dimensions {actual_matrix.shape} vs {prediction_matrix.shape}"
            )
        
        difference_matrix = (actual_matrix - prediction_matrix) ** 2
        rmse = float(np.sqrt(np.nanmean(difference_matrix)))
        return rmse

    def fit(
        self, 
        rating_matrix: pd.DataFrame
    ):
        """
        Estimates factor matrices using rating_matrix
        """
        self.rating_df = rating_matrix[[
            self.user_col_header, 
            self.item_col_header,
            self.rating_column_header
        ]]

        self.R = self.rating_df.pivot(
            columns = self.item_col_header,
            index = self.user_col_header,
            values = self.rating_column_header
        ).to_numpy()
        self.R_T = self.R.T

        logger.info("Initializing user matrix")
        self._initialize()
        self.U = self.U_init
        i = 0

        logger.info("Start training")
        while i < self.max_iter:
            self.V = []

            for j in range(self.R.shape[1]): #estimate product matrix with user matrix
                dataset = np.hstack((self.U, np.expand_dims(self.R[:, j], axis = 1)))
                dataset = dataset[~np.isnan(dataset).any(axis = 1)] #remove any records with blanks
                X = dataset[:, :-1]
                y = dataset[:, -1]
                coefficients = self._get_coefficients(X, y, self.lambda_1)
                self.V.append(coefficients)

            self.V = np.array(self.V)
            self.U = []

            for j in range(self.R_T.shape[1]): #estimate user matrix with product matrix
                dataset = np.hstack((self.V, np.expand_dims(self.R_T[:, j], axis = 1)))
                dataset = dataset[~np.isnan(dataset).any(axis = 1)] #remove any records with blanks
                X = dataset[:, :-1]
                y = dataset[:, -1]
                coefficients = self._get_coefficients(X, y, 0.1)
                self.U.append(coefficients)

            self.U = np.array(self.U)

            prediction_matrix = self.U @ self.V.T
            pred_error = self._rmse(self.R, prediction_matrix)
            logger.info(f"iteration {i+1}: RMSE = {pred_error}")

            i += 1

    def predict_rating(
            self, 
            user_index: int,
            item_index: int
        ) -> float:
        """
            Returns rating estimate for given user index and product index
        """
        
        prediction = self.U[user_index] @ self.V[item_index].T
        return prediction.item()

    def predict_ratings_matrix(self) -> np.ndarray:
        """
            Returns full prediction matrix
        """
        ratings_matrix_prediction = self.U @ self.V.T
        return ratings_matrix_prediction

if __name__ == "__main__":
    ratings = pd.read_csv("../../data/movie_lens/rating.csv", nrows=900000)

    als = ALS(
        n_features = 100,
        user_column_header = "userId",
        item_column_header = "movieId",
        rating_column_header = "rating",
        max_iter = 20
    )

    als.fit(rating_matrix = ratings)

    print(f"rating matrix size: {als.R.shape}")
    print(f"fitted U size: {als.U.shape}")
    print(f"fitted V size: {als.V.shape}")
    print(f"prediction of 10th user on 145th movie: {als.predict_rating(10, 145)}")
    print(f"actual 10th user on 145th movie: {als.R[10, 145]}")
    
