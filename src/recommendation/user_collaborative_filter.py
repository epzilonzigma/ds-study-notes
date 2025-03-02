import numpy as np
import pandas as pd
from scipy.spatial import distance

class UserCollaborativeFilter:
    def __init__(
            self, 
            data: pd.DataFrame, 
            user_column: str, 
            item_column: str, 
            rating_column: str
        ):
        
        self.user_column_header = user_column
        self.item_column_header = item_column
        self.rating_column_header = rating_column
        self.data = data

    def fit(
            self, 
            similarity_measure: str = "cosine", 
        ):
    
        self.matrix = self.data.pivot(
            index = self.user_column_header, 
            columns = self.item_column_header, 
            values = self.rating_column_header
        )

        self.model_similarity = similarity_measure
        self.similarity_dictionary = self._calculate_similarity_dictionary(
            similarity_measure
        )

    def _calculate_similarity_dictionary(
            self, 
            similarity_measure: str = "cosine"
        ) -> dict:
        """
        3 similarity measures available:
        - cosine
        - Pearson correlation
        - Euclidean distance
        """

        self.user_list = list(
            self.data[self.user_column_header].unique()
        )

        similarity_dictionary = {}

        for user in self.user_list:
            user_list = self.user_list.copy()
            user_list.remove(user)

            user_similarities = {}

            for u in user_list:
                target_user_vector = self.matrix.loc[user].to_numpy()
                user_vector = self.matrix.loc[u].to_numpy()
                bad = ~np.logical_or(
                    np.isnan(target_user_vector), 
                    np.isnan(user_vector)
                )

                target_user_vector = np.compress(bad, target_user_vector)
                user_vector = np.compress(bad, user_vector)

                if similarity_measure.lower() == "cosine":
                    cosine_measure = distance.cosine(
                        target_user_vector, 
                        user_vector
                    )
                    similarity = float(cosine_measure)
                elif similarity_measure.lower() == "pearson":
                    corr_matrix = np.corrcoef(
                        target_user_vector, 
                        user_vector
                    )
                    similarity = float(corr_matrix[0, 1])
                elif similarity_measure.lower() == "euclidean":
                    cosine_measure = distance.euclidean(
                        target_user_vector, 
                        user_vector
                    )
                    similarity = float(cosine_measure)

                user_similarities[u] = similarity
            
            similarity_dictionary[user] = user_similarities
        
        return similarity_dictionary

    def predict(self, user: str, item: str) -> float:
        if user not in set(self.user_list):
            raise Exception
        
        item_ratings = self.data[
            (self.data[self.item_column_header] == item)
            & (self.data[self.user_column_header] != user)
        ]

        item_ratings["similarity"] = (
            item_ratings[self.user_column_header]
            .map(self.similarity_dictionary[user])
        )

        denominator = item_ratings["similarity"].sum()

        prediction = float(
            (
                item_ratings[self.rating_column_header] 
                * item_ratings["similarity"]
            ).sum() / denominator
        )

        return prediction