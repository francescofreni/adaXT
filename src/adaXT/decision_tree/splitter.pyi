import numpy as np
from ..criteria import Criteria


class Splitter:
    """
    Splitter class used to create splits of the data.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, criteria: type[Criteria]) -> None:
        """
        Parameters
        ----------
            X : memoryview of NDArray
                The feature values used for splitting.
            Y : memoryview of NDArray
                The response values used for splitting.
            criteria : Criteria
                The criteria class used to find the impurity of a split.
        """
        pass

    def get_split(self, indices: np.ndarray, feature_indices: np.ndarray):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices for which to find a split.
        feature_indices : memoryview of NDArray
            Features at which to consider splitting.

        Returns
        -----------
        (list, double, int, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, and (5) the best impurity
        """
        pass

class Splitter_DG_base_v1:
    """
    Splitter class used to create splits of the data.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, E: np.ndarray,
                 all_idx: np.ndarray, best_preds: np.ndarray) -> None:
        """
        Parameters
        ----------
            X : memoryview of NDArray
                The feature values used for splitting.
            Y : memoryview of NDArray
                The response values used for splitting.
        """
        pass

    def get_split(self, indices: np.ndarray, feature_indices: np.ndarray, alpha: float):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices for which to find a split.
        feature_indices : memoryview of NDArray
            Features at which to consider splitting.

        Returns
        -----------
        (list, double, int, double, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, (5) the best impurity and (6) the
            worst environment
        """
        pass

class Splitter_DG_base_v2:
    """
    Splitter class used to create splits of the data.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, E: np.ndarray,
                 all_idx: np.ndarray, best_preds: np.ndarray) -> None:
        """
        Parameters
        ----------
            X : memoryview of NDArray
                The feature values used for splitting.
            Y : memoryview of NDArray
                The response values used for splitting.
        """
        pass

    def get_split(self, indices: np.ndarray, feature_indices: np.ndarray, alpha: float):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices for which to find a split.
        feature_indices : memoryview of NDArray
            Features at which to consider splitting.

        Returns
        -----------
        (list, double, int, double, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, (5) the best impurity and (6) the
            worst environment
        """
        pass

class Splitter_DG_fullopt:
    """
    Splitter class used to create splits of the data.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, E: np.ndarray,
                 all_idx: np.ndarray) -> None:
        """
        Parameters
        ----------
            X : memoryview of NDArray
                The feature values used for splitting.
            Y : memoryview of NDArray
                The response values used for splitting.
        """
        pass

    def get_split(self, indices: np.ndarray, feature_indices: np.ndarray, alpha: float,
                  nodes_indices: list, pos: int):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices for which to find a split.
        feature_indices : memoryview of NDArray
            Features at which to consider splitting.

        Returns
        -----------
        (list, double, int, double, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, (5) the best impurity and (6) the
            worst environment
        """
        pass

class Splitter_DG_adafullopt:
    """
    Splitter class used to create splits of the data.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, E: np.ndarray,
                 all_idx: np.ndarray) -> None:
        """
        Parameters
        ----------
            X : memoryview of NDArray
                The feature values used for splitting.
            Y : memoryview of NDArray
                The response values used for splitting.
        """
        pass

    def get_split(self, feature_indices: np.ndarray, alpha: float, nodes_indices: list, mask: list):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices for which to find a split.
        feature_indices : memoryview of NDArray
            Features at which to consider splitting.

        Returns
        -----------
        (list, double, int, double, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, (5) the best impurity and (6) the
            worst environment
        """
        pass
