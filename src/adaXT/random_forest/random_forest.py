from typing import Literal
from numpy import int32 as INT

import numpy as np
import cvxpy as cp
from numpy.random import Generator, default_rng
import ctypes
from multiprocessing import RawArray

from adaXT import parallel
from adaXT.parallel import ParallelModel, shared_numpy_array

from numpy.typing import ArrayLike

from ..criteria import Criteria
from ..decision_tree import DecisionTree
from ..decision_tree.splitter import (Splitter, Splitter_DG_base_v1,
                                      Splitter_DG_base_v2, Splitter_DG_fullopt,
                                      Splitter_DG_adafullopt)
from ..base_model import BaseModel
from ..predictor import Predictor
from ..predictor.predictor import predict_default
from ..leaf_builder import LeafBuilder, LeafBuilder_DG

from collections import defaultdict


def tree_based_weights(
    tree: DecisionTree,
    X0: np.ndarray | None,
    X1: np.ndarray | None,
    size_X0: int,
    size_X1: int,
    scaling: str,
) -> np.ndarray:
    hash0 = tree.predict_leaf(X=X0)
    hash1 = tree.predict_leaf(X=X1)
    return tree._tree_based_weights(
        hash0=hash0,
        hash1=hash1,
        size_X0=size_X0,
        size_X1=size_X1,
        scaling=scaling,
    )


def get_sample_indices(
    gen: Generator,
    X_n_rows: int,
    sampling_args: dict,
    sampling: str | None,
) -> tuple:
    """
    Assumes there has been a previous call to self.__get_sample_indices on the
    RandomForest.
    """
    if sampling == "resampling":
        ret = (
            gen.choice(
                np.arange(0, X_n_rows),
                size=sampling_args["size"],
                replace=sampling_args["replace"],
            ),
            None,
        )
    elif sampling == "honest_tree":
        indices = np.arange(0, X_n_rows)
        gen.shuffle(indices)
        if sampling_args["replace"]:
            resample_size0 = sampling_args["size"]
            resample_size1 = sampling_args["size"]
        else:
            resample_size0 = np.min(
                [sampling_args["split"], sampling_args["size"]])
            resample_size1 = np.min(
                [X_n_rows - sampling_args["split"], sampling_args["size"]]
            )
        fit_indices = gen.choice(
            indices[: sampling_args["split"]],
            size=resample_size0,
            replace=sampling_args["replace"],
        )
        pred_indices = gen.choice(
            indices[sampling_args["split"]:],
            size=resample_size1,
            replace=sampling_args["replace"],
        )
        ret = (fit_indices, pred_indices)
    elif sampling == "honest_forest":
        indices = np.arange(0, X_n_rows)
        if sampling_args["replace"]:
            resample_size0 = sampling_args["size"]
            resample_size1 = sampling_args["size"]
        else:
            resample_size0 = np.min(
                [sampling_args["split"], sampling_args["size"]])
            resample_size1 = np.min(
                [X_n_rows - sampling_args["split"], sampling_args["size"]]
            )
        fit_indices = gen.choice(
            indices[: sampling_args["split"]],
            size=resample_size0,
            replace=sampling_args["replace"],
        )
        pred_indices = gen.choice(
            indices[sampling_args["split"]:],
            size=resample_size1,
            replace=sampling_args["replace"],
        )
        ret = (fit_indices, pred_indices)
    else:
        ret = (np.arange(0, X_n_rows), None)

    if sampling_args["OOB"]:
        # Only fitting indices
        if ret[1] is None:
            picked = ret[0]
        else:
            picked = np.concatenate(ret[0], ret[1])
        out_of_bag = np.setdiff1d(np.arange(0, X_n_rows), picked)
    else:
        out_of_bag = None

    return (*ret, out_of_bag)


def build_single_tree(
    fitting_indices: np.ndarray | None,
    prediction_indices: np.ndarray | None,
    X: np.ndarray,
    Y: np.ndarray,
    honest_tree: bool,
    criteria: type[Criteria],
    predictor: type[Predictor],
    leaf_builder: type[LeafBuilder] | type[LeafBuilder_DG],
    splitter: type[Splitter] | type[Splitter_DG_base_v1] |
              type[Splitter_DG_base_v2] | type[Splitter_DG_fullopt] | type[Splitter_DG_adafullopt],
    tree_type: str | None = None,
    max_depth: int = (2**31 - 1),
    impurity_tol: float = 0.0,
    min_samples_split: int = 1,
    min_samples_leaf: int = 1,
    min_improvement: float = 0.0,
    max_features: int | float | Literal["sqrt", "log2"] | None = None,
    skip_check_input: bool = True,
    sample_weight: np.ndarray | None = None,
    E: np.ndarray | None = None,
) -> DecisionTree:
    # subset the feature indices
    tree = DecisionTree(
        tree_type=tree_type,
        max_depth=max_depth,
        impurity_tol=impurity_tol,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_improvement=min_improvement,
        max_features=max_features,
        skip_check_input=skip_check_input,
        criteria=criteria,
        leaf_builder=leaf_builder,
        predictor=predictor,
        splitter=splitter,
    )
    if (tree_type == "MinMaxRegression") and E is None:
        raise ValueError("E is required for MinMaxRegression.")
    if (tree_type != "MinMaxRegression") and E is not None:
        raise ValueError("E is only supported for MinMaxRegression.")
    if tree_type != "MinMaxRegression":
        tree.fit(
            X=X,
            Y=Y,
            sample_indices=fitting_indices,
            sample_weight=sample_weight)
    else:
        tree.fit(
            X=X,
            Y=Y,
            E=E,
            sample_indices=fitting_indices,
            sample_weight=sample_weight)
    if honest_tree:
        tree.refit_leaf_nodes(
            X=X,
            Y=Y,
            sample_weight=sample_weight,
            sample_indices=prediction_indices)

    return tree


def oob_calculation(
    idx: np.int64,
    trees: list,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    parallel: ParallelModel,
    predictor: type[Predictor],
) -> tuple:
    X_pred = np.expand_dims(X_train[idx], axis=0)
    Y_pred = predictor.forest_predict(
        X_train=X_train,
        Y_train=Y_train,
        X_pred=X_pred,
        trees=trees,
        n_jobs=1,
        parallel=parallel,
    ).astype(np.float64)
    Y_true = Y_train[idx]
    return (Y_pred, Y_true)


class RandomForest(BaseModel):
    """
    Attributes
    ----------
    max_features: int | float | Literal["sqrt", "log2"] | None = None
        The number of features to consider when looking for a split.
    max_depth : int
        The maximum depth of the tree.
    forest_type : str
        The type of random forest, either  a string specifying a supported type
        (currently "Regression", "Classification", "Quantile" or "Gradient").
    n_estimators : int
        The number of trees in the random forest.
    n_jobs : int | tuple[int, int]
        The number of jobs used to fit and predict. If tuple, then different
        between the two
    sampling: str | None
        Either resampling, honest_tree, honest_forest or None.
    sampling_args: dict | None
        A parameter used to control the behavior of the sampling scheme. The following arguments
        are available:
            'size': Either int or float used by all sampling schemes (default 1.0).
                Specifies the number of samples drawn. If int it corresponds
                to the number of random resamples. If float it corresponds to the relative
                size with respect to the training sample size.
            'replace': Bool used by all sampling schemes (default True).
                If True resamples are drawn with replacement otherwise without replacement.
            'split': Either int or float used by the honest splitting schemes (default 0.5).
                Specifies how to divide the sample into fitting and prediction indices.
                If int it corresponds to the size of the fitting indices, while the remaining indices are
                used as prediction indices (truncated if value is too large). If float it
                corresponds to the relative size of the fitting indices, while the remaining
                indices are used as prediction indices (truncated if value is too large).
            'OOB': Bool used by all sampling schemes (default False).
                Computes the out of bag error given the data set.
                If set to True, an attribute called oob will be defined after
                fitting, which will have the out of bag error given by the
                Criteria loss function.
        If None all parameters are set to their defaults.
    impurity_tol : float
        The tolerance of impurity in a leaf node.
    min_samples_split : int
        The minimum number of samples in a split.
    min_samples_leaf : int
        The minimum number of samples in a leaf node.
    min_improvement: float
        The minimum improvement gained from performing a split.
    """

    def __init__(
        self,
        forest_type: str | None,
        n_estimators: int = 100,
        n_jobs: int | tuple[int, int] = 1,
        sampling: str | None = "resampling",
        sampling_args: dict | None = None,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        max_depth: int = (2**31 - 1),
        impurity_tol: float = 0.0,
        min_samples_split: int = 1,
        min_samples_leaf: int = 1,
        min_improvement: float = 0.0,
        seed: int | None = None,
        criteria: type[Criteria] | None = None,
        leaf_builder: type[LeafBuilder] | type[LeafBuilder_DG] | None = None,
        predictor: type[Predictor] | None = None,
        splitter: type[Splitter] | type[Splitter_DG_base_v1] | type[Splitter_DG_base_v2] |
                  type[Splitter_DG_fullopt] | type[Splitter_DG_adafullopt] | None = None,
        minmax_method: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        forest_type : str
            The type of random forest, either  a string specifying a supported type
            (currently "Regression", "Classification", "Quantile", "Gradient",
             "MinMaxRegression").
        n_estimators : int
            The number of trees in the random forest.
        n_jobs : int
            The number of processes used to fit, and predict for the forest, -1
            uses all available processors.
        sampling : str | None
            Either resampling, honest_tree, honest_forest or None.
        sampling_args : dict | None
            A parameter used to control the behavior of the sampling scheme. The following arguments
            are available:
                'size': Either int or float used by all sampling schemes (default 1.0).
                    Specifies the number of samples drawn. If int it corresponds
                    to the number of random resamples. If float it corresponds to the relative
                    size with respect to the training sample size.
                'replace': Bool used by all sampling schemes (default True).
                    If True resamples are drawn with replacement otherwise without replacement.
                'split': Either int or float used by the honest splitting schemes (default 0.5).
                    Specifies how to divide the sample into fitting and prediction indices.
                    If int it corresponds to the size of the fitting indices, while the remaining indices are
                    used as prediction indices (truncated if value is too large). If float it
                    corresponds to the relative size of the fitting indices, while the remaining
                    indices are used as prediction indices (truncated if value is too large).
            If None all parameters are set to their defaults.
        max_features : int | float | Literal["sqrt", "log2"] | None = None
            The number of features to consider when looking for a split.
        max_depth : int
            The maximum depth of the tree.
        impurity_tol : float
            The tolerance of impurity in a leaf node.
        min_samples_split : int
            The minimum number of samples in a split.
        min_samples_leaf : int
            The minimum number of samples in a leaf node.
        min_improvement : float
            The minimum improvement gained from performing a split.
        seed: int | None
            Seed used to reproduce a RandomForest
        criteria : Criteria
            The Criteria class to use, if None it defaults to the forest_type
            default.
        leaf_builder : LeafBuilder
            The LeafBuilder class to use, if None it defaults to the forest_type
            default.
        predictor : Predictor
            The Prediction class to use, if None it defaults to the forest_type
            default.
        splitter : Splitter | None
            The Splitter class to use, if None it defaults to the default
            Splitter class.
        minmax_method: str | None
            Method to use with MinMaxRegression.
            Accepted values are {"base", "fullopt", "adafullopt"}.
        """

        self.impurity_tol = impurity_tol
        self.max_features = max_features
        self.forest_type = forest_type
        self.n_estimators = n_estimators
        self.sampling = sampling
        self.sampling_args = sampling_args
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement

        self.forest_type = forest_type
        self.criteria = criteria
        self.splitter = splitter
        self.leaf_builder = leaf_builder
        self.predictor = predictor

        self.n_jobs = n_jobs

        self.seed = seed

        self.minmax_method = minmax_method

    def __get_random_generator(self, seed) -> Generator:
        if isinstance(seed, int) or (seed is None):
            return default_rng(seed)
        else:
            raise ValueError("Random state either has to be Integral or None")

    def __get_sampling_parameter(self, sampling_args: dict | None) -> dict:
        if sampling_args is None:
            sampling_args = {}

        if self.sampling == "resampling":
            if "size" not in sampling_args:
                sampling_args["size"] = self.X_n_rows
            elif isinstance(sampling_args["size"], float):
                sampling_args["size"] = int(
                    sampling_args["size"] * self.X_n_rows)
            elif not isinstance(sampling_args["size"], int):
                raise ValueError(
                    "The provided sampling_args['size'] is not an integer or float as required."
                )
            if "replace" not in sampling_args:
                sampling_args["replace"] = True
            elif not isinstance(sampling_args["replace"], bool):
                raise ValueError(
                    "The provided sampling_args['replace'] is not a bool as required."
                )
        elif self.sampling in ["honest_tree", "honest_forest"]:
            if "split" not in sampling_args:
                sampling_args["split"] = np.min(
                    [int(0.5 * self.X_n_rows), self.X_n_rows - 1]
                )
            elif isinstance(sampling_args["size"], float):
                sampling_args["split"] = np.min(
                    [int(sampling_args["split"] * self.X_n_rows), self.X_n_rows - 1]
                )
            elif not isinstance(sampling_args["size"], int):
                raise ValueError(
                    "The provided sampling_args['split'] is not an integer or float as required."
                )
            if "size" not in sampling_args:
                sampling_args["size"] = sampling_args["split"]
            elif isinstance(sampling_args["size"], float):
                sampling_args["size"] = int(
                    sampling_args["size"] * sampling_args["split"]
                )
            elif not isinstance(sampling_args["size"], int):
                raise ValueError(
                    "The provided sampling_args['size'] is not an integer or float as required."
                )
            if "replace" not in sampling_args:
                sampling_args["replace"] = True
            elif not isinstance(sampling_args["replace"], bool):
                raise ValueError(
                    "The provided sampling_args['replace'] is not a bool as required."
                )
        elif self.sampling is not None:
            raise ValueError(
                f"The provided sampling scheme '{self.sampling}' does not exist."
            )

        if "OOB" not in sampling_args:
            sampling_args["OOB"] = False
        elif not isinstance(sampling_args["OOB"], bool):
            raise ValueError(
                "The provided sampling_args['OOB'] is not a bool as required."
            )

        return sampling_args

    def __is_honest(self) -> bool:
        return self.sampling in ["honest_tree", "honest_forest"]

    # Function to build all the trees of the forest, differentiates between
    # running in parallel and sequential

    def __build_trees(self) -> None:
        # parent_rng.spawn() spawns random generators that children can use
        indices = self.parallel.async_map(
            get_sample_indices,
            map_input=self.parent_rng.spawn(self.n_estimators),
            sampling_args=self.sampling_args,
            X_n_rows=self.X_n_rows,
            n_jobs=self.n_jobs_fit,
            sampling=self.sampling,
        )
        self.fitting_indices, self.prediction_indices, self.out_of_bag_indices = zip(
            *indices)
        self.trees = self.parallel.starmap(
            build_single_tree,
            map_input=zip(self.fitting_indices, self.prediction_indices),
            X=self.X,
            Y=self.Y,
            honest_tree=self.__is_honest(),
            criteria=self.criteria,
            predictor=self.predictor,
            leaf_builder=self.leaf_builder,
            splitter=self.splitter,
            tree_type=self.forest_type,
            max_depth=self.max_depth,
            impurity_tol=self.impurity_tol,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_improvement=self.min_improvement,
            max_features=self.max_features,
            skip_check_input=True,
            sample_weight=self.sample_weight,
            E=self.E,
            n_jobs=self.n_jobs_fit,
        )

    def fit(self, X: ArrayLike, Y: ArrayLike,
            E: ArrayLike | None = None,
            sample_weight: ArrayLike | None = None) -> None:
        """
        Fit the random forest with training data (X, Y).

        Parameters
        ----------
        X : array-like object of dimension 2
            The feature values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        Y : array-like object
            The response values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        E : array-like object
            The environment labels used for training.
        sample_weight : np.ndarray | None
            Sample weights. Currently not implemented.
        """
        if (self.forest_type == "MinMaxRegression") and E is None:
            raise ValueError("E is required for MinMaxRegression.")
        if (self.forest_type != "MinMaxRegression") and E is not None:
            raise ValueError("E is only supported for MinMaxRegression.")

        if self.minmax_method is not None and self.forest_type != "MinMaxRegression":
            raise ValueError(f"{self.forest_type} only supports minmax_method=None.")

        # Initialization for the random forest
        # Can not be done in __init__ to conform with scikit-learn GridSearchCV
        self._check_tree_type(
            self.forest_type,
            self.criteria,
            self.splitter,
            self.leaf_builder,
            self.predictor,
            self.minmax_method,
        )
        self.parallel = ParallelModel()
        self.parent_rng = self.__get_random_generator(self.seed)

        # Check input
        X, Y = self._check_input(X, Y)
        self.X = shared_numpy_array(X)
        self.Y = shared_numpy_array(Y)
        if E is not None:
            E = np.ascontiguousarray(E, dtype=INT)
            row = E.shape[0]
            shared_E = RawArray(ctypes.c_int, row)
            shared_E_np = np.ndarray(
                shape=row, dtype=INT, buffer=shared_E
            )
            np.copyto(shared_E_np, E)
            self.E = shared_E_np
        else:
            self.E = None
        self.X_n_rows, self.n_features = self.X.shape
        self.max_features = self._check_max_features(
            self.max_features, X.shape[1])
        self.sample_weight = self._check_sample_weight(sample_weight)
        self.sampling_args = self.__get_sampling_parameter(self.sampling_args)

        # Check n_jobs
        if isinstance(self.n_jobs, tuple):
            self.n_jobs_fit = self.n_jobs[0]
            self.n_jobs_pred = self.n_jobs[1]
        elif isinstance(self.n_jobs, int):
            self.n_jobs_fit = self.n_jobs
            self.n_jobs_pred = self.n_jobs
        else:
            raise ValueError("n_jobs is neither a tuple or int")

        # Fit trees
        self.__build_trees()
        self.forest_fitted = True

        if self.sampling_args["OOB"]:
            # Dict, but creates empty list instead of keyerror
            tree_dict = defaultdict(list)

            # Compute a dictionary, where every key is an index, which is out of
            # bag for at least one tree. Each value is a list of the indices for
            # trees, which said value is out of bag for.
            for idx, array in enumerate(self.out_of_bag_indices):
                for num in array:
                    tree_dict[num].append(self.trees[idx])

            # Expand dimensions as Y will always only be predicted on a single
            # value. Thus when we combine them in this list, we will be missing
            # a dimension
            Y_pred, Y_true = (
                np.expand_dims(np.array(x).flatten(), axis=-1)
                for x in zip(
                    *self.parallel.async_starmap(
                        oob_calculation,
                        map_input=tree_dict.items(),
                        X_train=self.X,
                        Y_train=self.Y,
                        parallel=self.parallel,
                        predictor=self.predictor,
                        n_jobs=self.n_jobs_pred,
                    )
                )
            )

            # sanity check
            if Y_pred.shape != Y_true.shape:
                raise ValueError(
                    "Shape of predicted Y and true Y in oob oob_calculation does not match up!"
                )
            self.oob = self.criteria.loss(
                Y_pred, Y_true, np.ones(Y_pred.shape[0], dtype=np.double)
            )

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        """
        Predicts response values at X using fitted random forest.  The behavior
        of this function is determined by the Prediction class used in the
        decision tree. For currently existing tree types the corresponding
        behavior is as follows:

        Classification:
        ----------
        Returns the class based on majority vote among the trees. In the case
        of tie, the lowest class with the maximum number of votes is returned.

        Regression:
        ----------
        Returns the average response among all trees.

        Quantile:
        ----------
        Returns the conditional quantile of the response, where the quantile is
        specified by passing a list of quantiles via the `quantile` parameter.


        Parameters
        ----------
        X : array-like object of dimension 2
            New samples at which to predict the response. Internally it will be
            converted to np.ndarray with dtype=np.float64.

        Returns
        -------
        np.ndarray
            (N, K) numpy array with the prediction, where K depends on the
            Prediction class and is generally 1

        """
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call predict"
            )

        X, _ = self._check_input(X)
        self._check_dimensions(X)

        predict_value = shared_numpy_array(X)
        prediction = self.predictor.forest_predict(
            X_train=self.X,
            Y_train=self.Y,
            X_pred=predict_value,
            trees=self.trees,
            parallel=self.parallel,
            n_jobs=self.n_jobs_pred,
            **kwargs,
        )
        return prediction

    def refine_weights(
        self, X_val: ArrayLike, Y_val: ArrayLike, E_val: ArrayLike, X: ArrayLike, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refines the weights of the trees in the forest to reduce the maximum
        error over the training environments.

        At the moment, this only works for Regression and MinMaxRegression.
        """
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call predict"
            )

        X, _ = self._check_input(X)
        self._check_dimensions(X)

        X_val, _ = self._check_input(X_val)
        self._check_dimensions(X_val)

        Y_val, _ = self._check_input(Y_val)
        self._check_dimensions(Y_val)

        X_val = shared_numpy_array(X_val)
        Y_val = shared_numpy_array(Y_val)
        X = shared_numpy_array(X)

        E_val = np.ascontiguousarray(E_val, dtype=np.int64)
        E_val = np.expand_dims(E_val, axis=1)
        row, col = E_val.shape
        shared_E = RawArray(ctypes.c_int64, (row * col))
        shared_E_np = np.ndarray(
            shape=(row, col), dtype=np.int64, buffer=shared_E
        )
        np.copyto(shared_E_np, E_val)
        E_val = shared_E_np

        weights_minmax = self.predictor.refine_forest(
            X_val=X_val,
            Y_val=Y_val,
            E_val=E_val,
            trees=self.trees,
            parallel=self.parallel,
            n_jobs=self.n_jobs_pred,
            **kwargs,
        )

        predictions = self.parallel.async_map(
            predict_default,
            self.trees,
            X_pred=X,
            n_jobs=self.n_jobs,
        )
        predictions = np.array(predictions).T
        weighted_predictions = predictions @ weights_minmax.value
        return weighted_predictions, weights_minmax

    def modify_predictions_trees(
        self, E: ArrayLike,
    ) -> None:
        if self.forest_type not in ["Regression", "MinMaxRegression"]:
            raise ValueError("modify_predictions only works for Regression and MinMaxRegression")
        unique_envs = np.unique(E)

        for i, tree in enumerate(self.trees):
            indices = self.fitting_indices[i]
            E_sample = E[indices]

            leaves = tree.leaf_nodes
            n_leaves = len(leaves)

            c = cp.Variable(n_leaves)
            t = cp.Variable(nonneg=True)

            constraints = []

            for env in unique_envs:
                expr = 0
                n_env = np.sum(E_sample == env)
                for j, leaf in enumerate(leaves):
                    leaf_idxs = leaf.indices
                    Y_leaf = self.Y[leaf_idxs, 0]
                    E_leaf = E[leaf_idxs]
                    mask = E_leaf == env
                    if np.sum(mask) > 0:
                        expr += cp.sum(cp.square(Y_leaf[mask] - c[j]))
                constraints.append(expr / n_env <= t)

            objective = cp.Minimize(t)
            problem = cp.Problem(objective, constraints)
            problem.solve(warm_start=True)

            for j in range(n_leaves):
                self.trees[i].leaf_nodes[j].value = np.array(c[j].value, dtype=np.float64)

    def predict_weights(
        self, X: ArrayLike | None = None, scale: bool = True
    ) -> np.ndarray:
        """
        Predicts a weight matrix Z, where Z_{i,j} indicates if X_i and
        X0_j are in the same leaf node, where X0 denotes the training data.
        If scaling is True, then the value is divided by the number of other
        training data in the leaf node and averaged over all the estimators of
        the tree. If scaling is None, it is neither row-wise scaled and is
        instead summed up over all estimators of the forest.

        Parameters
        ----------
        X: array-like object of shape Mxd
            New samples to predict a weight.
            If None then X is treated as the training and or prediction data
            of size Nxd.

        scale: bool
            Whether to do row-wise scaling

        Returns
        -------
        np.ndarray
            A numpy array of shape MxN, wehre N denotes the number of rows of
            the training and or prediction data.
        """
        if X is None:
            size_0 = self.X_n_rows
            X = self.X
        else:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
            X = shared_numpy_array(X)
            size_0 = X.shape[0]

        if scale:
            scaling = "row"
        else:
            scaling = "none"

        weight_list = self.parallel.async_map(
            tree_based_weights,
            map_input=self.trees,
            X0=X,
            X1=None,
            size_X0=size_0,
            size_X1=self.X_n_rows,
            scaling=scaling,
            n_jobs=self.n_jobs_pred,
        )

        if scale:
            ret = np.mean(weight_list, axis=0)
        else:
            ret = np.sum(weight_list, axis=0)
        return ret

    def similarity(self, X0: ArrayLike, X1: ArrayLike):
        """
        Computes a similarity Z of size NxM, where each element Z_{i,j}
        is 1 if element X0_i and X1_j end up in the same leaf node.
        Z is the averaged over all the estimators of the forest.

        Parameters
        ----------
        X0: array-like object of shape Nxd
            Array corresponding to row elements of Z.
        X1: array-like object of shape Mxd
            Array corresponding to column elements of Z.

        Returns
        -------
        np.ndarray
            A NxM shaped np.ndarray.
        """
        X0, _ = self._check_input(X0)
        self._check_dimensions(X0)
        X1, _ = self._check_input(X1)
        self._check_dimensions(X1)

        size_0 = X0.shape[0]
        size_1 = X1.shape[0]
        weight_list = self.parallel.async_map(
            tree_based_weights,
            map_input=self.trees,
            X0=X0,
            X1=X1,
            size_X0=size_0,
            size_X1=size_1,
            scaling="similarity",
            n_jobs=self.n_jobs_pred,
        )
        return np.mean(weight_list, axis=0)
