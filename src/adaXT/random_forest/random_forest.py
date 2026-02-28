from typing import Literal
from numpy import int32 as INT

import numpy as np
import cvxpy as cp
import torch
import warnings
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
    sols_erm: np.ndarray | None,
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
    minmax_obj: str | None = None,
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
            sample_weight=sample_weight,
            minmax_obj=minmax_obj,
            sols_erm=sols_erm)
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
        minmax_obj: str | None = None,
        sols_erm_trees: np.ndarray | None = None,
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
        minmax_obj: str | None
            Objective to use with MinMaxRegression.
            Accepted values are {"mse", "reward", "regret"}.
        sols_erm_trees : np.ndarray or None, default=None
            A reference set of predictions from each tree of the standard RF, required if `minmax_obj='regret'`.
            It should be an array with as many rows as the number of trees
            and as many columns as the target values.
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
        self.minmax_obj = minmax_obj
        if sols_erm_trees is None:
            self.sols_erm_trees = sols_erm_trees
        else:
            self.sols_erm_trees = list(sols_erm_trees)

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
            map_input=zip(self.fitting_indices, self.prediction_indices, self.sols_erm_trees),
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
            minmax_obj=self.minmax_obj,
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

        if self.minmax_obj is not None and self.forest_type != "MinMaxRegression":
            raise ValueError(f"{self.forest_type} only supports minmax_obj=None.")

        if self.minmax_obj is not None and self.minmax_obj not in ["mse", "reward", "regret"]:
            raise ValueError("minmax_obj must be 'mse', 'reward', 'regret'.")

        if self.sols_erm_trees is not None and self.forest_type != "MinMaxRegression":
            raise ValueError(f"{self.forest_type} only supports sols_erm_trees=None.")

        if self.sols_erm_trees is None and self.minmax_obj == "regret":
            raise ValueError("sols_erm_trees cannot be None when minmax_obj is 'regret'.")

        if self.minmax_obj is None and self.forest_type == "MinMaxRegression":
            self.minmax_obj = "mse"

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

        if self.sols_erm_trees is not None:
            if (len(self.sols_erm_trees) != self.n_estimators or
                    len(self.sols_erm_trees[0]) != self.X_n_rows):
                raise ValueError(f"sols_erm_tree should have dimension ({self.n_estimators}, {self.X_n_rows}).")
        else:
            self.sols_erm_trees = [None] * self.n_estimators

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

    def predict(self, X: ArrayLike, revert_to_rf: bool = False, **kwargs) -> np.ndarray:
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
        
        revert_to_rf : bool, default=False
            If True, use the reverted leaf values (where indeterminate leaves are reset to ERM)
            instead of the fully optimized values.
            Only applicable if modify_predictions_trees has been called.

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

        # Handle revert_to_rf
        swapped = False
        current_values = []
        if revert_to_rf:
            # check if we have the reverted values (where indeterminate leaves are reset to ERM)
            if not hasattr(self, "reverted_leaf_values") or self.reverted_leaf_values is None:
                warnings.warn("Reverted leaf values not available. Using current leaf values.")
            else:
                # if we do, we temporarily swap the current optimized values in the trees
                # with these reverted ones. We save the current values so we can put them back later.
                swapped = True
                for i, (tree, rev_values) in enumerate(zip(self.trees, self.reverted_leaf_values)):
                    leaves = tree.leaf_nodes
                    tree_current_values = []
                    for j, leaf in enumerate(leaves):
                        tree_current_values.append(leaf.value)
                        leaf.value = np.array(rev_values[j], dtype=np.float64)
                    current_values.append(tree_current_values)

        try:
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
        finally:
            if swapped:
                # put the original (optimized) values back
                for i, (tree, curr_values) in enumerate(zip(self.trees, current_values)):
                    leaves = tree.leaf_nodes
                    for j, leaf in enumerate(leaves):
                        leaf.value = curr_values[j]

        return prediction

    def refine_weights(
        self,
        X_val: ArrayLike,
        Y_val: ArrayLike,
        E_val: ArrayLike,
        X: ArrayLike | None = None,
        risk: str = "mse",
        sols_erm: np.ndarray | None = None,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Adjust the weights of the trees in the forest to minimize the maximum risk
        across training environments.

        Parameters
        ----------
        X_val : ArrayLike
            feature matrix

        Y_val : ArrayLike
            response values

        E_val : ArrayLike
            environment labels

        X : ArrayLike | None
            feature matrix for predictions (optional)

        risk: str
            risk to use. Accepted values are 'mse', 'nrw', 'reg'

        sols_erm : ArrayLike | None
            predictions obtained with maximum risk minimization.
            Must be provided if risk is 'reg'

        kwargs
            contains additional arguments, such as "solver"

        Examples
        --------
        >>> model = RandomForest("Regression", n_estimators=50, min_samples_leaf=20, seed=42)
        >>> model.refine_weights(X_val=X_val, Y_val=Y_val, E_val=E_val, X=X, risk="mse", solver="ECOS")
        """
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call predict"
            )
        if risk not in ["mse", "nrw", "reg"]:
            raise ValueError("risk must be one of 'mse', 'nrw', or 'reg'")
        if sols_erm is None and risk == "reg":
            raise ValueError("sols_erm is required if risk is 'reg'")

        if X is not None:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
            X = shared_numpy_array(X)

        X_val, Y_val = self._check_input(X_val, Y_val)
        self._check_dimensions(X_val)

        X_val = shared_numpy_array(X_val)
        Y_val = shared_numpy_array(Y_val)

        if sols_erm is not None:
            sols_erm, _ = self._check_input(sols_erm)
            sols_erm = shared_numpy_array(sols_erm)

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
            risk=risk,
            sols_erm=sols_erm,
            **kwargs,
        )

        if X is not None:
            predictions = self.parallel.async_map(
                predict_default,
                self.trees,
                X_pred=X,
                n_jobs=self.n_jobs,
            )
            predictions = np.array(predictions).T
            weighted_predictions = predictions @ weights_minmax.value
            return weighted_predictions, weights_minmax

        return weights_minmax

    @staticmethod
    def _project_onto_simplex(v: np.ndarray) -> np.ndarray:
        """
        Projection onto the probability simplex.
        Reference: Wang et al. (2013).
            "Projection onto the probability simplex:
            An efficient algorithm with a simple proof, and an application"
            https://arxiv.org/pdf/1309.1541
        """
        original_shape = v.shape
        v_flat = v.flatten()
        D = v_flat.size

        # Step 1: Sort in descending order
        u = np.sort(v_flat)[::-1]

        # Step 2: Find rho
        cssv = np.cumsum(u)
        j = np.arange(1, D + 1)
        condition = u + (1.0 / j) * (1 - cssv)
        rho = np.where(condition > 0)[0].max() + 1

        # Step 3: Compute lambda
        lambda_val = (1 - np.sum(u[:rho])) / rho

        # Step 4: Compute projection
        x = np.maximum(v_flat + lambda_val, 0)

        return x.reshape(original_shape)

    ########################
    # OPTIMIZATION HELPERS #
    ########################

    @staticmethod
    def _build_leaf_env_stats(
        leaf_indices_list: list[np.ndarray],
        Y: np.ndarray,
        E: np.ndarray,
        method: str,
        alpha: float,
        sols_erm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build compact per-(env, leaf) sufficient statistics.

        Returns
        -------
        unique_envs : (n_envs,)
        counts      : (n_envs, n_leaves)
        sum_y       : (n_envs, n_leaves)
        sumsq_y     : (n_envs, n_leaves)
        env_baseline: (n_envs,)
            0 for mse,
            mean(Y^2) for reward,
            alpha * mean((Y - sols_erm)^2) for regret.
        """
        if len(leaf_indices_list) == 0:
            raise ValueError("A tree must have at least one leaf.")

        # Unique envs across this tree's samples
        all_indices = np.concatenate([np.asarray(idxs, dtype=np.int64) for idxs in leaf_indices_list])
        unique_envs = np.unique(E[all_indices, 0])
        n_envs = len(unique_envs)
        n_leaves = len(leaf_indices_list)

        counts = np.zeros((n_envs, n_leaves), dtype=np.float64)
        sum_y = np.zeros((n_envs, n_leaves), dtype=np.float64)
        sumsq_y = np.zeros((n_envs, n_leaves), dtype=np.float64)

        env_counts = np.zeros(n_envs, dtype=np.float64)
        env_sumsq_total = np.zeros(n_envs, dtype=np.float64)
        env_regret_total = np.zeros(n_envs, dtype=np.float64) if method == "regret" else None

        for j, idxs in enumerate(leaf_indices_list):
            idxs = np.asarray(idxs, dtype=np.int64)
            if idxs.size == 0:
                continue

            e_leaf = E[idxs, 0]
            y_leaf = Y[idxs, 0].astype(np.float64, copy=False)

            # Map env labels to [0, n_envs)
            env_pos = np.searchsorted(unique_envs, e_leaf)

            cnt = np.bincount(env_pos, minlength=n_envs).astype(np.float64)
            sy = np.bincount(env_pos, weights=y_leaf, minlength=n_envs).astype(np.float64)
            sy2 = np.bincount(env_pos, weights=y_leaf * y_leaf, minlength=n_envs).astype(np.float64)

            counts[:, j] = cnt
            sum_y[:, j] = sy
            sumsq_y[:, j] = sy2

            env_counts += cnt
            env_sumsq_total += sy2

            if method == "regret":
                if sols_erm is None:
                    raise ValueError("sols_erm is required when method='regret'.")
                sols_leaf = sols_erm[idxs, 0].astype(np.float64, copy=False)
                reg = np.bincount(
                    env_pos,
                    weights=(y_leaf - sols_leaf) ** 2,
                    minlength=n_envs,
                ).astype(np.float64)
                env_regret_total += reg

        if np.any(env_counts == 0):
            raise ValueError("Encountered an empty environment while building leaf stats.")

        if method == "mse":
            env_baseline = np.zeros(n_envs, dtype=np.float64)
        elif method == "reward":
            env_baseline = env_sumsq_total / env_counts
        elif method == "regret":
            env_baseline = alpha * (env_regret_total / env_counts)
        else:
            raise ValueError(f"Unknown method: {method}")

        return unique_envs, counts, sum_y, sumsq_y, env_baseline

    @staticmethod
    def _optimize_cp_standard(
        counts: np.ndarray,
        sum_y: np.ndarray,
        sumsq_y: np.ndarray,
        env_baseline: np.ndarray,
        initial_values: np.ndarray,
        method: str,
        solver: str | None,
    ) -> np.ndarray:
        n_envs, n_leaves = counts.shape
        env_counts = counts.sum(axis=1)

        c = cp.Variable(n_leaves)
        t = cp.Variable(nonneg=(method == "mse"))
        c.value = initial_values

        constraints = []
        for e in range(n_envs):
            # Sum_j [ n_ej * c_j^2 - 2 * sum_y_ej * c_j + sumsq_y_ej ]
            expr = cp.sum(
                cp.multiply(counts[e], cp.square(c))
                - 2.0 * cp.multiply(sum_y[e], c)
                + sumsq_y[e]
            )
            constraints.append(expr / env_counts[e] - env_baseline[e] <= t)

        problem = cp.Problem(cp.Minimize(t), constraints)

        solve_kwargs = {"warm_start": True}
        if solver is not None:
            solve_kwargs["solver"] = solver

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            problem.solve(**solve_kwargs)

        if c.value is None:
            raise ValueError(f"CVXPY failed to solve the problem. Status: {problem.status}")

        return np.asarray(c.value, dtype=np.float64)

    @staticmethod
    def _optimize_cp_bcd(
        counts: np.ndarray,
        sum_y: np.ndarray,
        sumsq_y: np.ndarray,
        env_baseline: np.ndarray,
        initial_values: np.ndarray,
        method: str,
        solver: str | None,
        block_size: int,
        max_iter: int,
        patience_bcd: int,
        min_delta: float,
        verbose: bool,
    ) -> np.ndarray:
        n_envs, n_leaves = counts.shape
        env_counts = counts.sum(axis=1)

        block_size = max(1, int(block_size))
        c_full = initial_values.astype(np.float64, copy=True)

        blocks = [
            np.arange(start, min(start + block_size, n_leaves))
            for start in range(0, n_leaves, block_size)
        ]
        n_blocks = len(blocks)

        best_t = np.inf
        iters_no_improvement = 0

        if verbose:
            print(f"Starting BCD optimization with {n_blocks} blocks of size {block_size}")
            print(f"Total variables: {n_leaves}, Max iterations: {max_iter}")
            print("-" * 60)

        for iter_idx in range(max_iter):
            block_id = iter_idx % n_blocks
            active_idx = blocks[block_id]
            inactive_mask = np.ones(n_leaves, dtype=bool)
            inactive_mask[active_idx] = False

            c_block = cp.Variable(len(active_idx))
            c_block.value = c_full[active_idx]
            t = cp.Variable(nonneg=(method == "mse"))

            if verbose and iter_idx % n_blocks == 0:
                cycle = iter_idx // n_blocks + 1
                print(f"Cycle {cycle}: best_t = {best_t:.6f}, no_improvement = {iters_no_improvement}")

            constraints = []
            for e in range(n_envs):
                # Fixed contribution
                fixed_expr = np.sum(
                    counts[e, inactive_mask] * (c_full[inactive_mask] ** 2)
                    - 2.0 * sum_y[e, inactive_mask] * c_full[inactive_mask]
                    + sumsq_y[e, inactive_mask]
                )

                # Variable contribution
                var_expr = cp.sum(
                    cp.multiply(counts[e, active_idx], cp.square(c_block))
                    - 2.0 * cp.multiply(sum_y[e, active_idx], c_block)
                    + sumsq_y[e, active_idx]
                )

                total_expr = fixed_expr + var_expr
                constraints.append(total_expr / env_counts[e] - env_baseline[e] <= t)

            problem = cp.Problem(cp.Minimize(t), constraints)

            solve_kwargs = {"warm_start": True}
            if solver is not None:
                solve_kwargs["solver"] = solver

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(**solve_kwargs)

            if c_block.value is not None and t.value is not None:
                c_full[active_idx] = np.asarray(c_block.value, dtype=np.float64)
                curr_t = float(t.value)
                improvement = max(best_t - curr_t, 0.0)

                if verbose:
                    print(f"Block {block_id:2d}: t={curr_t:.6f}, improvement={improvement:.2e}")

                if curr_t < best_t:
                    if best_t - curr_t < min_delta:
                        iters_no_improvement += 1
                    else:
                        iters_no_improvement = 0
                    best_t = curr_t
                else:
                    iters_no_improvement += 1

                if iters_no_improvement >= patience_bcd:
                    if verbose:
                        print(f"Converged after {iter_idx + 1} iterations (patience reached)")
                    break
            else:
                if verbose:
                    print(f"Block {block_id:2d}: SOLVER FAILED - status={problem.status}")
                iters_no_improvement += 1
                if iters_no_improvement >= patience_bcd:
                    if verbose:
                        print(f"Stopping after {iter_idx + 1} iterations (too many solver failures)")
                    break

        if verbose:
            print("-" * 60)
            print(f"BCD completed: {iter_idx + 1} iterations, final_t = {best_t:.6f}")
            print("-" * 60)

        return c_full

    @staticmethod
    def _optimize_extragradient(
        counts: np.ndarray,
        sum_y: np.ndarray,
        sumsq_y: np.ndarray,
        env_baseline: np.ndarray,
        initial_values: np.ndarray,
        gamma: float,
        epochs: int,
        min_delta: float,
        early_stopping: bool,
        patience: int,
        verbose: bool,
        tree_idx: int,
    ) -> np.ndarray:
        if verbose:
            print("-" * 60)
            print("Starting Extragradient optimization")
            print("-" * 60)

        counts = np.asarray(counts, dtype=np.float64)
        sum_y = np.asarray(sum_y, dtype=np.float64)
        sumsq_y = np.asarray(sumsq_y, dtype=np.float64)
        env_baseline = np.asarray(env_baseline, dtype=np.float64)
        env_counts = counts.sum(axis=1).astype(np.float64)

        n_envs, n_leaves = counts.shape
        c = np.asarray(initial_values, dtype=np.float64).copy()
        p = np.full(n_envs, 1.0 / n_envs, dtype=np.float64)

        best_max_loss = np.inf
        epochs_no_improvement = 0
        print_every = max(1, epochs // 10)

        sumsq_y_by_env = np.sum(sumsq_y, axis=1)

        def compute_losses_and_gradients(
                c_input: np.ndarray,
                p_input: np.ndarray,
                compute_grad: bool = True,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            # Per-env numerator:
            # sum_j [ n_ej * c_j^2 - 2 * sum_y_ej * c_j + sumsq_y_ej ]
            residual_num = (
                counts @ (c_input ** 2)
                - 2.0 * (sum_y @ c_input)
                + sumsq_y_by_env
            )

            losses = residual_num / env_counts - env_baseline

            if not compute_grad:
                return losses, None

            # Gradient wrt c_j:
            # sum_e p_e * (2 / n_e) * (n_ej * c_j - sum_y_ej)
            grad = np.sum(
                ((p_input * (2.0 / env_counts))[:, None])
                * (counts * c_input[None, :] - sum_y),
                axis=0,
            )

            return losses, grad

        for epoch in range(epochs):
            losses, grad = compute_losses_and_gradients(c, p, compute_grad=True)

            # Half step
            c_half = c - gamma * grad
            p_half = RandomForest._project_onto_simplex(p + gamma * losses)

            # Evaluate at half step
            losses_h, grad_h = compute_losses_and_gradients(c_half, p_half, compute_grad=True)

            # Full step
            c = c - gamma * grad_h
            p = RandomForest._project_onto_simplex(p + gamma * losses_h)

            losses_new, _ = compute_losses_and_gradients(c, p, compute_grad=False)
            max_loss = np.max(losses_new)
            weighted_loss = np.sum(p * losses_new)

            if verbose and epoch % print_every == 0:
                print(
                    f"Tree {tree_idx}, Epoch {epoch}: "
                    f"max_loss = {max_loss:.6f}, weighted_loss = {weighted_loss:.6f}"
                )

            if best_max_loss - max_loss > min_delta:
                best_max_loss = max_loss
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if early_stopping and (epochs_no_improvement >= patience):
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch}, "
                        f"best max_loss = {best_max_loss:.6f}"
                    )
                break

        return c.astype(np.float64, copy=False)

    ##########################
    # POST-PROCESSING HELPER #
    ##########################

    @staticmethod
    def _postprocess_indeterminate_leaves(
        counts: np.ndarray,
        sum_y: np.ndarray,
        sumsq_y: np.ndarray,
        env_baseline: np.ndarray,
        initial_values: np.ndarray,
        optimized_values: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Identifies worst-case environments and reverts non-contributing leaves to their original RF state."""
        # idea: if a leaf doesn't contain obs of any of the "worst" environments (the ones with max loss),
        # then its value doesn't matter for the worst-case objective (as long as it doesn't cause a different
        # env to become the worst one).
        # We thus provide the option to revert such leaves back to their initial values (from standard RF)
        env_counts = counts.sum(axis=1)

        # Per-env objective at optimized values
        residual_num = (
            counts @ (optimized_values ** 2)
            - 2.0 * (sum_y @ optimized_values)
            + np.sum(sumsq_y, axis=1)
        )
        env_metrics = residual_num / env_counts - env_baseline

        max_metric = np.max(env_metrics)
        worst_env_mask = np.isclose(env_metrics, max_metric, atol=1e-6)

        # A leaf is "indeterminate" if it has zero mass in every worst-case env
        leaf_has_worst = np.sum(counts[worst_env_mask, :], axis=0) > 0
        indeterminate_mask = ~leaf_has_worst

        reverted_values = optimized_values.copy()
        reverted_values[indeterminate_mask] = initial_values[indeterminate_mask]
        indeterminate_count = int(np.sum(indeterminate_mask))

        return reverted_values, indeterminate_count

    #########
    # MaxRM #
    #########

    @staticmethod
    def _modify_single_tree_predictions(
        tree_data,
        Y,
        E,
        method: str = "mse",
        alpha: float = 1.0,
        solver: str | None = None,
        bcd: bool = False,
        block_size: int = 15,
        max_iter: int = 100,
        gamma: float = 0.1,
        epochs: int = 100,
        verbose: bool = False,
        opt_method: str = "cp",
        early_stopping: bool = False,
        patience: int = 5,
        patience_bcd: int = 1,
        min_delta: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Modify the leaf constants of a single tree.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, int]
            Initial values and optimized values for the tree's leaf nodes.
            Reverted values and indeterminate count.
        """
        if method == "regret":
            tree_idx, leaf_indices_list, initial_values, sols_erm = tree_data
        else:
            tree_idx, leaf_indices_list, initial_values = tree_data
            sols_erm = None

        initial_values = np.asarray(initial_values, dtype=np.float64).flatten()

        _, counts, sum_y, sumsq_y, env_baseline = RandomForest._build_leaf_env_stats(
            leaf_indices_list=leaf_indices_list,
            Y=Y,
            E=E,
            method=method,
            alpha=alpha,
            sols_erm=sols_erm,
        )

        if opt_method == "cp":
            if bcd:
                optimized_values = RandomForest._optimize_cp_bcd(
                    counts=counts,
                    sum_y=sum_y,
                    sumsq_y=sumsq_y,
                    env_baseline=env_baseline,
                    initial_values=initial_values,
                    method=method,
                    solver=solver,
                    block_size=block_size,
                    max_iter=max_iter,
                    patience_bcd=patience_bcd,
                    min_delta=min_delta,
                    verbose=verbose,
                )
            else:
                optimized_values = RandomForest._optimize_cp_standard(
                    counts=counts,
                    sum_y=sum_y,
                    sumsq_y=sumsq_y,
                    env_baseline=env_baseline,
                    initial_values=initial_values,
                    method=method,
                    solver=solver,
                )
        elif opt_method == "extragradient":
            optimized_values = RandomForest._optimize_extragradient(
                counts=counts,
                sum_y=sum_y,
                sumsq_y=sumsq_y,
                env_baseline=env_baseline,
                initial_values=initial_values,
                gamma=gamma,
                epochs=epochs,
                min_delta=min_delta,
                early_stopping=early_stopping,
                patience=patience,
                verbose=verbose,
                tree_idx=tree_idx,
            )
        else:
            raise ValueError(f"Unknown opt_method: {opt_method}")

        reverted_values, indet_count = RandomForest._postprocess_indeterminate_leaves(
            counts=counts,
            sum_y=sum_y,
            sumsq_y=sumsq_y,
            env_baseline=env_baseline,
            initial_values=initial_values,
            optimized_values=optimized_values,
        )

        return initial_values, optimized_values, reverted_values, indet_count

    def modify_predictions_trees(
        self,
        E: ArrayLike,
        method: str = "mse",
        sols_erm: np.ndarray | None = None,
        sols_erm_trees: np.ndarray | None = None,
        alpha: float = 1.0,
        solver: str | None = None,
        bcd: bool = False,
        block_size: int = 15,
        max_iter: int = 100,
        gamma: float = 0.1,
        epochs: int = 100,
        verbose: bool = False,
        opt_method: str = "cp",
        early_stopping: bool = False,
        patience: int = 5,
        patience_bcd: int = 1,
        min_delta: float = 1e-3,
        n_jobs: int = 1,
    ) -> None:
        """
        Adjust the leaf predictions of each tree to minimize the worst-case loss across
        different environments, optionally using an extragradient optimization approach.

        Parameters
        ----------
        E : ArrayLike
            Environment labels.

        method : {'mse', 'regret', 'reward'}, default='mse'
            The type of objective to minimize across environments.
            - 'mse': Minimize the maximum mean squared error across environments.
            - 'regret': Minimize the maximum regret, defined as the difference between current MSE and
                        a reference ERM solution (sols_erm), scaled by `alpha`.
            - 'reward': Minimize the maximum negative reward, which is equivalent
                        to maximizing the minimal reward.

        sols_erm : np.ndarray or None, default=None
            A reference set of predictions from an ERM model, required if `method='regret'`.
            Should be of the same shape as the target values.

        sols_erm_trees : np.ndarray or None, default=None
            A reference set of predictions from each tree of the standard RF, required if `method='regret'`.
            Should be an array with as many rows as the number of trees
            and as many columns as the target values.

        alpha : float, default=1.0
            Scaling factor for the reference loss in regret computation (only used when method='regret').

        solver : str or None, default=None
            Solver used by cvxpy for the convex optimization problem.
            Examples are 'ECOS', 'SCS', 'CLARABEL'.

        bcd : bool, default=False
            If True, use block-coordinate descent (BCD) to solve the convex program.
            Only used when opt_method='cp'.

        block_size : int, default=10
            Number of leaf values to update per block in BCD.
            Determines the size of each coordinate block.
            Only used when opt_method='cp' and bcd=True.

        max_iter : int, default=100
            Maximum number of BCD iterations. Only used when opt_method='cp' and bcd=True.

        gamma : float, default=0.01
            Step size for the extragradient optimizer (only used if `opt_method='extragradient'`).

        epochs : int, default=500
            Number of iterations for the extragradient optimization procedure.

        verbose : bool, default=False
            Whether to print optimization progress and diagnostics.

        opt_method : {'cp', 'extragradient'}, default='cp'
            Optimization method to use:
            - 'cp': Use convex programming (via CVXPY).
            - 'extragradient': Use an extragradient algorithm implemented with PyTorch.

        early_stopping : bool, default=False
            If True, the optimization will stop early if the loss does not improve
            over a number of consecutive epochs defined by `patience`.

        patience : int, default=5
            Number of consecutive epochs without sufficient improvement in loss
            before stopping the extragradient optimization early.
            Only used if `early_stopping=True`.

        patience_bcd : int, default=1
            Number of consecutive epochs without sufficient improvement in loss
            before stopping the block-coordinate descent optimization early.

        min_delta : float, default=1e-3
            Minimum change in the maximum loss between epochs to qualify as an
            improvement. Changes smaller than `min_delta` are considered as no improvement.
            Only used if `early_stopping=True`.

        n_jobs : int | tuple[int, int]
            The number of jobs used to modify the leaf predictions.

        Notes
        -----
        - If the optimization increases the worst-case error (based on the specified objective),
          the original predictions are restored.

        Examples
        --------
        >>> model.modify_predictions_trees(E=envs)
        >>> model.modify_predictions_trees(E=envs, opt_method="extragradient", verbose=True)
        """
        if self.forest_type not in ["Regression", "MinMaxRegression"]:
            raise ValueError("modify_predictions only works for Regression and MinMaxRegression")

        if method not in ["mse", "regret", "reward"]:
            raise ValueError("method must be 'mse', 'regret' or 'reward'")

        if opt_method not in ["cp", "extragradient"]:
            raise ValueError("opt_method must be 'cp' or 'extragradient'")

        if self.min_samples_leaf == 1:
            warnings.warn(
                "modify_predictions_trees could fail if min_samples_leaf == 1."
                "\nNote: if all leaves have only one observation, MaxRM-RF and RF yield identical solutions"
            )

        def compute_max_env_mse(preds):
            max_mse = 0.0
            for env in unique_envs:
                mask = E[:, 0] == env
                if np.sum(mask) > 0:
                    mse = np.mean((self.Y[mask, 0] - preds[mask]) ** 2)
                    max_mse = max(max_mse, mse)
            return max_mse

        def compute_max_env_regret(preds):
            if sols_erm is None or sols_erm_trees is None:
                raise ValueError("sols_erm and sols_erm_trees must be provided when method='regret'")
            max_regret = -np.inf
            for env in unique_envs:
                mask = E[:, 0] == env
                if np.sum(mask) > 0:
                    loss_current = np.mean((self.Y[mask, 0] - preds[mask]) ** 2)
                    loss_best = np.mean((self.Y[mask, 0] - sols_erm[mask, 0]) ** 2)
                    regret = loss_current - alpha * loss_best
                    max_regret = max(max_regret, regret)
            return max_regret

        def compute_max_env_neg_rw(preds):
            max_neg_reward = -np.inf
            for env in unique_envs:
                mask = E[:, 0] == env
                if np.sum(mask) > 0:
                    neg_reward = (
                            np.mean((self.Y[mask, 0] - preds[mask]) ** 2)
                            - np.mean(self.Y[mask, 0] ** 2)
                    )
                    max_neg_reward = max(max_neg_reward, neg_reward)
            return max_neg_reward

        unique_envs = np.unique(E)

        E = np.ascontiguousarray(E, dtype=np.int64)
        E = np.expand_dims(E, axis=1)
        row, col = E.shape
        shared_E = RawArray(ctypes.c_int64, row * col)
        shared_E_np = np.ndarray(shape=(row, col), dtype=np.int64, buffer=shared_E)
        np.copyto(shared_E_np, E)
        E = shared_E_np

        if sols_erm is not None:
            _, sols_erm = self._check_input(Y=sols_erm)
            sols_erm = shared_numpy_array(sols_erm)

            _, sols_erm_trees = self._check_input(Y=sols_erm_trees)
            sols_erm_trees = shared_numpy_array(sols_erm_trees)

        initial_preds = self.predict(self.X)
        if method == "mse":
            initial_score = compute_max_env_mse(initial_preds)
        elif method == "regret":
            initial_score = compute_max_env_regret(initial_preds)
        else:
            initial_score = compute_max_env_neg_rw(initial_preds)

        # Build lightweight payloads
        tree_payloads = []
        for i, tree in enumerate(self.trees):
            leaf_indices_list = [np.asarray(leaf.indices, dtype=np.int64) for leaf in tree.leaf_nodes]
            initial_values = np.array([leaf.value for leaf in tree.leaf_nodes], dtype=np.float64).flatten()

            if method == "regret":
                tree_payloads.append(
                    (i, leaf_indices_list, initial_values, np.expand_dims(sols_erm_trees[i], axis=1))
                )
            else:
                tree_payloads.append((i, leaf_indices_list, initial_values))

        # Process all trees in parallel
        results = self.parallel.async_map(
            RandomForest._modify_single_tree_predictions,
            tree_payloads,
            Y=self.Y,
            E=E,
            method=method,
            alpha=alpha,
            solver=solver,
            bcd=bcd,
            block_size=block_size,
            max_iter=max_iter,
            gamma=gamma,
            epochs=epochs,
            verbose=verbose,
            opt_method=opt_method,
            early_stopping=early_stopping,
            patience=patience,
            patience_bcd=patience_bcd,
            min_delta=min_delta,
            n_jobs=n_jobs
        )

        # Extract initial and optimized values
        initial_values_per_tree = []
        optimized_values_per_tree = []
        reverted_values_per_tree = []
        indeterminate_counts = []

        for initial_vals, optimized_vals, reverted_vals, indet_count in results:
            initial_values_per_tree.append(initial_vals)
            optimized_values_per_tree.append(optimized_vals)
            reverted_values_per_tree.append(reverted_vals)
            indeterminate_counts.append(indet_count)

        # Update tree leaf values with optimized values
        for i, (tree, optimized_values) in enumerate(zip(self.trees, optimized_values_per_tree)):
            leaves = tree.leaf_nodes
            for j, leaf in enumerate(leaves):
                leaf.value = np.array(optimized_values[j], dtype=np.float64)

        # Evaluate global objective after all tree updates
        optimized_preds = self.predict(self.X)
        if method == "mse":
            optimized_score = compute_max_env_mse(optimized_preds)
        elif method == "regret":
            optimized_score = compute_max_env_regret(optimized_preds)
        else:
            optimized_score = compute_max_env_neg_rw(optimized_preds)

        if verbose:
            print(f"Initial score: {initial_score:.6f}")
            print(f"Optimized score: {optimized_score:.6f}")

        # Roll back if worse
        if optimized_score > initial_score:
            if verbose:
                print("Optimization made objective worse, rolling back...")
            for i, (tree, initial_values) in enumerate(zip(self.trees, initial_values_per_tree)):
                leaves = tree.leaf_nodes
                for j, leaf in enumerate(leaves):
                    leaf.value = np.array(initial_values[j], dtype=np.float64)
            self.reverted_leaf_values = None
            self.indeterminate_counts = None
        else:
            if verbose:
                print("Optimization successful.")
            self.reverted_leaf_values = reverted_values_per_tree
            self.indeterminate_counts = indeterminate_counts

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
