# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:50:36 2016

@author: Hi
"""

from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.sparse import issparse
__all__ = ["MonoRuleRandomForest"]

MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


class MonoRuleRandomForest(RandomForestClassifier):
    """A random forest classifier with monotone feature capability.

    This algorithm fits `sci-kit learn's` RandomForestClassifier and then
    filters the leaf rules to eliminate monotone non-compliant rules. The
    approach is described in the paper by Bartley et al. 2017. It is very fast,
    but not as accurate as the approach described in our later papers.

    Other than the additional constructor variables `incr_feats` and
    `decr_feats` all other variables are as per `sci-kit learn`
    `RandomForestClassifier`.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).


    Parameters
    ----------
    incr_feats : list, optional (default=[])
        The indices of monotone increasing features (note: one based, not zero)
    decr_feats : list, optional (default=[])
        The indices of monotone decreasing features (note: one based, not zero)
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.


    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    .. [2] C. Bartley, W. Liu, M. Reynolds, "A Novel Framework for \
    Constructing Partially Monotone Rule Ensembles", ICDE, prepub, tba, 2017.

    See also
    --------
    """
    def __init__(
            self,
            incr_feats=[],
            decr_feats=[],
            n_estimators=10,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
        self.incr_feats = incr_feats if incr_feats is not None else []
        self.decr_feats = decr_feats if decr_feats is not None else []
        self.mt_feats = np.asarray(list(incr_feats) + list(decr_feats))
        self.oob_score_local = oob_score

    def fit(self, X, y, sample_weight=None):
        # Validate or convert input data
        super().fit(X, y)
        # Monotonise the trees
        if len(self.mt_feats) > 0:
            for est in self.estimators_:
                tree_ = est.tree_
                monotonise_tree(
                    tree_,
                    X.shape[1],
                    self.incr_feats,
                    self.decr_feats)
        # prepare data for OOB score calculation
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()
        # Remap output
        n_samples, self.n_features_ = X.shape
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]
        y, expanded_class_weight = self._validate_y_class_weight(y)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")
        if self.oob_score_local:
            self._set_oob_score(X, y)
        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # Save majority class for indecisive cases
        unq, cnts = np.unique(y, return_counts=True)
        self.maj_class_training = self.classes_[int(unq[np.argmax(cnts)])]
        return

    def predict(self, X):
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            classes = self.classes_.take(np.argmax(proba, axis=1), axis=0)
            classes[np.sum(proba, axis=1) == 0] = self.maj_class_training
            return classes
        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions

    def get_leaf_counts(self, only_count_non_zero=True):
        numtrees = np.int(self.get_params()['n_estimators'])
        num_leaves = np.zeros(numtrees, dtype='float')
        for itree in np.arange(numtrees):
            tree = self.estimators_[itree].tree_
            n_nodes = tree.node_count
            children_left = tree.children_left
            children_right = tree.children_right
            node_depth = np.zeros(shape=n_nodes)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1

                # If we have a test node
                if node_is_leaf(
                        tree,
                        node_id,
                        only_count_non_zero=only_count_non_zero):
                    is_leaves[node_id] = True
                elif node_is_leaf(tree, node_id, only_count_non_zero=False):
                    is_leaves[node_id] = False
                else:  # (children_left[node_id] != children_right[node_id]):
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))

            num_leaves[itree] = np.sum(is_leaves)
        return num_leaves

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_score = 0.0
        predictions = []
        maj_class_from_sampled = np.zeros(n_samples)
        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            # estimate majority class (intercept) from 'training' data (ie
            # sampled_indices)
            sampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            unq, cnts = np.unique(y[sampled_indices], return_counts=True)
            maj_class_from_sampled[unsampled_indices] = unq[np.argmax(cnts)]
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            pred_classes = np.argmax(predictions[k], axis=1)
            if (predictions[k].sum(axis=1) == 0).any():
                # this can now be due to the monotonisation. For 0 predictions,
                # use majority class as determined from sampled 'training' data
                print('num pts with no predictions: ' +
                      str(np.sum(predictions[k].sum(axis=1) == 0)))
                pred_classes[predictions[k].sum(
                    axis=1) == 0] = maj_class_from_sampled[
                    predictions[k].sum(axis=1) == 0]
            oob_score += np.mean(y[:, k] ==
                                 pred_classes, axis=0)
        self.oob_score_ = oob_score / self.n_outputs_


def node_is_leaf(tree, node_id, only_count_non_zero=False):
    if only_count_non_zero:
        return tree.children_left[node_id] == tree.children_right[node_id] and\
            not np.all(np.asarray(tree.value[node_id][0]) == 0.)
    else:
        return tree.children_left[node_id] == tree.children_right[node_id]


def monotonise_tree(tree, n_feats, incr_feats, decr_feats):
    """Helper to turn a tree into as set of rules
    """
    PLUS = 0
    MINUS = 1
    mt_feats = np.asarray(list(incr_feats) + list(decr_feats))

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       path=None):
        if path is None:
            path = np.zeros([n_feats, 2])
        else:
            path[feature, PLUS if operator[0] == '>' else MINUS] = 1

        if not node_is_leaf(
                tree,
                node_id):
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, path.copy())

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, path.copy())
        else:  # a leaf node
            if np.sum(path) > 0:
                # check if all increasing
                all_increasing = np.sum(np.asarray([path[i_feat,
                        MINUS] if i_feat + 1 in incr_feats else path[i_feat,
                        PLUS] for i_feat in mt_feats - 1])) == 0
                all_decreasing = np.sum(np.asarray([path[i_feat,
                        MINUS] if i_feat + 1 in decr_feats else path[i_feat,
                        PLUS] for i_feat in mt_feats - 1])) == 0
                counts = np.asarray(tree.value[node_id][0])
                probs = counts / np.sum(counts)
                predicted_value = np.sign(probs[1] - 0.5)
                if predicted_value >= 0 and all_increasing:  # ok
                    pass
                elif predicted_value <= 0 and all_decreasing:  # ok
                    pass
                else:  # not a valid rule
                    tree.value[node_id][0] = [0., 0.]
            else:
                print('Tree has only one node (i.e. the root node!)')
            return None
    if len(mt_feats) > 0:
        traverse_nodes()

    return tree
