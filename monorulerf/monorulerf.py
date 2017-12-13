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
                    axis=1) == 0] = maj_class_from_sampled[predictions[k].sum(
                            axis=1) == 0]
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
