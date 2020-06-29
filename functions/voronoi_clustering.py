import abc
from warnings import warn

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

data_type_feature_values = 'feature_values'
datatype_distance_matrix = 'distance_matrix'


def numba_voronoi(distance_matrix, maximum_iterations, tolerance, multiplicity=None,
                  indices_of_initial_medoids=None, kmpp_seed=None, n_medoids=None):
    """
    Numba implementation of the Voronoi K-medoids clustering algorithm

    Parameters
    ----------
    distance_matrix: np.array of size (n_samples, n_samples)
        The distance matrix
    maximum_iterations: int
        The maximum number of iterations for the Voronoi algorithm
    tolerance: float
        The maximum distance between two successive medoids for the algorithm to stop
    multiplicity: np.array of size (n_samples,) or None
        The multiplicity of each datapoint
    indices_of_initial_medoids: np.array of size (n_medoids,) or None
        The indices of the initial medoids.
    kmpp_seed: int or None
        Seed for k-medoids++. Ignored if `indices_of_initial_medoids` is not None
    n_medoids: int or None
        Number of medoids.

    Notes
    -----
    Exactly one of `indices_of_initial_medoids` and `n_medoids` should be None. If the former is None, the initial indices
    are computed using :py:meth:`kmeans_plus_plus()<functions.voronoi_clustering.kmeans_plus_plus>`. If the latter is
    is None, the former is used as initial indices. `kmpp_seed` is only used if `n_medoids` is not None, it is used to
    seed the kmeans_plus_plus algorithm.

    Raises
    ------
    AssertionError: If both of `indices_of_initial_medoids` and `n_medoids` are None. If `indices_of_initial_medoids`
    and `n_medoids` are both not None, but `len(indices_of_initial_medoids)` != `n_medoids`

    Warnings
    --------
    If `kmpp_seed` is not None, but n_medoids is.

    Returns
    -------
    medoids : np.array of size (n_medoids,)
        The indices of the final medoids
    labels : np.array of size(n_samples)
        The labels of the final medoids. The datapoint with index `di` is associated to the medoid
        with index `medoids[labels[di]]`

    """
    # Input data
    n_samples = distance_matrix.shape[0]
    assert n_samples == distance_matrix.shape[1]

    if multiplicity is None:
        multiplicity = np.ones(shape=(n_samples,))

    # Verify indices_of_initial_medoids, n_medoids, and kmpp_seed
    if indices_of_initial_medoids is None:
        if n_medoids is None:
            raise AssertionError('indices_of_initial_medoids and n_medoids cannot both be None')
        else:
            indices_of_old_medoids = kmedoids_plus_plus(distance_matrix, n_medoids, multiplicity=multiplicity,
                                                        random_state=kmpp_seed)
    else:
        indices_of_old_medoids = indices_of_initial_medoids.copy()
        if n_medoids is None:
            n_medoids = len(indices_of_initial_medoids)
        else:
            assert n_medoids == len(indices_of_initial_medoids), \
                'mismatch between n_medoids and indices_of_initial_medoids'
        if kmpp_seed is not None:
            warn('kmpp_seed is ignored if indices_of_initial_medoids is given')

    iteration_number = 0

    while True:
        iteration_number += 1

        # Find clusters
        labels = __numba_voronoi_update_clusters(distance_matrix, indices_of_old_medoids, n_samples)

        # Update medoids
        indices_of_new_medoids = __numba_voronoi_update_medoids(labels, distance_matrix, n_medoids, multiplicity)

        # Check if end
        if __numba_voronoi_stop(iteration_number, maximum_iterations,
                                distance_matrix, indices_of_old_medoids, indices_of_new_medoids, n_medoids, tolerance):
            break
        else:
            indices_of_old_medoids = indices_of_new_medoids

    return indices_of_new_medoids, labels


@njit
def __numba_voronoi_update_clusters(distance_matrix, indices_of_current_medoids, n_samples):
    """
    Find the new clusters given the medoids

    Parameters
    ----------
    distance_matrix: np.array of size (n_samples, n_samples)
        The distance matrix
    indices_of_current_medoids: np.array of size (n_medoids,)
        The indices of the medoids
    n_samples: int
        The value of n_samples

    Returns
    -------
    labels: np.array of size (n_samples,)
        The clusters. Datapoint with index `di` is associated to the medoid with index
        `indices_of_current_medoids[labels[di]]`
    """
    labels = np.empty((n_samples,), dtype=np.int32)
    for dpi in range(n_samples):
        labels[dpi] = np.argmin(np.array([distance_matrix[dpi, mi] for mi in indices_of_current_medoids]))
    return labels


@njit
def __numba_voronoi_update_medoids(labels, distance_matrix, n_medoids, multiplicity):
    """
    Find the medoids of the clusters given the current labels

    Parameters
    ----------
    labels: np.array of size (n_samples,)
        The label of each datapoint
    distance_matrix: np.array of size(n_samples, n_samples)
        The distance matrix
    n_medoids: int
        The value of n_medoids
    multiplicity: np.array of size (n_samples,)
        The multiplicity of each datapoint

    Returns
    -------
    new_medoids: np.array of size(n_medoids)
        The new medoids, given the labels
    """
    new_medoid_indices = np.empty((n_medoids,), dtype=np.int32)
    for i in range(n_medoids):
        cluster_indices = np.where(labels == i)[0]
        best_index = -1
        best_cost = np.inf

        # new_meds[i] = argmin_{r \in R_i} (sum_{r' in R_i} D(r')*\delta(r',r))
        for r in cluster_indices:

            cost = np.sum(np.array(
                [multiplicity[ri] * distance_matrix[r, ri] for ri in cluster_indices]
            ))

            if cost < best_cost:
                best_cost = cost
                best_index = r

        new_medoid_indices[i] = best_index
    return new_medoid_indices


@njit
def __numba_voronoi_stop(iteration_number, max_iterations, distance_matrix, indices_of_current_medoids,
                         indices_of_new_medoids,
                         n_medoids, epsilon):
    """
    checks whether to stop k-medoids.

    Parameters
    ----------
    iteration_number: int
        The number of finished iterations
    max_iterations: int
        The maximum number of iterations
    distance_matrix: np.ndarray of size (n_samples, n_samples)
        The distance matrix
    indices_of_current_medoids: np.ndarray of size (n_medoids)
        The indices in the distance matrix with the old medoids
    indices_of_new_medoids: np.ndarray of size (n_medoids)
        The indices in the distance matrix with the new medoids
    n_medoids: int
        The value of n_medoids
    epsilon: float
        The maximum distance between two successive medoids


    Returns
    -------
    ret: bool
        True if all successive medoids are at most `epsilon` apart. False otherwise
    """
    if iteration_number >= max_iterations:
        return True

    for i in range(n_medoids):
        if distance_matrix[indices_of_current_medoids[i], indices_of_new_medoids[i]] > epsilon:
            return False
    return True


def kmedoids_plus_plus(distance_matrix, n_medoids, multiplicity=None, initial_medoid_index=0, random_state=0):
    """
    k-medoids++ algorithm

    Parameters
    ----------
    distance_matrix: np.array of size (n_samples, n_samples)
        The distance matrix
    n_medoids: int
        The desired number of medoids (n_medoids)
    multiplicity: np.array of size (n_samples,)
        The multiplicity of the datapoints
    initial_medoid_index: int
        Index of the first medoid
    random_state: int
        Seed for np.random

    Returns
    -------
    medoid_indices: np.array of size (n_medoids,)
        The initial medoids for clustering.
    """
    n_samples = len(distance_matrix)
    medoid_indices = [initial_medoid_index]
    non_medoid_indices = list(range(n_samples))
    non_medoid_indices.remove(initial_medoid_index)

    if multiplicity is None:
        multiplicity = np.ones(shape=(n_samples,))

    old_state = np.random.get_state()

    np.random.seed(random_state)
    while len(medoid_indices) < n_medoids:
        probabilities = np.array([min([multiplicity[ri] * distance_matrix[mi, ri] for mi in medoid_indices])
                                  for ri in non_medoid_indices])
        new_medoid_index = np.random.choice(non_medoid_indices, size=1, p=probabilities / sum(probabilities))[0]
        medoid_indices.append(new_medoid_index)
        non_medoid_indices.remove(new_medoid_index)
    np.random.set_state(old_state)

    return np.array(medoid_indices)


class Voronoi:

    def __init__(self, maximum_iterations, tolerance, data_type='distance_matrix', n_clusters=None,
                 kmpp_seed=0, distance_metric=None, **pdist_kwargs):

        """
        Base Voronoi algorithm wrapper.

        Parameters
        ----------
        maximum_iterations: int
            Maximum number of iterations for the Voronoi algorithm
        tolerance: float
            Maximum difference in medoids for the Voronoi algorithm.
        data_type: str
            Input type for fit. Must be `data_type_distance_matrix` or `data_type_feature_values`
        n_clusters: int
            Number of clusters (k). Must be positive
        kmpp_seed: int or None
            Seed to use in k-medoids++. Used if no initial_medoids_indices are given to fit
        distance_metric: str or callable
            Distance metric applicable to two datapoints. Must be accepted by scipy's pdist.

        Other Parameters
        ----------------
        See scipy.spatial.distance.pdist for key-worded arguments
        """

        if data_type == data_type_feature_values:
            if distance_metric is None:
                raise ValueError(f'distance_metric may not be None if data_type is {data_type_feature_values}')
            self.distance_metric = distance_metric
            self.pdist_kwargs = pdist_kwargs
        elif data_type == datatype_distance_matrix:
            if distance_metric is not None:
                warn(f'distance_metric is ignored when data_type = {datatype_distance_matrix}')
            self.distance_metric = None
            self.pdist_kwargs = None
        else:
            raise ValueError(f'unknown value for data_type: {data_type}')
        self.data_type = data_type

        assert n_clusters > 0 and isinstance(n_clusters, int)
        self.n_medoids = n_clusters

        assert kmpp_seed is None or isinstance(kmpp_seed, int)
        self.kmpp_seed = kmpp_seed

        assert maximum_iterations > 0 and isinstance(maximum_iterations, int)
        self.maximum_iterations = maximum_iterations

        assert 0 <= tolerance <= 1
        self.tolerance = tolerance

        self.training_data = None
        self.medoid_indices = None
        self.labels_ = None

    def fit(self, x, indices_of_initial_medoids=None):
        """
        Cluster the given data.

        Parameters
        ----------
        x: np.array of size (n_samples, n_samples) or np.array of size (n_samples, n_features)
            Distance matrix or feature values (depends on initialization).
        indices_of_initial_medoids: np.array of size (k,) or None
            Indices of initial medoids. Defaults to k-medoids-plus-plus
        multiplicity: np.array of size (n_samples) or None
            Multiplicity of initial medoids. Defaults to 1 for each medoid.

        Notes
        -----
        Stores training_data, medoid indices and labels as `self.training_data`, `self.medoids` and `self.labels_`, such
        that the datapoint with index di is part of the cluster that has medoid
        `self.training_data[self.medoid_indices[self.labels_[di]]]`

        """

        self.training_data = x

        if self.data_type == data_type_feature_values:
            distance_matrix = squareform(pdist(x, metric=self.distance_metric), **self.pdist_kwargs)
        else:
            assert x.shape[0] == x.shape[1]
            distance_matrix = x

        if indices_of_initial_medoids is None:
            indices_of_initial_medoids = kmedoids_plus_plus(distance_matrix=distance_matrix,
                                                            n_medoids=self.n_medoids,
                                                            random_state=self.kmpp_seed)

        self.medoid_indices, self.labels_ = self.implementation(distance_matrix=distance_matrix,
                                                                indices_of_initial_medoids=indices_of_initial_medoids)

    @abc.abstractmethod
    def implementation(self, distance_matrix, indices_of_initial_medoids):
        pass


class NumbaVoronoi(Voronoi):
    def implementation(self, distance_matrix, indices_of_initial_medoids):
        return numba_voronoi(distance_matrix=distance_matrix, indices_of_initial_medoids=indices_of_initial_medoids,
                             maximum_iterations=self.maximum_iterations, tolerance=self.tolerance)

