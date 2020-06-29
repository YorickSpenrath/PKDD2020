import math
import numpy as np
from numba import njit
from scipy import optimize
from cjo.weighted_adapted_jaccard.distances.implementation import DSX, distance_function
from functions.voronoi_clustering import numba_voronoi


class IEMFactory:
    LOG = 'nlw'

    modes = [LOG]

    def __init__(self, mode):
        """
        This is a factory class, that creates IEM functions a clustering and dataset are given.
        Parameters
        ----------
        mode
        """
        assert mode in IEMFactory.modes, f'Given mode {mode} not known. Select one of {IEMFactory.modes}'
        self.implementation = {
            IEMFactory.LOG: self.__log
        }[mode]

    def create_function(self, dataset, clustering):
        """
        Create a new IEM function given a dataset and its clustering.

        Parameters
        ----------
        dataset: DSX
            The full dataset
        clustering: iterable of DSX
            The clusters of the dataset

        Returns
        -------
        func: callable
            The IEM that takes a weights vector as input
        """
        assert isinstance(dataset, DSX)
        dataset.assert_partition(clustering)
        return self.implementation(dataset, clustering)

    @staticmethod
    def __log(dataset, clustering):
        def function(weights_vector):
            return math.log(sum([1 / 2 / dsx_i.size_d * dsx_i.get_sum(weights_vector) for dsx_i in clustering])) - \
                   math.log(dataset.get_sum(weights_vector))

        return function


class DSXWeightsOptimizer:
    DIFFERENTIAL_EVOLUTION = 'de'
    BRUTE_FORCE = 'bf'

    strategy_options = [DIFFERENTIAL_EVOLUTION, BRUTE_FORCE]

    def __init__(self, w_max, h, strategy, **kwargs):
        """
        Optimization wrapper for DSX Weights optimization.

        Parameters
        ----------
        w_max: int
            Maximum value a weight can get
        h: int
            Number of regular super-categories
        strategy: str
            The optimization strategy.

            -- de or DSXWeightsOptimizer.differential_evolution for scipy differential evolution
            -- bf or DSXWeightsOptimizer.brute_force for scipy brute force

        Other Parameters
        ----------------
        ISC0: bool, default = True
            Whether the weight of base.stringconstants.ISC_WEIGHT (the inter-super-category distance weight) can be 0.
            If True, all others cannot be 0. This is enforced by setting the minimum of the respective weight to 1
            instead of 0.
        seed: int, default = 0
            Random state, used in 'de' strategy. Can be overriden on method call.
        max_evaluations: int, default = 1e6
            Maximum number of evaluations, used in the 'bf' strategy.
        ns: int, default = 10
            Number of values per weights evaluated, used in the 'bf' strategy if 'max_evaluations' is None

        """
        assert isinstance(w_max, int)
        assert w_max > 0
        assert isinstance(h, int)

        if kwargs.get('ISC0', False):
            self.bounds = [(0, w_max)] + [(1, w_max)] * h
        else:
            self.bounds = [(1, w_max)] + [(0, w_max)] * h
        self.kwargs = kwargs

        assert strategy in DSXWeightsOptimizer.strategy_options, 'Given strategy not known.'
        self.minimization = {DSXWeightsOptimizer.DIFFERENTIAL_EVOLUTION: self.__genetic,
                             DSXWeightsOptimizer.BRUTE_FORCE: self.__brute_force}[strategy]

    def get_optimal_weights(self, func, **kwargs):
        """
        Get the optimal weights for the given function.

        Parameters
        ----------
        func: callable
            The function to minimize. The input of the function is expected to be an Number array of length :math:`h+1`

        Other Parameters
        ----------------
        seed: int
            Random state, used in 'de' strategy. If not given, the global setting is used.

        Returns
        -------
        weights: np.ndarray of length :math:`h + 1`
            The result of the minimization

        """
        return self.minimization(func, **kwargs)

    def __genetic(self, func, **kwargs):
        seed = kwargs.get('seed', self.kwargs.get('seed', 0))
        return optimize.differential_evolution(func, self.bounds, seed=seed, polish=False).x

    # noinspection PyUnusedLocal
    def __brute_force(self, func, **kwargs):
        if self.kwargs.get('max_evaluations', 1e6) is not None:
            n = int(self.kwargs.get('max_evaluations', 1e6) ** (1 / len(self.bounds)))
        else:
            n = self.kwargs.get('ns', 10)

        return optimize.brute(func, ranges=self.bounds, Ns=n, full_output=False, finish=None)


class DSXClustering:
    NUMBA_VORONOI = 'numba_voronoi'
    ALL_MODES = [NUMBA_VORONOI]

    def __init__(self, mode, max_iterations=1e99, tolerance=0.001):
        """
        Clustering method used in the algorithm. The selected mode is the actual implementation.

        Parameters
        ----------
        mode: str
            Implementation of clustering algorithm to use. Must be in DSXClustering.ALL_MODES
        max_iterations: Numeric
            Maximum number of iterations of the clustering algorithm
        tolerance: float
            Maximum distance between two sequential medoids to stop before max_iterations
        """
        self.implementation = {
            DSXClustering.NUMBA_VORONOI: self.__numba
        }[mode]
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def cluster(self, dsx, weights, initial_medoids=None, k=None):
        """
            Find a clustering of size :math:`k` given the full dataset :math:`R'` and weights `\\vec{w}`.
            Can be initiated with medoids or k. If not initiated with medoids, k means ++ is used.
            The implementation used is given is determined at the constructor

            Parameters
            ----------
            dsx: DSX
                The full dataset, :math:`R'`
            weights: dict of str to Number, iterable of Number
                The weights to be used. See :py:meth:`validate_and_convert_weights()
                <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
            initial_medoids: iterable of int, optional
                The initial medoids to be used. See also the Notes.
            k: int, optional
                The number of clusters. See also the Notes

            Returns
            -------
            medoids: list of int
                The found medoids.
            dsx_clusters: list of DSX
                The found clusters, as DSX objects.

            Raises
            ------
            AssertionError
                If a given initial_medoid is not in dsx.
                If k and initial_medoids are given, but k != len(initial_medoids).
            ValueError
                If both k and initial_medoids are None.

            Notes
            -----
            If initial_medoids and k are given, the length of initial_medoids must be k. If only k is given, initial
            medoids are chosen using :py:meth:`kmedoids_plus_plus
            <cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers.DSXClustering.kmedoids_plus_plus>`. If only the
            initial_medoids are given, those are used. If neither are given, a ValueError is raised.
            """
        # Check input
        assert isinstance(dsx, DSX)
        weights = dsx.validate_and_convert_weights(weights)

        # Check initial medoids
        if k is not None and initial_medoids is not None:
            assert len(initial_medoids) == k, 'initial_medoids and k given, but values do not match'
        elif k is None and initial_medoids is not None:
            pass
        elif k is not None and initial_medoids is None:
            initial_medoids = self.kmedoids_plus_plus(dsx, weights, None, k)
        else:
            raise ValueError('Missing both: initial_medoids and k')

        for rp in initial_medoids:
            assert rp in dsx, f'Datapoint {rp} not in dataset'

        # Run kmedoids
        medoids, clusters = self.implementation(dsx, weights, initial_medoids)

        # put results back into dsx objects
        dsx_clusters = []
        for cluster in clusters:
            dsx_clusters.append(dsx.get_subset(cluster))
        return medoids, dsx_clusters

    def __numba(self, dsx, weights, initial_medoids):
        """using VORONOI implemented in Numba"""
        r_values = dsx.rp_values

        # TODO : If you extend the NumbaVoronoi Clustering class with multiplicity, you can skip these conversions
        initial_medoid_indices = np.array([dsx.rp_values.tolist().index(m) for m in initial_medoids])
        dm = dsx.get_distance_matrix_over_r(weights)
        final_medoid_indices, labels = numba_voronoi(distance_matrix=dm,
                                                     indices_of_initial_medoids=initial_medoid_indices,
                                                     maximum_iterations=self.max_iterations,
                                                     tolerance=self.tolerance,
                                                     multiplicity=dsx.drp_values)
        medoids = [r_values[i] for i in final_medoid_indices]
        clusters = [r_values[np.where(labels == i)] for i in range(len(initial_medoids))]
        return medoids, clusters

    @staticmethod
    def kmedoids_plus_plus(dsx, weights, initial_medoid=None, k=5):
        """
        Implements an altered version of kmeans plus plus using the WAJ distance metric

        Parameters
        ----------
        dsx: DSX
            The dataset for which to get the initial medoids.
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        initial_medoid: int or None
            The initial receipt that is the first medoid. If None, the smallest value in dsx is chosen. Needs to be in
            dsx
        k: int
            The number of medoids

        Returns
        -------
        medoids: list of int
            The k medoids chosen by this algorithm

        Raises
        ------
        AssertionError
            If the given initial_medoid is not in dsx

        Notes
        -----
        Temporarily changes the random state of np to ensure deterministic calls. Restores the random state afterwards.

        """
        assert isinstance(dsx, DSX)
        d = dsx.get_pure_distance_metric(weights)
        if initial_medoid is None:
            initial_medoid = min(dsx.keys())
        assert initial_medoid in dsx, f'First medoid {initial_medoid} not in dsx'
        assert isinstance(k, int)

        medoids = [initial_medoid]
        other_points = list(dsx.keys())
        other_points.remove(initial_medoid)

        state = np.random.get_state()
        np.random.seed(0)
        while len(medoids) < k:
            other_distances = np.array([min([d(a, m) for m in medoids]) for a in other_points])
            new_medoid = int(np.random.choice(other_points, 1, p=other_distances / sum(other_distances))[0])
            medoids.append(new_medoid)
            other_points.remove(new_medoid)

        # noinspection PyTypeChecker
        np.random.set_state(state)

        return medoids
