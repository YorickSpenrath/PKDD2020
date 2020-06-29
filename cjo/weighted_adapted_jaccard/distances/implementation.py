from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

from cjo.base import logs
from cjo.base.hierarchy import Hierarchy
from cjo.base.stringconstants import supercat, cat, ISC_WEIGHT, multiplicity
from functions.general_functions import listified, assert_valid_partition
from cjo.weighted_adapted_jaccard.distances import bitops
from cjo.weighted_adapted_jaccard.distances.bitops import hamming_weight


def al2dsx(dataset, hierarchy, additional_rp=None):
    """
    Wrapper to adapt the dataset and the hierarchy to the new formulations of Subsection \\ref{sec:compopt:ws2w}

    Parameters
    ----------
    dataset: Counter, logs.ActivationLog, str, or Path.
        The value of :math:`\mathcal{D'}`. If ActivationLog, str or Path, it is (imported and) transformed into a bag
    hierarchy: hierarchy.Hierarchy
        Hierarchy object defining the relations between (super-)categories
    additional_rp: Counter
        Additional r_prime values to be added apart from the dataset. These values are not part of visit_ids_per_rp

    Returns
    -------
    dataset: DSX
        D initiated with the adaption presented in Subsection \\ref{sec:compopt:ws2w}

        - :math:`r'`, :math:`\mathcal{R}'`
        - hierarchy class as given
        - invoice_ids_per_rp, reference to original iids that belong to each rp
    """
    # import the data
    if isinstance(dataset, str) or isinstance(dataset, Path):
        dataset = logs.ActivationLog(dataset)
    else:
        assert isinstance(dataset, logs.ActivationLog)

    assert isinstance(hierarchy, Hierarchy)

    if additional_rp is None:
        additional_rp = Counter()
    else:
        assert isinstance(additional_rp, Counter)

    df = dataset.df.copy()

    h_vector = [bitops.subset2int(hierarchy[k], hierarchy.c) for k in hierarchy.sc]

    def row2rp(row):
        r = int(''.join([str(row[i]) for i in hierarchy.c]), 2)
        return r + (int(''.join([str(1 if (hamming_weight(ci & r) > 0) else 0) for ci in h_vector]),
                        2) << hierarchy.num_cat)

    df['rp'] = df.apply(row2rp, axis=1)
    dataset = Counter(df['rp'])

    dataset += additional_rp
    # noinspection PyTypeChecker
    visit_ids_per_rp = {v: list(df[df['rp'] == v].index) for v in df['rp'].unique()}

    return DSX(dataset, hierarchy, visit_ids_per_rp)


class DSX:
    def __init__(self, dataset, hierarchy, visit_ids_per_rp=None):
        """
        This class tracks the dataset, hierarchy, and num_cat, such that all assertions are done once. It is meant to
        represent :math:`X`, though we track the hierarchy in order allow computations of :math:`s(r)`, and as such of
        :math:`X^2|_z`. The given input is based on :math:`r' \in R'`, as discussed in Subsection
        \\ref{sec:compopt:ws2w}.

        Parameters
        ----------
        dataset : Counter{np.uint64}
            The frequency of each representation (the value of :math:`\mathcal{D'}`)
        hierarchy: Hierarchy
            The hierarchy used for this DSX
        visit_ids_per_rp: dict : int -> (list of str) or None
            The visit_ids per rp value

        Raises
        ------
        AssertionError
            If the largest original receipt in the dataset is not smaller than :math:`2^{|\mathcal{C}|}`.

            If the smallest original receipt in the dataset is not larger than 0.

            If the sum of hierarchy_vector_prime[1:] is not equal to :math:`2^{|\mathcal{C}|} - 1` or if any of the
            values of the hierarchy_vector_prime is 0. (i.e. if the hierarchy is not a partition over the categories).

            If the first value of the hierarchy_vector_prime is not equal to
            :math:`(2^h) \ll |\mathcal{C}|`.

            If the first value of the hierarchy_names_prime is not equal to :math:`c_0`.
        """
        num_cat = hierarchy.num_cat
        h_vector = [bitops.subset2int(hierarchy[k], hierarchy.c) for k in hierarchy.sc]
        hierarchy_vector_prime = [((2 ** hierarchy.h - 1) << hierarchy.num_cat)] + h_vector
        hierarchy_names_prime = [ISC_WEIGHT] + hierarchy.sc
        self.hierarchy = hierarchy

        # INPUT TYPE VERIFICATION #
        assert isinstance(dataset, Counter)
        assert all(isinstance(di, int) for di in dataset.keys())
        hierarchy_vector_prime = np.array(hierarchy_vector_prime, dtype=np.uint64)
        assert isinstance(hierarchy_names_prime, list)
        hierarchy_names_prime = listified(hierarchy_names_prime, str)
        assert isinstance(num_cat, int)

        if visit_ids_per_rp is None:
            self.visit_ids_per_rp = {rp: [] for rp in dataset.keys()}
        else:
            assert isinstance(visit_ids_per_rp, dict)
            self.visit_ids_per_rp = visit_ids_per_rp

        h = len(hierarchy_names_prime) - 1

        # DATASET VERIFICATION #
        original_receipts = {r_prime & (2 ** num_cat - 1) for r_prime in dataset.keys()}
        # All original datapoints must be a subset of C
        assert max(original_receipts) < 2 ** num_cat, \
            'maximum category encoding is not smaller than 2 ^ number of categories'
        # Datapoints must be larger than 0 (non-empty receipts)
        assert min(original_receipts) > 0, 'Encodings should be bigger than 0'
        self.__dataset = dataset

        # HIERARCHY VERIFICATION #
        # after the first index
        assert sum(hierarchy_vector_prime[1:]) == 2 ** num_cat - 1, \
            'hierarchy should contain all categories at least once (i.e. its sum should be 2 ** num_cat - 1)'
        assert all([h > 0 for h in hierarchy_vector_prime]), \
            'hierarchy cannot contain empty sets (i.e. 0 valued integer representations)'
        # on the first index
        assert hierarchy_vector_prime[0] == (2 ** h - 1) << num_cat, \
            'the first value of the hierarchy vector needs to be' \
            ' (2 ** h - 1) << num_cat'
        assert hierarchy_names_prime[0] == ISC_WEIGHT, f'first super-category should be "{ISC_WEIGHT}"'

        # DATA IS OKAY, STORE IT
        self.__dataset = dataset
        self.hierarchy_vector = hierarchy_vector_prime
        self.__hierarchy_names = hierarchy_names_prime
        self.__num_cat = num_cat
        self.__h = h

        # OPTIMIZE : Precompute only useful z-values, not all
        # You are currently "precomputing" all values of z and z', but this might not be necessary, as some might not
        # have pairs of receipts a', b' such that z = s'(a') | s'(b').

        self.rp_values = None
        self.drp_values = None
        self.__zp_values = None
        self.__d_ci_x_z_matrix = None
        self.prep()

    def prep(self):
        self.rp_values = np.array(list(self.__dataset.keys()), dtype=np.uint64)
        self.drp_values = np.array([self[r] for r in self.rp_values], dtype=np.uint64)
        self.__d_ci_x_z_matrix = precompute_stuff(h=self.__h,
                                                  rp_values=self.rp_values,
                                                  drp_values=self.drp_values,
                                                  num_cat=self.__num_cat,
                                                  hierarchy_vector=self.hierarchy_vector)
        self.__zp_values = np.arange(2 ** self.__h, 2 ** (self.__h + 1))

    def get_distance_matrix_over_d(self, weights):
        w = self.validate_and_convert_weights(weights)
        return compute_distance_matrix(rp_values=self.pure_values, w=w, hierarchy_vector=self.hierarchy_vector)

    def get_distance_matrix_over_r(self, weights):
        w = self.validate_and_convert_weights(weights)
        return compute_distance_matrix(rp_values=self.rp_values, w=w, hierarchy_vector=self.hierarchy_vector)

    @property
    def dataset(self):
        return self.__dataset.copy()

    @property
    def original_df_r(self):
        """
        A DataFrame describing the original values of :math:`\\mathcal{R}`.

        Returns
        -------
        original_r: pd.DataFrame
            DataFrame with the values of :math:`r'` as index, and the respective values of :math:`\\mathcal{D}(r')`,
            :math:`r` and :math:`\\vec{s}(r)` as columns.
        """
        df = pd.DataFrame(columns=[multiplicity, cat, supercat])
        df.index.name = "r'"
        for k, v in self.__dataset.items():
            df.loc[k, multiplicity] = v
            df.loc[k, cat] = k & (2 ** self.__num_cat - 1)
            df.loc[k, supercat] = k >> self.__num_cat
        df = df.astype(int)
        return df

    def get_iids(self, rp_value):
        return self.visit_ids_per_rp[rp_value]

    @property
    def all_iids(self):
        return sum(self.visit_ids_per_rp.values(), [])

    @property
    def d_ci_x_z_matrix(self):
        return self.__d_ci_x_z_matrix

    def __contains__(self, item):
        return item in self.__dataset

    def __str__(self):
        return f'dsx object with {self.size_r} [{self.size_d}] receipts'

    def __repr__(self):
        return str(self)

    @property
    def hierarchy_names(self):
        return self.__hierarchy_names

    @property
    def size_d(self):
        return sum(self.__dataset.values())

    @property
    def size_r(self):
        return len(self.__dataset)

    def unit_weight_vector(self):
        """
        Generates a weight vector that is valid for this DSX, and where each weight is 1

        Returns
        -------
        weights: np.ndarray
            weights vector for this DSX, where each weight is 1.

        """
        return self.validate_and_convert_weights([1] * (self.__h + 1))

    def get_named_weight_vector(self, weights):
        """
        Converts a weight vector to mapping from this hierarchy to the weights.

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.

        Returns
        -------
        w: dict from str to Number
            Mapping from each of super-category to its weight in the given weight vector.
        """
        weights = self.validate_and_convert_weights(weights)
        return {hi: wi for hi, wi in zip(self.__hierarchy_names, weights)}

    @property
    def h(self):
        return self.__h

    def medoid(self, weights):
        m = self.get_distance_matrix_over_r(weights)
        distance_matrix_prime = m.dot(self.drp_values.T)

        argmin = distance_matrix_prime.argmin()

        return self.rp_values[argmin]

    def medoid_and_avg_distance(self, weights):
        m = self.get_distance_matrix_over_r(weights)
        distance_matrix_prime = m.dot(self.drp_values.T)

        argmin = distance_matrix_prime.argmin()

        return self.rp_values[argmin], distance_matrix_prime[argmin] / self.size_d

    def rp_closest_to_other_dsx_medoid_and_distances(self, other, weights):
        # Input
        assert isinstance(other, DSX)
        assert self.hierarchy == other.hierarchy
        w = self.validate_and_convert_weights(weights)

        m = compute_double_distance_matrix(rp_values_a=self.rp_values,
                                           rp_values_b=np.array([other.medoid(weights)]),
                                           w=w,
                                           hierarchy_vector=self.hierarchy_vector
                                           )[:, 0]
        idx = m.argmin()
        rp = self.rp_values[idx]
        dist_other = m[idx]
        dist_self = self.get_pure_distance_metric(weights=weights)(rp, self.medoid(weights))
        return rp, dist_self, dist_other

    def __getitem__(self, rp):
        """
        Gets the number of visits represented by the receipt :math:`r'`, i.e. the value :math:`\mathcal{D'}(r')`

        Parameters
        ----------
        rp: int
            The receipt :math:`r'` for which to get the number of visits represented by it

        Returns
        -------
        count: int
            The value of :math: `$\dataset'(r')$`
        """
        return self.__dataset[rp]

    def keys(self):
        """
        Gets the receipt values of this dataset; the value of :math:`\mathcal{R}'`

        Returns
        -------
        R: Set of int
            The set :math:`\mathcal{R}'`
        """
        return self.__dataset.keys()

    def get_subset(self, rp_values):
        """
        Generate a DSX object for the given subset :math:`k\subseteq\mathcal{R}'`

        Parameters
        ----------
        rp_values: iterable of Int
            The representations for which to get the new DSX

        Returns
        -------
        x: DSX
            Subset of this dataset
        """
        return DSX(dataset=Counter({rp: v for rp, v in self.__dataset.items() if rp in rp_values}),
                   hierarchy=self.hierarchy,
                   visit_ids_per_rp={rp: v for rp, v in self.visit_ids_per_rp.items() if rp in rp_values})

    def assert_partition(self, clustering):
        """
        Asserts that the given clustering is a valid partition over this DSX

        Parameters
        ----------
        clustering: iterable of DSX
            collection of DSX that is the partition

        Raises
        ------
        AssertionError
            If the given collection of DSX is not a valid partition over this DSX

        """
        clustering = listified(clustering, DSX)
        assert_valid_partition(set(self.keys()), [set(dsx_i.keys()) for dsx_i in clustering])

    def get_pure_distance_metric(self, weights):
        """
        Generate a distance metric function for the given weights, not correcting for multiplicity. This is useful
        when making comparisons on two single visits (i.e. on a basis of :math:`\\mathcal{D}`)

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        """
        return self.__get_distance_metric(weights, False)

    def __get_distance_metric(self, weights, multi):
        w = self.validate_and_convert_weights(weights)

        if multi:
            def inner(ap, bp):
                return distance_function(ap, bp, self.hierarchy_vector, w, self[ap], self[bp])
        else:
            def inner(ap, bp):
                return distance_function(ap, bp, self.hierarchy_vector, w, 1, 1)

        return inner

    def get_sum(self, weights):
        """
        Implementation of Equation \\ref{eq:distance:dwxz:bitp:tensor}.

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.

        Returns
        -------
        dwx: float
            The value of :math:`\delta_{\\vec{w}}(X)`
        """
        w = self.validate_and_convert_weights(weights)

        # Dot product for the numerator vector
        n = np.dot(self.__d_ci_x_z_matrix, w)

        # Bitwise operations for the denominator vector
        # In the text, there is a Z'-matrix, but this is less trouble
        d = np.zeros(len(self.__zp_values))
        for i in range(self.__h + 1):
            d += np.bitwise_and(self.__zp_values >> (self.__h - i), 1) * w[i]

        # If w0 is 0, then d[0] will be 0 as well. By its definition, n[0] will be 0 too, as such the following
        # operation prevents a division by 0, while maintaining correctness
        d[0] += 1

        return np.sum(n / d)

    def remove(self, receipts_to_be_removed):
        if not isinstance(receipts_to_be_removed, Iterable):
            receipts_to_be_removed = [receipts_to_be_removed]
        receipts_to_be_removed = Counter(receipts_to_be_removed)
        for receipt, m in receipts_to_be_removed.items():
            assert self.__dataset[receipt] >= m
        self.__dataset -= receipts_to_be_removed
        self.prep()

    def validate_and_convert_weights(self, weights):
        """
        Validates and converts the given weights

        Parameters
        ----------
        weights: dict of str to float, iterable of float
            The weights to be used.
            If dict or pd.Series, missing weights are treated as 0. The value of base.stringconstants.ISC_WEIGHT or all
            others must be non-zero. If iterable, the length must be :math:`h+1`. The first or all other values must be
            non-zero.

        Returns
        -------
        w: np.ndarray
            Vector of weights, validated to match the requirements.

        Raises
        ------
        AssertionError
            If the weights are an iterable of the wrong length. If any of the weights is negative. If the weight of
            base.stringconstants.ISC_WEIGHT (the first index) is not positive and the rest of the weights are
            non-positive.

        """
        # convert series to dict
        if isinstance(weights, pd.Series):
            weights = weights.to_dict()

        # convert dict to vector of appropriate order
        if isinstance(weights, dict):
            w = weights.copy()
            weights = [w.pop(k, 0) for k in self.__hierarchy_names]
            assert len(w) == 0, f'Found unknown weights : {w.keys()}'

        # verify vector
        assert len(weights) == len(self.hierarchy_vector), \
            f'Given weight vector is of wrong length (is {len(weights)}, should be {len(self.hierarchy_vector)}'
        assert all([wi >= 0 for wi in weights]), 'Weights should be non-negative'
        assert (weights[0] > 0 or all([wi > 0 for wi in weights[1:]])), \
            'The first or all other weights should be positive'

        return np.array(weights)

    @property
    def pure_values(self):
        return np.array(sum([[r] * self.__dataset[r] for r in self.keys()], []), dtype=np.uint64)


@njit
def distance_function(ap, bp, hierarchy_vector, w, freq_a, freq_b):
    """
    Computes the distance between ap and bp for a predefined set of weights.

    Parameters
    ----------
    ap: int
        Receipt :math:`a'`
    bp: int
        Receipt :math:`a'`
    hierarchy_vector: list of int
        :math:`c_i` values
    w: list of float
        :math:`w_i` values
    freq_a: int
        multiplicity of receipt :math:`a`
    freq_b: int
        multiplicity of receipt :math:`b`

    Returns
    -------
    d: float
        The distance :math:`\\delta_{\\vec{w}}(a',b')` between :math:`a'` and :math:`b`

    """

    distance_numerator = 0
    distance_denominator = 0

    ap_and_bp = ap & bp
    ap_or_bp = ap | bp

    for ci, wi in zip(hierarchy_vector, w):
        sc_denominator = hamming_weight(ci & ap_or_bp)
        if sc_denominator != 0:
            sc_numerator = hamming_weight(ci & ap_and_bp)
            distance_numerator += wi * (1 - sc_numerator / sc_denominator)
            distance_denominator += wi
    return freq_a * freq_b * distance_numerator / distance_denominator


@njit
def compute_distance_matrix(rp_values, w, hierarchy_vector):
    ret = np.zeros((rp_values.shape[0], rp_values.shape[0]))

    for i in range(len(rp_values)):
        for j in range(i + 1, len(rp_values)):
            ret[i, j] = distance_function(ap=rp_values[i], bp=rp_values[j],
                                          hierarchy_vector=hierarchy_vector, w=w, freq_a=1, freq_b=1)
            ret[j, i] = ret[i, j]

    return ret


@njit
def compute_double_distance_matrix(rp_values_a, rp_values_b, w, hierarchy_vector):
    ret = np.zeros((rp_values_a.shape[0], rp_values_b.shape[0]))

    for i in range(len(rp_values_a)):
        for j in range(len(rp_values_b)):
            ret[i, j] = distance_function(ap=rp_values_a[i], bp=rp_values_b[j],
                                          hierarchy_vector=hierarchy_vector, w=w, freq_a=1, freq_b=1)

    return ret


@njit
def precompute_stuff(h, rp_values, num_cat, hierarchy_vector, drp_values):
    """
    The actual precomputation of the :math:`\delta_{c_i}(X^2|_z)` values. The return value is the
    :math:`\delta_c(X)` matrix of equation \\ref{eq:distance:dwx:tensor}. Each value of this matrix computed using
    equation \\ref{eq:distance:dcix2z:bitp}. As a result, this method also implements \\ref{eq:distance:dciab:bit}.
    Parameters
    ----------
    h : int
        The number of actual super-categories (not accounting for the special super-category)
    rp_values : iterable of int
        The value of :math:`\mathcal{R'}`
    num_cat : int
        The number of actual categories
    hierarchy_vector : iterable of int
        The values of :math:`c_0, c_1, ..., c_h`
    drp_values : iterable of int
        The value of :math:`\mathcal{D'}`, ordered in the same way as r_values
    Returns
    -------
    d_c_x_matrix : np.array of size :math:`2^h` by :math:`(h+1)`
        The matrix :math:`\delta(X)`
    """
    d_c_x_matrix = np.zeros((2 ** h, h + 1))
    for ia in range(len(rp_values)):
        for ib in range(ia + 1, len(rp_values)):
            ap = rp_values[ia]
            bp = rp_values[ib]
            z = (ap >> num_cat) | (bp >> num_cat)
            for i in range(len(hierarchy_vector)):
                d = hamming_weight(hierarchy_vector[i] & (ap | bp))
                aj = (0 if d == 0 else (1 - hamming_weight(hierarchy_vector[i] & (ap & bp)) / d))
                d_c_x_matrix[z, i] += drp_values[ia] * drp_values[ib] * aj
    return d_c_x_matrix * 2
