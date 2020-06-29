import string
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from cjo.weighted_adapted_jaccard.distances.implementation import DSX
from cjo.weighted_adapted_jaccard.result_computations.clustering_computation import \
    create_clusters_and_assign_them_to_given_medoids
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult
from cjo.weighted_adapted_jaccard.bootstrap import bootstraphelpers
from cjo.base.stringconstants import DATASET_NAME, invoice, medoid, multiplicity, super_medoid, ALGORITHM, OUR_FRAMEWORK
from functions import file_functions
from functions import dataframe_operations

from data import data_loader
from functions.progress import ProgressShower
from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS

iem_factory = bootstraphelpers.IEMFactory(bootstraphelpers.IEMFactory.LOG)

# Algorithm parameters
hac_complete = 'HAC - complete'
hac_average = 'HAC - average'
hac_single = 'HAC - single'
voronoi = 'Voronoi'
spectral = 'Spectral'
hac_algorithms = [hac_complete, hac_average, hac_single]
competitors = [voronoi, spectral] + hac_algorithms
all_algorithms = [OUR_FRAMEWORK] + competitors


# This is the computation script of the third experiment: Competitor Analysis.

def sk_learn_wrapper(clustering_instance, w, dsx, matrix):
    """
    Compute the score of a sklearn clustering algorithm on a dataset.

    Parameters
    ----------
    clustering_instance: Scikit-Learn Clustering Algorithm Object
        An object of the scikit-learn clustering algorithms
    w: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
    dsx: DSX
        The DSX object of the dataset. This is used to create clusters subsets for the IEM computation
    matrix: np.ndarray of size (n_visits, n_visits)
        The distance or similarity matrix (depending on what the clustering algorithm needs)

    Returns
    -------
    IEM: float
        The Internal Evaluation Metric of applying the given algorithm to the given dataset.
    """
    # Verify input
    assert matrix.shape[0] == len(dsx.pure_values)

    # Do the clustering
    clustering_instance.fit(X=matrix)

    # Get the cluster labels
    labels = np.unique(clustering_instance.labels_)

    # Generate DSX objects for the clusters
    clusters = [dsx.get_subset(rp_values=set(dsx.pure_values[clustering_instance.labels_ == label])) for label in
                labels]

    # Return the IEM
    return iem_factory.create_function(dsx, clusters)(w)


def iem_wrapper(**kwargs):
    alg = kwargs['alg']
    if alg == OUR_FRAMEWORK:
        mbr = kwargs['mbr']
        assert isinstance(mbr, MultipleBootstrapResult)
        iem = mbr.the_iem
    elif alg in hac_algorithms:
        linkage = alg.split(' - ')[1]
        w = kwargs['w']
        dsx = kwargs['dsx']
        matrix = kwargs['matrix']
        a = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage=linkage)
        iem = sk_learn_wrapper(a, w, dsx, matrix)
    elif alg == spectral:
        w = kwargs['w']
        dsx = kwargs['dsx']
        matrix = kwargs['matrix']
        a = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=0)
        iem = sk_learn_wrapper(a, w, dsx, 1 - matrix)
    elif alg == voronoi:
        w = kwargs['w']
        dsx = kwargs['dsx']
        dsx_c = bootstraphelpers.DSXClustering(bootstraphelpers.DSXClustering.NUMBA_VORONOI)
        medoids, clusters = dsx_c.cluster(dsx, weights=w, initial_medoids=None, k=4)
        iem = iem_factory.create_function(dsx, clusters)(w)
    else:
        raise ValueError(f'Unknown alg: {alg}')

    return iem


def run():
    if not PKDD_PARAMETERS.RESULTS_3.exists():
        PKDD_PARAMETERS.RESULTS_3.mkdir(parents=True)

    # Compute the iem, medoids, and clusters for each sampling
    for sampling in 'H':

        # Run over every experiment
        experiment_names = file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling), False)

        # index = dataset, columns = algorithms, values = iem
        iem_x = pd.DataFrame()
        iem_x.index.name = DATASET_NAME

        ps = ProgressShower(pre=f'Quantitative Analysis {sampling}', total_steps=experiment_names)
        for fdx in experiment_names:

            # load dataset
            mbr = MultipleBootstrapResult(PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling) / fdx)
            ds_name = mbr.settings.dataset
            ps.update_post(f'{ds_name} : loading')
            dsx = data_loader.generic_dsx_loader(ds_name, mbr.settings.hierarchy_name)
            w = dsx.unit_weight_vector()
            pure_distance_matrix = dsx.get_distance_matrix_over_d(w)
            assert np.amax(pure_distance_matrix) <= 1.0

            # Add results from every algorithm
            for alg in all_algorithms:
                ps.update_post(f'{ds_name} : {alg}')
                # Computation
                iem = iem_wrapper(alg=alg, dsx=dsx, w=w, matrix=pure_distance_matrix, mbr=mbr, as_iids=True)
                # Add to (intermediate) results
                iem_x.loc[ds_name, alg] = iem

            # Finish this dataset
            ps.update()

        # Wrap IEM
        dataframe_operations.export_df(iem_x, PKDD_PARAMETERS.RESULTS_3 / f'iem_{sampling}.csv', index=True)


if __name__ == '__main__':
    run()
