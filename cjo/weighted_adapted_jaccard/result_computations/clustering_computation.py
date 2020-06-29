from collections import Counter
from pathlib import Path

from cjo.base.logs import ActivationLog
from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import DSXClustering
from cjo.weighted_adapted_jaccard.distances.implementation import DSX, al2dsx
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult
from data import data_loader
from functions.general_functions import listified

clustering_algorithm = DSXClustering(DSXClustering.NUMBA_VORONOI).cluster


def create_clusters_and_assign_them_to_given_medoids(mbr, dataset_name=None):
    """
    Computes clusters given the fixed point of the mbr.

    Parameters
    ----------
    mbr: MultipleBootstrapResult, or str
        Information of the Result. If str, it will load the MultipleBootstrapResult from this location
    dataset_name: str or ActivationLog or None
        Name of the dataset to cluster. If None, the dataset from the mbr is used. This also implies that add_fp_medoids
         should be False

    Returns
    -------
    mapping: dict {str: DSX}
        The mapping from medoids to DSX. See Notes

    Notes
    -----
    The medoids are taken from the fixed point of the mbr. If mbr.settings.dataset != dataset_name or dataset_name is
    not None, the medoids of the fixed point of the mbr are added to the dataset prior to clustering, and are removed
    after assigning the clusters to the initial clusters.

    The return value maps each medoid to the DSX object that it is in after the clustering. The medoids are removed from
    these objects if added because of add_fp_medoids being True.

    Note that this is not necessarily a bijective mapping between the medoids and the clusters. If the dataset and
    medoids are not 'sufficiently similarly representing' the receipt space, multiple medoids may end up in a single
    cluster, and as a result some cluster may not get any medoid.

    """

    if isinstance(mbr, str) or isinstance(mbr, Path):
        mbr = MultipleBootstrapResult(mbr)
    elif isinstance(mbr, MultipleBootstrapResult):
        pass
    else:
        raise TypeError(f'Type of mbr not accepted:{type(mbr)}')

    if dataset_name is None:
        dataset_name = mbr.settings.dataset
    al = data_loader.generic_al_loader(dataset_name)

    add_fp_medoids = dataset_name != mbr.settings.dataset

    fp = mbr.the_fixed_point
    initial_medoids = Counter([int(k) for k in fp[1].values])

    if add_fp_medoids:
        dsx = al2dsx(al, hierarchy=mbr.hierarchy, additional_rp=initial_medoids)
    else:
        dsx = al2dsx(al, hierarchy=mbr.hierarchy)

    w = dsx.validate_and_convert_weights(fp[0])

    # Cluster the data
    _, clustering = clustering_algorithm(dsx, w, initial_medoids=fp[1])

    clustering = listified(clustering, DSX)

    # Map the medoids to the clusters
    res = {im: clustering[[im in dsx for dsx in clustering].index(True)] for im in initial_medoids}

    if add_fp_medoids:
        # Remove receipts from the clusters
        # Note that, since DSX is mutable, the removing is reflected in res.
        for dsx in clustering:
            receipts_to_be_removed = Counter(initial_medoids) - (Counter(initial_medoids) - dsx.dataset)
            dsx.remove(receipts_to_be_removed=receipts_to_be_removed)

    return res
