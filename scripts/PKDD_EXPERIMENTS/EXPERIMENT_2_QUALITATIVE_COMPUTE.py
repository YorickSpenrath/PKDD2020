from collections import Counter

import numpy as np
import pandas as pd

from cjo.base import stringconstants
from cjo.base.stringconstants import cluster_size, DATASET_NAME
from cjo.weighted_adapted_jaccard.distances.implementation import DSX
from cjo.weighted_adapted_jaccard.result_computations.clustering_computation import \
    create_clusters_and_assign_them_to_given_medoids
from functions import file_functions
from functions import dataframe_operations
from cjo.weighted_adapted_jaccard.result_computations import cluster_analysis
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult, \
    MultipleBootstrapSettings
from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import DSXClustering

from functions.progress import ProgressShower, NoProgressShower

from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS
from data import data_loader


# This is the computation script of the second experiment: Cluster Analysis. It generates k+1 files, one for the cluster
# contents of each dataset for each medoid (k total), and 1 for the contents of each dataset. The names of the files
# are the abstract representations of the original medoids.

def run():
    for sampling in 'H':
        if not PKDD_PARAMETERS.RESULTS_2(sampling).exists():
            PKDD_PARAMETERS.RESULTS_2(sampling).mkdir(parents=True)

        if not (PKDD_PARAMETERS.RESULTS_2(sampling) / 'SuperClusters.csv').exists():
            medoids_df, meta_medoids, meta_clusters = compute_super_clusters(sampling)

            # Save superclusters
            df = create_clustering_map({k: v for k, v in zip(meta_medoids, meta_clusters)}) \
                .rename(columns={stringconstants.medoid: stringconstants.super_medoid})
            df.index.name = stringconstants.medoid
            dataframe_operations.export_df(df, PKDD_PARAMETERS.RESULTS_2(sampling) / 'SuperClusters.csv', index=True)

        compute_clusters(sampling)


def create_clustering_map(d):
    assert isinstance(d, dict)
    df = pd.DataFrame()
    for k, v in d.items():
        assert isinstance(v, DSX)
        df = df.append(pd.DataFrame(data={stringconstants.RP: pd.Series(v.rp_values, dtype=np.int64),
                                          stringconstants.medoid: str(k),
                                          stringconstants.multiplicity: pd.Series(v.drp_values, dtype=int)}))
    return df.set_index(stringconstants.RP)


def compute_super_clusters(sampling):
    df_w = pd.DataFrame()

    all_experiments = file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling), False)

    ps = ProgressShower(all_experiments, pre='Retrieving results')

    # TODO fix that h is saved at super-bootstrap-level
    h = data_loader.generic_hierarchy('A3')
    medoids_df = pd.DataFrame(columns=range(4))
    medoids_df.set_index = DATASET_NAME
    for fd in all_experiments:
        ps.update_post(fd)
        mbr = MultipleBootstrapResult(PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling) / fd)
        fp = mbr.the_fixed_point
        df_w = df_w.append(fp[0].to_frame(fd).T)
        medoids_df.loc[fd, :] = [int(m) for m in fp[1].values]
        ps.update()

    medoids = Counter(medoids_df.values.flatten())

    # Compute used weight
    w = df_w.mean(axis=0)

    # Compute dataset
    super_dsx = DSX(dataset=medoids, hierarchy=h, visit_ids_per_rp=None)

    # Compute super clusters
    meta_medoids, meta_clusters = DSXClustering(DSXClustering.NUMBA_VORONOI) \
        .cluster(super_dsx, weights=w, initial_medoids=None, k=4)

    return medoids_df, meta_medoids, meta_clusters


def compute_clusters(sampling):
    superclusters_data = dataframe_operations.import_sr(PKDD_PARAMETERS.RESULTS_2(sampling) / 'SuperClusters.csv',
                                                        dtype={stringconstants.medoid: np.int64,
                                                               stringconstants.super_medoid: np.int64})
    all_experiments = file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling), False)
    hierarchy = data_loader.generic_hierarchy(
        MultipleBootstrapSettings(PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling) / all_experiments[0]).hierarchy_name)
    super_medoids = [np.int64(i) for i in superclusters_data.unique()]

    cluster_dfs = {mp: pd.DataFrame(columns=[cluster_size] + hierarchy.sc + hierarchy.c) for mp in
                   (super_medoids + ['All'])}

    # For each cluster that we will find; save one DataFrame
    ps = ProgressShower(total_steps=all_experiments, pre=f'Analyzing clusters')

    for en in all_experiments:
        ps.update_post(en)
        fd = PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling) / en
        mbr = MultipleBootstrapResult(fd)
        dataset_name = mbr.settings.dataset

        # TODO you are basically recomputing the clusters (since we are only saving the medoids).
        # Maybe save the clusters somewhere as well, since they are used more often.

        res = create_clusters_and_assign_them_to_given_medoids(mbr)

        for k, dsx in res.items():
            df = cluster_analysis.single_cluster_2_statistics(dsx, hierarchy) \
                .to_frame() \
                .T \
                .assign(**{DATASET_NAME: dataset_name, cluster_size: dsx.size_d})
            super_medoid = superclusters_data.loc[k]
            cluster_dfs[super_medoid] = cluster_dfs[super_medoid].append(df)

        # Compute the statistics of the entire dataset
        dsx = data_loader.generic_dsx_loader(dataset_name, hierarchy)
        cluster_dfs['All'].loc[dataset_name, :] = cluster_analysis.single_cluster_2_statistics(dsx, hierarchy)
        cluster_dfs['All'].loc[dataset_name, cluster_size] = dsx.size_d
        cluster_dfs['All'].index.name = DATASET_NAME

        # Update progress
        ps.update()

    # Export the results, one file per medoid
    for k, v in cluster_dfs.items():
        # Save statistics (dataset x hierarchy)
        v = v.reset_index(drop=True)
        dataframe_operations.export_df(v, PKDD_PARAMETERS.RESULTS_2(sampling) / f'cluster_{k}.csv', index=True)


if __name__ == '__main__':
    run()
