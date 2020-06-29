import pandas as pd

from cjo.base.stringconstants import DATASET_NAME
from functions import file_functions
from functions import confidence_interval, dataframe_operations
from cjo.weighted_adapted_jaccard.result_computations import cluster_analysis
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult

from data import data_loader
from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS


# This is the results script of the second experiment: Cluster Analysis. It generates a tex file with the results.
# The names of the clusters are the abstract representations of the original medoids.

def run():
    # TODO FIX hard coded
    h = data_loader.generic_hierarchy('A3')

    for sample in 'H':

        # Read the cluster inclusion values.
        # Each file contains the inclusion values from one cluster set over all datasets
        cluster_dfs = dict()
        csv_files = [fn for fn in file_functions.list_files(PKDD_PARAMETERS.RESULTS_2(sample), False) if
                     fn[-4:] == '.csv']

        for fn in csv_files:
            if '_' in fn:
                m = fn.split('_')[1][:-4]
                cluster_dfs[m] = dataframe_operations.import_df(PKDD_PARAMETERS.RESULTS_2(sample) / fn)
                if DATASET_NAME in cluster_dfs[m].columns:
                    cluster_dfs[m].drop(columns=DATASET_NAME)

        # Combine the results for each cluster
        df_mean = pd.DataFrame()
        df_ci = pd.DataFrame()
        for k, v in cluster_dfs.items():
            df_mean[k] = v.mean()
            df_ci[k] = confidence_interval.std_n_to_ci(v.std(), len(v), 0.95)

        # Create a tex file
        cluster_analysis.cluster_statistics_2_tex(df_mean, h,
                                                  PKDD_PARAMETERS.RESULTS_2(sample) / 'cluster_analysis.tex',
                                                  df_ci=df_ci,
                                                  inclusion_missing=False,
                                                  label='tab:res:inclusion')
        cluster_analysis.cluster_statistics_2_tex(df_mean, h,
                                                  PKDD_PARAMETERS.RESULTS_2(sample) / 'cluster_analysis_full.tex',
                                                  df_ci=df_ci, num_c=99,
                                                  inclusion_missing=False,
                                                  label='tab:res:inclusion')


if __name__ == '__main__':
    run()
