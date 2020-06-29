import itertools

import pandas as pd

from cjo.base.stringconstants import IEM, ISC_WEIGHT, ISC_WEIGHT_LATEX, SUPPORT, DATASET_NAME, fpw_str
from functions import file_functions
from functions import tex_functions, confidence_interval
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult
from functions.progress import ProgressShower

from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS


# This scripts computes the results for the Weights Learning experiment. It assumes the EXPERIMENT_1_COMPUTE is run

def get_results_for_one_dataset(dataset_results):
    """
    Get the results of a single dataset (of PKDD_PARAMETERS.REPETITION_COUNT repetitions)

    Parameters
    ----------
    dataset_results : str
        The folder name of the results on the datasets

    Returns
    -------
    sc : List of str
        The super-categories used (for verification)
    results : pd.DataFrame
        The results of this single dataset, containing:

            - The weights; converted to a two-digit value, separated by a ';' in a single column, in the order given by
            sc
            - The number of iterations of each repetition
            - The source; which is the value of i
            - The IEM of each repetition


    """

    # Get the results
    fd = PKDD_PARAMETERS.RESULTS_BOOTSTRAP('H') / dataset_results
    mbr = MultipleBootstrapResult(fd)

    # Compute the new DataFrames
    df = pd.DataFrame()
    df[SUPPORT] = mbr.summary[SUPPORT]
    df[DATASET_NAME] = dataset_results
    df[IEM] = mbr.summary[IEM]
    df = df.reset_index(drop=False)

    # Return results
    return df


def run():
    # Make results directory
    if not PKDD_PARAMETERS.RESULTS_1.exists():
        PKDD_PARAMETERS.RESULTS_1.mkdir(parents=True)

    # Get the names of the bootstrap results (each folder == one repetition == iterations until stability)
    repetition_folders_names = file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP('H'), False)

    # Progress updates
    ps = ProgressShower(total_steps=len(repetition_folders_names),
                        pre='Analyzing results for fixed points ')

    # Get all results
    super_categories = None
    df = pd.DataFrame()
    for folder_name in repetition_folders_names:
        ps.update_post(folder_name)
        # Get the results
        df = df.append(get_results_for_one_dataset(folder_name))
        ps.update()
    # Fill missing values (some datasets might not have some weight vectors)
    all_sources = df[DATASET_NAME].unique()
    all_fpw_strings = df[fpw_str].unique()
    mi_all = pd.MultiIndex.from_tuples(([(a, b) for a, b in itertools.product(all_sources, all_fpw_strings)]),
                                       names=[DATASET_NAME, fpw_str])
    df.set_index(keys=[DATASET_NAME, fpw_str], inplace=True)
    df = df.reindex(mi_all)
    df[SUPPORT] = df[SUPPORT].fillna(0) / PKDD_PARAMETERS.REPETITION_COUNT * 100

    # Final results (these get converted to the final table)
    res = pd.DataFrame()
    n_datasets = len(repetition_folders_names)

    # Compute the mean and 95%CI of the support
    res['mean'] = df.groupby(fpw_str).mean()[SUPPORT]
    res['std'] = df.groupby(fpw_str).std()[SUPPORT]
    res['CI95'] = confidence_interval.std_n_to_ci(res['std'], n_datasets, 0.95)
    res[SUPPORT] = res.apply(lambda r: f'${r["mean"]:.1f} \\pm {r["CI95"]:.1f}\\%$', axis=1)

    # Retrieve the weights
    res.index.name = fpw_str

    def create_df_row(fpws):
        return pd.DataFrame(data={s.split(',')[0]: [float(s.split(',')[1])] for s in fpws.split(';')}, index=[fpws])

    df_weights = pd.concat([create_df_row(x) for x in res.index], axis=0)
    df_weights = df_weights[sorted(df_weights.columns)]
    res = pd.merge(left=res, right=df_weights, left_index=True, right_index=True)

    # Sort on support
    res.sort_values('mean', inplace=True, ascending=False)
    res.drop(columns=['mean', 'std', 'CI95'], inplace=True)
    res.rename(columns={ISC_WEIGHT: ISC_WEIGHT_LATEX}, inplace=True)

    # Generate a tex-file
    res.rename(columns={SUPPORT: SUPPORT.capitalize()})
    res.set_index(SUPPORT, inplace=True)
    tex_functions.df_to_table(res.head(5).T,
                              caption='The most frequently found weights, showing the number of repetitions that found '
                                      'this combination of weights, averaged over the 100 datasets, with their '
                                      '95\\%-confidence intervals.',
                              label='tab:res:weights',
                              add_phantom=True,
                              column_format='l' * (2 + 5),
                              fn_out=PKDD_PARAMETERS.RESULTS_1 / 'weights.tex', escape=False, floating='h!')


if __name__ == '__main__':
    run()
