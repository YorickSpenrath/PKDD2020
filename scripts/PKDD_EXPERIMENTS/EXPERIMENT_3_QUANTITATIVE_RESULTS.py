from sklearn.metrics import adjusted_rand_score

from cjo.base.stringconstants import IEM, ARI, IEM_tex, DATASET_NAME, invoice, ARI_tex, ALGORITHM
import pandas as pd
from functions import dataframe_operations, tex_functions
from functions import confidence_interval
from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS


def run():
    # Load iem

    for sample in 'H':
        df = dataframe_operations \
            .import_df(PKDD_PARAMETERS.RESULTS_3 / f'iem_{sample}.csv').set_index(DATASET_NAME)

        res = pd.DataFrame(columns=[IEM])
        for c in df.columns:
            res.loc[c, IEM] = confidence_interval.latex_string(df[c], 0.95, '{:.3f}', False)
        res = res.rename(columns={IEM: IEM_tex})
        tex_functions.df_to_table(res.reset_index(), escape=False, index=False,
                                  add_phantom=True, phantom_column_position=1, phantom_length=2,
                                  fn_out=PKDD_PARAMETERS.RESULTS_3 / f'competitor_analysis_{sample}.tex')


if __name__ == '__main__':
    run()
