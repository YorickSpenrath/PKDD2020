from pathlib import Path

import pandas as pd

from cjo.base.stringconstants import invoice
from functions import dataframe_operations


def pkdd_key():
    df = pd.DataFrame()
    for i in range(3):
        fn = Path(r'data\private\PKDD') / f'{i}.csv'
        df = df.append(dataframe_operations.import_df(fn)[[f'PKDD_{invoice}', invoice]])
    return df
