import pandas as pd
from pathlib import Path
from functions import dataframe_operations
from cjo.base.dtypes import dtypes
from cjo.base.stringconstants import sku, hl_encoding, sku_des, cat


class SKUMap:

    def __init__(self, source):
        if isinstance(source, Path) or isinstance(source, str):
            self.__df = dataframe_operations \
                .import_df(source, dtype=dtypes) \
                .set_index(sku)[[sku_des, cat, hl_encoding(0)]]
        elif isinstance(source, pd.DataFrame):
            source = source.reset_index(drop=False)[[sku, sku_des, cat, hl_encoding]].set_index(sku)
            self.__df = pd.DataFrame(data={c: source[c].astype(str) for c in source.columns}).set_index(sku)
        elif isinstance(source, SKUMap):
            self.__df = source.__df.copy()
        else:
            raise TypeError(r'Unknown type : {type(source)}')

        assert self.__df.index.is_unique, f'{sku} not unique'

    @property
    def cat_map(self):
        return self.__df[cat]

    @property
    def sku_des_map(self):
        return self.__df[sku_des]
