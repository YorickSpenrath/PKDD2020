import abc
from itertools import product

from pathlib import Path
import pandas as pd

from cjo.base.skumap import SKUMap
from cjo.base.stringconstants import consumer, timestamp, sku, item_count, sku_des, \
    value, invoice, cat, unknown
from functions import dataframe_operations
from functions.general_functions import listified
from cjo_settings.generalsettings import default_categories_uno, default_categories

dtypes = {consumer: str,
          timestamp: str,
          invoice: str,
          sku: str,
          item_count: float,
          value: float,
          sku_des: str,
          **{f'{c}_{item_count}': float for c in default_categories_uno},
          **{f'{c}_{value}': float for c in default_categories_uno},
          **{f'{c}': int for c in default_categories}
          }

rl_features = [consumer, timestamp, invoice, sku, item_count, value]
tl_features = [sku, item_count, value, invoice]

# These are all indexed on visit
uvl_features = [consumer, timestamp, invoice]
ntl_features = [sku_des, item_count, value]
ubl_features = [f'{c}_{item_count}' for c in default_categories_uno] + [f'{c}_value' for c in default_categories_uno]
al_features = list(default_categories)


class _LogBase:

    def __init__(self, source):
        """
        Initializes a log from a DataFrame, File, or Log

        Parameters
        ----------
        source: str, Path, pd.DataFrame, or same type
            source of the log. If str or Path, the source is read as DataFrame from this location. If same type, the df
            of the source is used
        """
        if isinstance(source, type(self)):
            self.df = source.df
            return
        elif isinstance(source, str) or isinstance(source, Path):
            assert Path(source).exists()
            source = dataframe_operations.import_df(source, dtype=dtypes)
        assert isinstance(source, pd.DataFrame), 'Source is not a file or DataFrame or same type'

        self.df = pd.DataFrame()
        for c in self.required_features():
            self.df.loc[:, c] = source[c].astype(dtypes[c])

    @abc.abstractmethod
    def required_features(self):
        """
        The list of features that need to be in the log. This is called on construction, and override in subclasses

        Returns
        -------
        required_features: List of str
            List of features that need to be in the log.
        """
        pass

    def assert_unique(self, f, idx=None):
        """
        Assert feature is unique for each index in the DataFrame.

        Parameters
        ----------
        f: str or Iterable(str)
            The feature(s) to be asserted
        idx: str or None
            The column to treat as index. If None, the index of the DataFrame is used.

        """
        f = listified(f, str, validation=lambda x: x in self.df.columns)
        if idx is None:
            len_vid = len(self.df)
            len_f = len(self.df.reset_index(drop=False)[[self.index.name] + f])
        else:
            # On column
            assert idx in self.df.columns, 'given idx not in df.columns'
            len_vid = len(self[idx].drop_duplicates())
            len_f = len(self[[idx] + f].drop_duplicates())

        if len_f != len_vid:
            if len(f) == 1:
                raise ValueError('Duplicate value of {}'.format(f[0]))
            else:
                raise ValueError('Duplicate combinations of {}'.format(*f))

    def assert_unique_index(self):
        """
        Assert the index of the DataFrame is unique.
        """
        assert self.index.is_unique, 'Index of source is not unique'

    @property
    def index(self):
        """
        Forward for DataFrame.index

        Returns
        -------
        index: pd.Index
            The index of the DataFrame
        """
        return self.df.index

    @property
    def columns(self):
        """
        Forward for DataFrame.columns

        Returns
        -------
        columns: pd.Index
            The columns of the DataFrame
        """
        return self.df.columns

    def export(self, fn):
        """
        Export this log to a given filename.

        Parameters
        ----------
        fn : str or Path
            The location where the log should be exported to.
        """
        dataframe_operations.export_df(self.df, fn)
        return self

    def __getitem__(self, k):
        """
        Forward method to getting items from the DataFrame. Equivalent to self.df[k]

        Parameters
        ----------
        k: Object
            The index to get from the DataFrame

        Returns
        -------
        v: Object
            The value of df[k]

        """
        return self.df[k]

    def __setitem__(self, k, v):
        """
        Forward method for setting items in the DataFrame. Equivalent to self.df[k] = v.

        Parameters
        ----------
        k: Object
            The index to set in the DataFrame
        v: Object
            The value to set
        """
        self.df[k] = v

    def __str__(self):
        """
        str representation of the Log.

        Returns
        -------
        str: str
            str representation of the Log.
        """
        return str(self.df)

    def to_str(self):
        """
        Forward method of to_str function of the DataFrame

        Returns
        -------
        str: str
            to_str value of the DataFrame of the Log.
        """
        return self.df.to_str()

    def subset_visit(self, invoices):
        """
        Get a subset of the log, for a given iterable of invoice ids

        Parameters
        ----------
        invoices: str or iterable of str
            Invoice ids to get.

        Returns
        -------
        Log: Log
            Log of same type with only the given invoices. Missing invoices are ignored.
        """
        invoices = listified(invoices, str)
        if invoice in self.columns:
            return type(self)(self.df[self.df[invoice].isin(invoices)])
        elif self.index.name == invoice:
            invoices = listified(invoices, str, filtering=lambda v: v in self.index)
            return type(self)(self.df.loc[invoices])
        else:
            raise TypeError(f'Cannot perform this operation on {type(self)}')

    def iterrows(self):
        """
        Forward method to df.iterrows()

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        it : generator
            A generator that iterates over the rows of the frame.

        """
        return self.df.iterrows()

    def __len__(self):
        """
        The length of this Log.

        Returns
        -------
        len: int
            The length of the log.
        """
        return len(self.df)


class RawLog(_LogBase):
    """
    Raw data log. This is the first step in any studies; and its source will be case-specific.
    """

    def __init__(self, source):
        super().__init__(source)
        self.assert_unique(f=[invoice, consumer, timestamp], idx=invoice)

    def required_features(self):
        return rl_features

    def to_tl(self):
        """
        Convert this Raw Log to a Transaction Log.

        Returns
        -------
        tl: TransactionLog
            The Transaction Log that results from this Raw log
        """
        return TransactionLog(self.df[tl_features])

    def to_uvl(self):
        """
        Convert this Raw Log to an Unlabelled Visit Log

        Returns
        -------
        uvl: UnlabelledVisitLog
            The Unlabelled Visit Log that results from this Raw Log
        """
        return UnlabelledVisitLog(self.df[uvl_features].drop_duplicates())


class _LogBaseIid(_LogBase):
    """
    Abstract base class for all logs with an invoice ID as index
    """

    @abc.abstractmethod
    def required_features(self):
        pass

    def __init__(self, source):
        super().__init__(source)

        if self.index.name != invoice:
            assert invoice in source.columns
            self[invoice] = source[invoice].astype(dtypes[invoice])
            self.df.set_index(keys=invoice, inplace=True)

    def export(self, fn):
        dataframe_operations.export_df(self.df, fn, index=True)
        return self


class TransactionLog(_LogBase):
    """
    Transaction data without date/invoice/consumer
    """

    def __init__(self, source):
        super().__init__(source)

    def required_features(self):
        return tl_features

    def to_ubl(self, sku_map):
        """
        Convert this Transaction Log to an Unlabelled Basket Log

        Parameters
        ----------
        sku_map: SKUMap, pd.DataFrame, str or pd.Series
            Source of the sku_map. If str, it is loaded from this location. If Series or DataFrame, it uses the
            column Category.

        Returns
        -------
        ubl: UnlabelledBasketLog
            The Unlabelled Basket Log that results from this Transaction Log and the given SKU Map.
        """
        # TODO : add discount stuff
        # TODO : what do missing SKU do here exactly?

        # Check sku_map
        if not isinstance(sku_map, pd.Series):
            sku_map = SKUMap(sku_map).cat_map

        ubl = pd. \
            merge(left=self.df, right=sku_map, left_on=sku, right_index=True, how='left'). \
            fillna({cat: unknown}). \
            drop(columns=sku). \
            groupby([invoice, cat]). \
            agg({item_count: sum, value: sum}). \
            unstack(level=cat, fill_value=0)

        # Rename columns to match ubl format
        ubl.columns = [f'{c}_{x}' for (x, c) in ubl.columns]

        # Add categories that were not in the TL at all
        for c, x in product(default_categories_uno, [value, item_count]):
            if f'{c}_{x}' not in ubl.columns:
                ubl.loc[:, f'{c}_{x}'] = 0

        # Convert to ubl
        return UnlabelledBasketLog(ubl)


class UnlabelledBasketLog(_LogBaseIid):
    """
    Aggregated Basket contents for each visit.
    """

    def __init__(self, source):
        super().__init__(source)

    def required_features(self):
        return ubl_features

    def to_activation(self):
        """
        Create an Activation Log from this Unlabelled Basket Log.

        Returns
        -------
        al: ActivationLog
            The Activation Log created from this Unlabelled Basket Log.
        """
        df = self.df[[f'{c}_{item_count}' for c in default_categories]]. \
            applymap(lambda x: 1 if x > 0 else 0). \
            rename(columns={f'{c}_{item_count}': c for c in default_categories})

        df = df[sorted(default_categories)]
        # Because of the dropping of unknown and not-a-product categories, some receipts may be empty
        df = df[df.sum(axis=1) > 0]

        return ActivationLog(df)


class ActivationLog(_LogBase):
    """
    Activation of the baskets.
    """

    def __init__(self, source):
        super().__init__(source)
        assert (self.df.sum(axis=1) > 0).all(), 'Activation log contains empty receipts'
        assert self.df.isin([0, 1]).all(None)

    def required_features(self):
        return al_features


class UnlabelledVisitLog(_LogBaseIid):
    """
    Visit descriptions (without Basket).
    """

    def __init__(self, source):
        super().__init__(source)
        self.assert_unique(f=[consumer, timestamp, invoice])

    def required_features(self):
        return uvl_features
