import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import data.data_loader
from cjo.base.stringconstants import IEM, supercat, medoid, CLUSTERING_TIME, OPTIMIZATION_TIME, \
    REPETITION, cluster_size, ITERATION, ISC_WEIGHT, SUPPORT, fp_id, fpw_str
from functions import tex_functions, dataframe_operations
from functions.general_functions import listified


def id_generator(values):
    values = values.unique()
    foo = int(math.ceil(math.log(len(values), 10)))
    ids = [f'{x:0{foo}d}' for x in range(len(values))]
    return {v: i for v, i in zip(values, ids)}


class MultipleBootstrapSettings:

    def __init__(self, fd_mbr):
        fn_param = Path(fd_mbr) / 'settings.txt'
        sr = dataframe_operations.import_sr(fn_param, header=None)
        self.__settings = dict()
        for s in ['k', 'n_max', 'dsx_size_d', 'dsx_size_r']:
            self.__settings[s] = int(sr[s])
        for s in ['randomization', 'epsilon']:
            self.__settings[s] = float(sr[s])
        for s in {'dataset', 'hierarchy', 'optimization_strategy', 'iem_mode'}:
            self.__settings[s] = sr[s]

    @property
    def dataset(self):
        return self.__settings['dataset']

    @property
    def hierarchy_name(self):
        return self.__settings['hierarchy']

    @property
    def optimization_strategy(self):
        return self.__settings['optimization_strategy']

    @property
    def iem_mode(self):
        return self.__settings['iem_mode']

    @property
    def randomization(self):
        return self.__settings['randomization']

    @property
    def epsilon(self):
        return self.__settings['epsilon']

    @property
    def k(self):
        return self.__settings['k']

    @property
    def n_max(self):
        return self.__settings['n_max']

    @property
    def dsx_size_d(self):
        return self.__settings['dsx_size_d']

    @property
    def dsx_size_r(self):
        return self.__settings['dsx_size_r']


class MultipleBootstrapResult:

    def __init__(self, fd, fd_out=None, figure_extension='svg'):
        """
        Object to parse the results of multiple bootstraps

        Parameters
        ----------
        fd: str or Path
            Location of the results of the multiple bootstraps
        fd_out: str, Path or None
            Location of the output. If None, fd/results is used.
        figure_extension: str
            Extension of saved figures
        """

        # Input / Output
        fd = Path(fd)
        assert fd.exists()

        if fd_out is None:
            self.__fd_out = fd / 'results'
        else:
            self.__fd_out = Path(fd_out)
        assert isinstance(figure_extension, str)

        self.ext = figure_extension

        # Read the general settings (common to all repetitions)
        self.settings = MultipleBootstrapSettings(fd)

        self.hierarchy = data.data_loader.generic_hierarchy(self.settings.hierarchy_name)
        self.weight_column_names = [ISC_WEIGHT] + self.hierarchy.sc
        self.medoid_column_names = [f'{medoid}_{i}' for i in range(self.settings.k)]

        self.sbr = dict()
        df = dataframe_operations.import_df(fd / 'results.csv')
        for rep in df[REPETITION].unique():
            df_rep = df[df[REPETITION] == rep].drop(columns=REPETITION).set_index(ITERATION)
            self.sbr[rep] = BootstrapResult(df_rep, self.settings.k)

        # TODO : refactor this to self.results = self.compute_results()
        self.results = pd.DataFrame(index=list(self.sbr.keys()),
                                    columns=[f'{supercat}_{i}' for i in range(self.hierarchy.h + 1)] +
                                            [f'w_{i}' for i in range(self.hierarchy.h + 1)] +
                                            [f'm_{i}' for i in range(self.settings.k)])
        self.fill_results()

        self.summary = self.results.groupby(fpw_str).mean()[[IEM] + self.weight_column_names] \
            .assign(**{IEM + '_std': self.results.groupby(fpw_str).std()[IEM],
                       SUPPORT: self.results[fpw_str].value_counts()}) \
            .sort_values(SUPPORT, ascending=False) \
            .assign(**{fp_id: range(1, len(self.results[fpw_str].unique()) + 1)})
        self.results[fp_id] = self.results[fpw_str].replace(dict(self.summary[fp_id].items()))

    ##############
    # PROPERTIES #
    ##############

    @property
    def fd_out(self):
        """
        Output directory for results. Created once requested.

        Returns
        -------
        fd_out: Path
            The folder where to put the output of this MBR. (This is usually used when a single MBR is analyzed, the
            PKDD2020 paper has one MBR per dataset to compute the mean and CI values).
        """
        self.__fd_out.mkdir(parents=True, exist_ok=True)
        return self.__fd_out

    @property
    def unique_fp_ids(self):
        """
        Names of all fixed points

        Returns
        -------
        fps: np.ndarray
            Array of fixed point names
        """
        return self.results[fp_id].unique()

    @property
    def clustering_time_per_iteration(self):
        return self.__time_per_iteration(CLUSTERING_TIME)

    @property
    def optimization_time_per_iteration(self):
        return self.__time_per_iteration(OPTIMIZATION_TIME)

    def __time_per_iteration(self, t):
        assert t in [CLUSTERING_TIME, OPTIMIZATION_TIME]
        sr = pd.Series(dtype=np.float)
        for sbr in self.sbr.values():
            sr = sr.append(sbr.durations[t].dropna())
        return sr.values

    def fill_results(self):
        # TODO this should be deprecated as per the combination results of multiple datasets

        # Getting the info
        for k, v in self.sbr.items():
            self.results.loc[k, fpw_str] = v.fp_weights_as_string
        self.results = self.results[[fpw_str]]

        keys = list(self.sbr.keys())
        self.results[IEM] = pd.Series(data={k: v.final_iem for k, v in self.sbr.items()})
        self.results['n*'] = pd.Series(data={k: v.n_star for k, v in self.sbr.items()})
        self.results[CLUSTERING_TIME] = pd.Series(data={k: v.total_cluster_time for k, v in self.sbr.items()})
        self.results[OPTIMIZATION_TIME] = pd.Series(data={k: v.total_optimization_time for k, v in self.sbr.items()})

        weights = pd.DataFrame(data=[self.sbr[k].final_weights for k in keys], index=keys)
        medoids = pd.DataFrame(data=[self.sbr[k].final_medoids for k in keys], index=keys)
        self.results = pd.concat([self.results, medoids, weights], axis=1, sort=False)

    def __len__(self):
        return len(self.results)

    def __str__(self):
        return f'MBR[{len(self)} / {self.settings.dataset} / {self.settings.hierarchy_name}]'

    def __repr__(self):
        return str(self)

    @property
    def super_categories(self):
        return sorted(self.weight_column_names)

    #########################
    # (FP-SPECIFIC) GETTERS #
    #########################

    def get_repetition_names(self, fp):
        """
        Returns the repetition names for a given Fixed Point.

        Parameters
        ----------
        fp: int or None
            The fixed point to be queried. If None, all repetition_names are returned.

        Returns
        -------
        index: pd.Index
            The repetition names, optionally only the ones of the given fixed point.

        """
        if fp is None:
            return self.results.index
        else:
            return self.results[self.results[fp_id] == fp].index

    def get_final_weights(self, fp=None):
        """
        Returns all final weights (of a specific fixed point).

        Parameters
        ----------
        fp: int or None
            Fixed Point for which to get the results. If None, all final weights are returned

        Returns
        -------
        final_weights: pd.DataFrame
            DataFrame with all final weights (of the given fixed point).

        """
        return self.results.loc[self.get_repetition_names(fp)][self.weight_column_names]

    def get_average_final_weight(self, fp=None):
        """
        Gets the average final weight (of a variant).

        Parameters
        ----------
        fp: int or None
            If not None, get the average for this fixed point. Otherwise get the global average.

        Returns
        -------
        w: pd.Series
        """
        return self.get_final_weights(fp).mean()

    def get_iem(self, fp=None):
        """
        Get the IEM values (for a given fixed point)

        Parameters
        ----------
        fp: int or None
            If not None, the IEM values of this fixed point are returned. Otherwise get all IEMs

        Returns
        -------

        """
        return self.results.loc[self.get_repetition_names(fp), IEM]

    def get_a_final_medoid(self, variant):
        return self.results.loc[self.get_repetition_names(variant)].iloc[0][
            [f'Medoid {i}' for i in range(self.settings.k)]]

    def get_sbr(self, fp=None):
        """
        Returns all Single Bootstrap Results (for the given fixed point).

        Parameters
        ----------
        fp : int
            Fixed Point for which to get the sbr. If None, all sbr are returned

        Returns
        -------
        sbr : dict of str to SingleBootstrapResult
            The repetition names with their SingleBootstrapResult (of the given fixed points)

        """
        return {k: self.sbr[k] for k in self.get_repetition_names(fp)}

    def get_results(self, fp=None):
        return self.results.loc[self.get_repetition_names(fp)]

    ##############
    # TEX TABLES #
    ##############

    def create_fixed_point_tex_table(self, top=5, ci=None, **kwargs):
        """
        Creates a tex file with a table that summarizes the variants.
b
        Parameters
        ----------
        top: int or None
            Number of variants to show. If None, all are shown
        ci: float or None
            If float, take this as confidence interval value. If None, skip confidence interval

        Other Parameters
        ----------------
        kwargs: dict
            Parameters that are passed to cjo.functions.TexFunctions.df_to_table

        Raises
        ------
        AssertionError
            If the confidence level is not in [0,1]
        """
        if top is None:
            top = len(self.unique_fp_ids)
        df = self.summary.head(top).copy()
        if ci is not None:
            assert 0 <= ci <= 1, 'ci must be in [0,1]'
            z = stats.norm.interval(ci, 0, 1)[1]
            ci_value = df[IEM + '_std'] / df[SUPPORT].apply(lambda n: n ** 0.5) * z
            df[IEM] = df[IEM].apply(lambda m: f'{m:.2f}') + '$\\pm$' + ci_value.apply(lambda c: f'{c:.2f}')

        df.drop(columns=IEM + '_std', inplace=True)

        df.rename(columns={
            IEM: r'$\Phi$',
        }, inplace=True)
        df = df.reset_index(drop=False).set_index([fp_id, SUPPORT, r'$\Phi$'])

        for i, r in df.iterrows():
            max_weight = r.idxmax()
            for sc, w in r.iteritems():
                df.loc[i, sc] = f'{df.loc[i, sc]:.2f}'
                if sc == max_weight:
                    df.loc[i, sc] = f'\\textbf{{{df.loc[i, sc]}}}'
        kwargs.setdefault('label', 'tab:res:variant')
        kwargs.setdefault('caption', 'Weights of most frequent variants, showing the number of repetitions and IEM. '
                                     'The highest weight in each variant is highlighted.')
        kwargs.setdefault('escape', False)
        tex_functions.df_to_table(
            df.T, fn_out=self.fd_out / 'variants_table.tex', **kwargs)

    ############
    # PRINTERS #
    ############

    def print_variants_summary(self, top=5):
        """

        Parameters
        ----------
        top: int or None
            Number of variants to show. If None, all are shown

        Prints the summary of all variants on screen.
        """
        if top is None:
            top = len(self.unique_fp_ids)
        print(self.summary.head(top).to_string())

    @property
    def the_fixed_point(self):
        """*The* fixed point"""
        # Sorting on IEM AND medoid column names will make the choice deterministic should there be multiple fixed
        # points with the same IEM and weights but with different medoids.
        # I do not this clash will realistically occur though.
        sr = self.get_results(fp=1).sort_values([IEM] + self.medoid_column_names, ascending=True).iloc[0]
        return sr[self.weight_column_names], sr[self.medoid_column_names]

    @property
    def the_iem(self):
        """the IEM of *The* fixed point"""
        return self.get_results(fp=1)[IEM].min()


class BootstrapResult:

    def __init__(self, df, k):
        """
        Loads the results of a *single* bootstrap repetition

        Parameters
        ----------
        df: pd.DataFrame
            Results DataFrame (generated by MultipleBootstrapResult)
        k: int
            Number of clusters (this is saved at MultipleBootstrapResult level)
        """

        medoid_cols = [f'{medoid}_{i}' for i in range(0, k)]
        self.medoids = pd.DataFrame()
        for c in medoid_cols:
            self.medoids[c] = df[c].astype('int64')
        size_cols = [f'{cluster_size}_{i}' for i in range(0, k)]
        self.cluster_sizes = pd.DataFrame()
        for c in size_cols:
            self.cluster_sizes[c] = df[c].astype(int)

        duration_cols = [OPTIMIZATION_TIME, CLUSTERING_TIME]
        self.durations = df[duration_cols]

        iem_cols = [IEM]
        self.iem_score = df[IEM]

        non_weight_cols = medoid_cols + size_cols + duration_cols + iem_cols
        weight_cols = [c for c in df.columns if c not in non_weight_cols]
        self.weights = df[weight_cols]

    def __str__(self):
        return f'SBR[{self.n_star} iterations]'

    def __repr__(self):
        return str(self)

    ##############
    # PROPERTIES #
    ##############

    @property
    def final_weights(self):
        """The final weights of the repetition"""
        return self.weights.iloc[-1]

    @property
    def final_iem(self):
        """The last computed value of :math:`\\Phi`"""
        return self.iem_score.iloc[-1]

    @property
    def final_medoids(self):
        """The last computed medoids"""
        return self.medoids.iloc[-1]

    @property
    def n_star(self):
        """The final iteration round"""
        return self.weights.index.max()

    @property
    def total_cluster_time(self):
        return self.durations[CLUSTERING_TIME].sum()

    @property
    def total_optimization_time(self):
        return self.durations[OPTIMIZATION_TIME].sum()

    @property
    def fixed_point_info(self):
        """All required information to describe the fixed point of this single bootstrap repetition"""
        final_medoids = listified(self.final_medoids, int, sort=True, validation=lambda x: x > 0)
        f = self.final_weights.sort_values(ascending=False)
        return pd.Series(data={**{f'{supercat}_{i}': v for i, v in enumerate(f.index)},
                               **{f'w_{i}': v for i, v in enumerate(f.values)},
                               **{f'm_{i}': v for i, v in enumerate(final_medoids)}})

    @property
    def fp_weights_as_string(self):
        return ";".join(f"{k},{v:.2f}" for k, v in self.final_weights.sort_values(ascending=False).iteritems())
