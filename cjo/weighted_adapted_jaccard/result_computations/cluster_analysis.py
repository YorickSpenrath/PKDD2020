import pandas as pd

from cjo.base.hierarchy import Hierarchy
from cjo.base.stringconstants import cat, supercat, cluster_size, multiplicity
from functions import tex_functions
from cjo.weighted_adapted_jaccard.distances.bitops import int2bitvector
from cjo.weighted_adapted_jaccard.distances.implementation import DSX
from cjo_settings.generalsettings import category_abbreviation_dict


def single_cluster_2_statistics(dsx, hierarchy):
    """
    Generates descriptive statistics on the activation of all non-root nodes of the hierarchy in the given dsx

    Parameters
    ----------
    dsx: DSX
        The dataset to compute summary statistics for.
    hierarchy: hierarchy.Hierarchy
        The hierarchy to use.

    Returns
    -------
    summary: pd.Series
        Summary statistics on the cluster in terms of fraction of :math:`\\mathcal{D}` with activations in each
        (super-)category.
    """
    # input
    assert isinstance(dsx, DSX)
    assert isinstance(hierarchy, Hierarchy)
    df = dsx.original_df_r

    rp_list = list(dsx.keys())
    sizes = df[multiplicity]

    # DataFrame receipt -> activations categories
    df2 = pd.DataFrame(data=[int2bitvector(df.loc[rp, cat], len(hierarchy.c)) for rp in rp_list], columns=hierarchy.c,
                       index=rp_list)
    # DataFrame receipt -> activation super-categories
    df3 = pd.DataFrame(data=[int2bitvector(df.loc[rp, supercat], hierarchy.h) for rp in rp_list], columns=hierarchy.sc,
                       index=rp_list)
    # DataFrame receipt -> activation categories and super-categories
    df = pd.concat([df2, df3], axis=1)

    # Multiply transpose by sizes (you get the multiplicity-adjusted activation) and divide by the total number of
    return ((df.T * sizes).T.sum() / (sizes.sum())).sort_values(ascending=False)


def clustering_2_statistics(clusters, hierarchy):
    """
    Generates descriptive statistics on the activation of all non-root nodes of the hierarchy of all clusters.

    Parameters
    ----------
    clusters: iterable of DSX
        The clusters
    hierarchy: dict of str to set of str
        The hierarchy

    Returns
    -------
    df: pd.DataFrame
        clustering statistics DataFrame
    """
    df = pd.DataFrame()
    for m, c in enumerate(clusters):
        df[m + 1] = single_cluster_2_statistics(c, hierarchy)
    return df


def cluster_statistics_2_tex(cluster_statistics, hierarchy, fn_out, df_ci=None, inclusion_missing=True, num_c=3,
                             **kwargs):
    """
    Generates a pdf from clustering activation statistics

    Parameters
    ----------
    cluster_statistics: pd.DataFrame
        hierarchy x clusters DataFrame
    hierarchy: dict of (str) -> (set of str)
        The hierarchy
    fn_out: str or Path
        Output location
    df_ci: pd.DataFrame
        As cluster_statistics, but then a standard deviation value
    inclusion_missing: bool
        If True, all values will be proceeded by True or False, and each values will at least be 50% (i.e. it gives the
        inclusion or missing percentage; whichever is higher). If False, the inclusion percentages are given.
    """
    if df_ci is None:
        f = '.0f'
    else:
        f = '.1f'
    cat_list = hierarchy.c
    df = pd.DataFrame()
    abbreviation_dict = dict()
    for cluster_name, stat in cluster_statistics.iteritems():
        s = f'${stat[cluster_size]:{f}}'
        if df_ci is not None:
            s += f'\\pm{df_ci.loc[cluster_size, cluster_name]:{f}}'
        s += '$'
        df.loc[cluster_name, cluster_size] = s
        for sc in hierarchy.sc:

            if inclusion_missing:
                # (adapted) average
                if stat[sc] < 0.5:
                    s = f'\\texttt{{False}} : ${100 - stat[sc] * 100:{f}}'
                else:
                    s = f'\\texttt{{True}} : ${stat[sc] * 100:{f}}'
            else:
                s = f'${stat[sc] * 100:{f}}'

            # std
            if df_ci is not None:
                s += f'\\pm{100 * df_ci.loc[sc, cluster_name]:{f}}'

            # close
            s += '$\\%'
            df.loc[cluster_name, sc] = s

        df.loc[cluster_name, ''] = ''

        for i, (k, v) in enumerate(stat[cat_list].sort_values(ascending=False).head(num_c).items()):

            # Make improvised multirow cells
            if df_ci is not None:
                df.loc[cluster_name, f'{cat} {i + 1}'] = f'\\texttt{{{category_abbreviation_dict[k]}}}'
                df.loc[cluster_name, f'{cat}|{i + 1}'] = \
                    f'${v * 100:{f}}\\pm{100 * df_ci.loc[k, cluster_name]:{f}}$\\%'

            else:

                df.loc[cluster_name, f'{cat} {i + 1}'] = \
                    f'\\texttt{{{category_abbreviation_dict[k]}}} : ${v * 100:{f}}$\\%'

            # save abbreviation for caption
            abbreviation_dict[k] = category_abbreviation_dict[k]

    df.index.name = 'Cluster'
    df.rename(index=lambda z: f'$K_{{{z}}}$', inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.set_index(['Cluster', cluster_size], inplace=True)

    # Remove the columns that are part of the improvised MultiRow cells
    if df_ci is not None:
        df.rename(columns={f'{cat}|{i + 1}': '' for i in range(num_c)}, inplace=True)

    df = df.T
    abbreviations = sorted(abbreviation_dict.keys(), key=lambda z: abbreviation_dict[z])

    def fix_abb(abb):
        abb = abb.capitalize()
        if abb.endswith(' np'):
            abb = abb[:-2] + 'NP'
        elif abb.endswith(' p'):
            abb = abb[:-1] + 'P'
        return abb

    abbreviations = [fix_abb(c) for c in abbreviations]
    categories = sorted(abbreviation_dict.values())

    caption = 'Descriptive Statistics of each cluster.' \
              ' The abbreviations are ' + \
              ', '.join([f'\\texttt{{{v}}}: {k}' for k, v in zip(abbreviations, categories)]) + '.'

    return tex_functions.df_to_table(df, caption=caption, fn_out=fn_out, escape=False, add_phantom=True,
                                     **kwargs), caption

