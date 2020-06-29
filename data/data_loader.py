import pandas as pd
from pathlib import Path
from cjo.base import logs
from cjo.base.hierarchy import Hierarchy
from functions import dataframe_operations
from cjo.weighted_adapted_jaccard.distances.implementation import al2dsx

# from pyyed import Graph

public_data_folder = Path('data/public')
retailer_folder = public_data_folder / 'retailer'


###########
# GENERIC #
###########

def generic_hierarchy(name):
    return Hierarchy(public_data_folder / 'generic_hierarchies' / f'{name}.csv')


def generic_dsx_loader(name, hierarchy):
    if name[0] == 'S':
        # Sample
        assert len(name) == 2, f'Invalid name : {name}'
        return dsx_sample(int(name[1]), hierarchy=hierarchy)
    elif name[0] == 'D':
        return dsx_data(int(name[1]), hierarchy=hierarchy, num=int(name.split('_')[1]))
    elif name.startswith('PKDD'):
        return dsx_pkdd(int(name[4:]), hierarchy=hierarchy)
    elif name.startswith('APKDD'):
        return dsx_apkdd(int(name[5:]), hierarchy=hierarchy)
    else:
        raise ValueError(f'Unknown name {name}')


def generic_al_loader(name):
    if name[0] == 'S':
        assert len(name) == 2, f'Invalid name : {name}'
    elif name[0] == 'D':
        exp = int(name[1])
        num = int(name.split('_')[1])
        return al_data(dataset_exponent=exp, dataset_number=num)
    elif name.startswith('PKDD'):
        return al_pkdd(dataset_number=int(name[4:]))
    elif name.startswith('APKDD'):
        return al_apkdd(dataset_number=int(name[5:]))
    else:
        raise ValueError(f'Unknown name {name}')


###############
# ACTUAL DATA #
###############

def al_data(dataset_exponent, dataset_number):
    assert isinstance(dataset_exponent, int)
    assert isinstance(dataset_number, int)

    if dataset_exponent in [3, 4, 5]:
        # These are saved with multiple values in one log
        assert dataset_number < 10 ** (7 - dataset_exponent)
        size = 10 ** dataset_exponent
        number_of_datasets_per_file = 500000 // size
        file_number = dataset_number // number_of_datasets_per_file
        df = dataframe_operations.import_df(retailer_folder / f'D{dataset_exponent}' / f'{file_number}.csv')
        dataset_number_in_file = dataset_number % number_of_datasets_per_file
        df = df.iloc[dataset_number_in_file * size: (dataset_number_in_file + 1) * size]
        return logs.ActivationLog(df)
    elif dataset_exponent in [6, 7]:
        assert (dataset_exponent == 6 and dataset_number < 10) or (dataset_exponent == 7 and dataset_number == 0)
        return logs.ActivationLog(retailer_folder / f'D{dataset_exponent}' / f'{dataset_number}.csv')
    else:
        raise ValueError('Illegal dataset_exponent')


def al_pkdd(dataset_number):
    """
    Load Activation Log number `dataset_number`

    Parameters
    ----------
    dataset_number: int
        index of the Activation Log, lower than 500

    Returns
    -------
    al: logs.ActivationLog
        The Activation Log with given index

    Notes
    -----
    The Activation Logs are saved with 500 per file (to limit the number of files). This function handles the retrieval.

    """
    assert dataset_number < 100

    df = dataframe_operations \
             .import_df(f'data/public/PKDD/ActivationLogs2.csv') \
             .iloc[dataset_number * 1000: (dataset_number + 1) * 1000]

    return logs.ActivationLog(df)


def al_apkdd(dataset_number):
    assert dataset_number < 100
    sr = pd.Series(data=range(100000)).apply(lambda x: (x % 1000) // 10 == dataset_number)
    df = dataframe_operations.import_df(f'data/public/PKDD/ActivationLogs2.csv')[sr]
    return logs.ActivationLog(df)


def dsx_data(exp, num, hierarchy):
    al = al_data(exp, num)
    if isinstance(hierarchy, str):
        hierarchy = generic_hierarchy(hierarchy)
    return al2dsx(al, hierarchy)


def dsx_pkdd(num, hierarchy):
    al = al_pkdd(num)
    if isinstance(hierarchy, str):
        hierarchy = generic_hierarchy(hierarchy)
    return al2dsx(al, hierarchy)


def dsx_apkdd(num, hierarchy):
    al = al_apkdd(num)
    if isinstance(hierarchy, str):
        hierarchy = generic_hierarchy(hierarchy)
    return al2dsx(al, hierarchy)


###########
# SAMPLES #
###########


def dsx_sample(exp, hierarchy='A3'):
    al = al_sample(exp)
    if isinstance(hierarchy, str):
        hierarchy = generic_hierarchy(hierarchy)
    return al2dsx(al, hierarchy)


def al_sample(exp):
    assert exp in range(1, 5), 'given exp must be in 1...4'
    return logs.ActivationLog(public_data_folder / 'samples' / 'real dataset' / f'S{exp}.csv')


def sample_hierarchy():
    return Hierarchy(public_data_folder / 'generic_hierarchies' / 'A3.csv')


############
# EXAMPLES #
############


def example_hierarchy():
    return Hierarchy(public_data_folder / 'samples' / 'implementation guide' / 'hierarchy.csv')


def example_cat_list():
    return example_hierarchy().c


def __dsx_example(loader):
    return al2dsx(loader(), example_hierarchy())


def dsx_running_example():
    return __dsx_example(al_running_example)


def dsx_extended_example():
    return __dsx_example(al_extended_example)


def dsx_extended_example2():
    return __dsx_example(al_extended_example2)


def al_running_example():
    return logs.ActivationLog(public_data_folder / 'samples' / 'implementation guide' / 'running_example.csv',
                              categories=example_cat_list())


def al_extended_example():
    return logs.ActivationLog(public_data_folder / 'samples' / 'implementation guide' / 'extended_example.csv',
                              categories=example_cat_list())


def al_extended_example2():
    return logs.ActivationLog(public_data_folder / 'samples' / 'implementation guide' / 'extended_example2.csv',
                              categories=example_cat_list())
