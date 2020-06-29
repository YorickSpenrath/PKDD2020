from pathlib import Path

REPETITION_COUNT = 1000

try:
    from __local_settings import REPETITION_COUNT
except (ImportError, ModuleNotFoundError):
    pass

EXPERIMENTS_ROOT = Path(f'results/PKDD {REPETITION_COUNT} Reps')

# RESULTS FOR THE BOOTSTRAPPING ----------------------------------------------------------------------------------------
__RESULTS_BOOTSTRAP_H = EXPERIMENTS_ROOT / 'bootstrap_H'
__RESULTS_BOOTSTRAP_V = EXPERIMENTS_ROOT / 'bootstrap_V'


def RESULTS_BOOTSTRAP(sampling):
    if sampling == 'H':
        return __RESULTS_BOOTSTRAP_H
    elif sampling == 'V':
        return __RESULTS_BOOTSTRAP_V
    else:
        raise ValueError(f'No such sampling: {sampling}')


RESULTS_1 = EXPERIMENTS_ROOT / 'EXPERIMENT_1'

# RESULTS FOR THE PRE CHECK OF THE CLUSTER COMBINATION (V sampling is not used here) -----------------------------------
RESULTS_2PREH = EXPERIMENTS_ROOT / 'EXPERIMENT_2PREH'

# RESULTS FOR THE CLUSTER COMBINATION ----------------------------------------------------------------------------------
__RESULTS_2H = EXPERIMENTS_ROOT / 'EXPERIMENT_2H'
__RESULTS_2V = EXPERIMENTS_ROOT / 'EXPERIMENT_2V'


def RESULTS_2(sampling):
    if sampling == 'H':
        return __RESULTS_2H
    elif sampling == 'V':
        return __RESULTS_2V
    else:
        raise ValueError(f'No such sampling: {sampling}')


# RESULTS FOR THE COMPETITOR ANALYSIS (H/V sampling is combined here by design) ----------------------------------------
RESULTS_3 = EXPERIMENTS_ROOT / 'EXPERIMENT_3'
