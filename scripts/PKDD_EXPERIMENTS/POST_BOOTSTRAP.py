from functions import file_functions
from scripts.PKDD_EXPERIMENTS import \
    PKDD_PARAMETERS, EXPERIMENT_1_WEIGHTS_RESULTS, \
    EXPERIMENT_2_QUALITATIVE_COMPUTE, EXPERIMENT_2_QUALITATIVE_RESULTS, \
    EXPERIMENT_3_QUANTITATIVE_COMPUTE, EXPERIMENT_3_QUANTITATIVE_RESULTS

if PKDD_PARAMETERS.REPETITION_COUNT in [100, 1000]:
    for sampling in 'H':
        file_functions.copydir(f'results/precomputed/PKDD {PKDD_PARAMETERS.REPETITION_COUNT} Reps/bootstrap_{sampling}',
                               PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling))
else:
    raise Exception(f'The results for {PKDD_PARAMETERS.REPETITION_COUNT} repetitions are not precomputed. Change the '
                    f'REPETITION_COUNT to 100 or '
                    '1000 in scripts/PKDD_EXPERIMENTS/PKDD_PARAMETERS.py to use precomputed results of Algorithm 1, or'
                    'use scripts/PKDD_EXPERIMENTS/ALL_EXPERIMENTS.py to compute these results.')

EXPERIMENT_1_WEIGHTS_RESULTS.run()

EXPERIMENT_2_QUALITATIVE_COMPUTE.run()
EXPERIMENT_2_QUALITATIVE_RESULTS.run()

EXPERIMENT_3_QUANTITATIVE_COMPUTE.run()
EXPERIMENT_3_QUANTITATIVE_RESULTS.run()
