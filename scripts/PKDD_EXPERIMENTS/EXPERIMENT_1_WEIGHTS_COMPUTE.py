from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import DSXClustering
from cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap import run_bootstrap
from data import data_loader

# Runs the bootstraps, used in all experiments in the paper
from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS
from scripts.PKDD_EXPERIMENTS.PKDD_PARAMETERS import REPETITION_COUNT


def run():
    for sampling in ['H', 'V']:

        for i in range(100):
            dsx = {'V': data_loader.dsx_apkdd, 'H': data_loader.dsx_pkdd}[sampling](i, 'A3')
            fd = PKDD_PARAMETERS.RESULTS_BOOTSTRAP(sampling)
            run_bootstrap(dsx=dsx, dataset=f'{"A" if sampling == "V" else ""}PKDD{i}', hierarchy='A3',
                          weight_vectors=REPETITION_COUNT, k=4, experiment_name=f'PKDD_{sampling}3_{i}',
                          save_start_time=False, make_noise=True,
                          fd=fd, cluster_mode=DSXClustering.NUMBA_VORONOI)
            print(i)


if __name__ == '__main__':
    run()
