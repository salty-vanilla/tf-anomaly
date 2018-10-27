import os
import sys
import shutil
import yaml
sys.path.append(os.getcwd())
sys.path.append('../../')
from solver import Solver
from datasets.mnist import load_specific_data
from mnist.autoencoder import AutoEncoder
from estimation_network import EstimationNetwork
from gmm import GMM
from dagmm import DAGMM


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)
    os.makedirs(config['logdir'], exist_ok=True)
    shutil.copy(yml_path, os.path.join(config['logdir'], 'config.yml'))

    x = load_specific_data(phase='train',
                           **config['train_data_params'])
    autoencoder = AutoEncoder(**config['autoencoder_params'])
    estimation_network = EstimationNetwork(**config['estimator_params'])
    gmm = GMM(config['estimator_params']['dense_units'][-1],
              config['autoencoder_params']['latent_dim']+1)

    dagmm = DAGMM(autoencoder,
                  estimation_network,
                  gmm)

    solver = Solver(dagmm,
                    **config['solver_params'],
                    logdir=config['logdir'])

    solver.fit(x,
               **config['fit_params'])


if __name__ == '__main__':
    main()
