import os
import sys
import shutil
import yaml
sys.path.append(os.getcwd())
sys.path.append('../../')
from solver import Solver
from datasets.mnist import load_specific_data
from mnist.model import DeepSVDD


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)
    os.makedirs(config['logdir'], exist_ok=True)
    shutil.copy(yml_path, os.path.join(config['logdir'], 'config.yml'))

    x = load_specific_data(phase='train',
                           **config['train_data_params'])
    model = DeepSVDD(**config['model_params'])

    solver = Solver(model,
                    **config['solver_params'],
                    logdir=config['logdir'])

    solver.fit(x,
               **config['fit_params'])


if __name__ == '__main__':
    main()
