from __future__ import print_function
import os
import argparse
import yaml

from datetime import datetime
now = datetime.now()
timestamp = str(now.year)[-2:] + "_" + str(now.month).zfill(2) + "_" + str(now.day).zfill(2) + "_" + \
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 
import numpy as np 
from evaluators.hypervolume import get_hypervolume, get_pareto_fronts
from main.optimizer import chebyshev_scalarization_batch


class MultiOracle(Oracle):
    def __init__(self, name, cheby=None, target_smiles=None, num_max_call=None, **kwargs):
        name_split = name.split('+')
        self.name_list = [i.split(':')[0] for i in name_split]
        self.weight_list = [float(i.split(':')[1]) for i in name_split]
        self.weight_list = [i/sum(self.weight_list) for i in self.weight_list]
        self.oracle_list = [Oracle(i, target_smiles, num_max_call, **kwargs)\
                             for i in self.name_list]
        self.name = name
        self.temp_oracle_score = {}
        self.pareto_rewards = np.array([])
        self.pareto_smiles = np.array([])
        self.cheby = cheby

    def __call__(self, return_all=False, *args, **kwargs):
        temp_list = []
        temp_hyper = []
        temp_smi = []
        for oracle, weight in zip(self.oracle_list, self.weight_list):
            score = oracle(*args, **kwargs)
            name = oracle.name
            if name == "sa":
                score /= 10
            temp_list.append(score * weight)
            self.temp_oracle_score[name] = score
            temp_hyper.append(score)

        temp_smi.append(*args)
        temp_hyper = np.array(temp_hyper)
        temp_smi = np.array(temp_smi)
        temp_smi = np.expand_dims(temp_smi, 0)
        temp_pare = np.expand_dims(temp_hyper, 0)

        if self.pareto_rewards.ndim == 2:
            temp_pare = np.append(temp_pare, self.pareto_rewards, axis=0)
            temp_smi = np.append(temp_smi, self.pareto_smiles, axis=0)
        candidates, pareto_rewards = get_pareto_fronts(temp_smi, temp_pare)
        self.pareto_rewards = pareto_rewards
        self.pareto_smiles = candidates
        if self.cheby:
            avg_score = chebyshev_scalarization_batch(temp_hyper, weights=self.weight_list)
        else:
            avg_score = np.sum(temp_list)
        if return_all:
            return avg_score, temp_hyper
        else:
            return avg_score


def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='graph_ga')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=200)
    parser.add_argument('--n_runs', type=int, default=5)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])  #
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument('--load_pretrained', default=None)
    parser.add_argument('--do_save', default=None)
    parser.add_argument('--timestamp', default=timestamp)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cheby', default=None)
    parser.add_argument('--dynamic_name', type=str, default="")
    parser.add_argument('--div_filter', type=str, default="IdenticalTopologicalScaffold", choices=["IdenticalMurckoScaffold", "NoFilter", "ScaffoldSimilarity", "IdenticalTopologicalScaffold"])
    parser.add_argument('--update_order', type=str, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.device}"
    os.environ["WANDB_MODE"] = args.wandb

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == "reinvent_cl":
        from main.reinvent_cl.run_CL import REINVENTGame_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
        args.memory_out_dir = os.path.join(path_main, "memory")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main

    
    for oracle_name in args.oracles:

        print(f'Optimizing oracle function: {oracle_name}')

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
        if len(args.oracles[0].split('+')) == 1:
            oracle = Oracle(name=oracle_name)
        else:
            oracle = MultiOracle(name=oracle_name, cheby=args.cheby)
        optimizer = Optimizer(args=args)

        for seed in args.seed:
            print('seed', seed)
            optimizer.optimize(oracle=oracle, config=config_default, seed=seed, project="MOMAGame")


    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    print("timestamp: ", args.timestamp)
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

