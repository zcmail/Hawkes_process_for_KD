import numpy as np
import argparse
import json
import sys
import os
import torch

# Internal libraries
import excitation_kernels
import hawkes_model_single

from make_data_for_samples import make_data  # 多个样本数据

def make_object(module, name, args):
    return getattr(module, name)(**args)


param_dict = {'exitation': {'name': 'ExponentialKernel', 'args': {'decay': 1.0, 'cut_off': 1500.0}}}

def decision_fun(mu, W, new_events, M, param_dict):
    events_num = len(new_events)
    dim = len(new_events[0])
    n_params = dim * (M * + 1)
    # Init Hawkes process model object
    excitation_obj = make_object(excitation_kernels, **param_dict)
    hawkes_model_obj = hawkes_model_single.HawkesModel(excitation=excitation_obj, verbose=False)

    loglik_all = [0.0] * events_num

    for i in range(events_num):
        hawkes_model_obj.set_data(new_events[i])
        loglik = hawkes_model_obj.log_likelihood(mu, W)
        loglik_all[i] = loglik

    return loglik_all


def estimate(exp_dir, param_filename, stdout=None, stderr=None):
    if stdout is not None:
        sys.stdout = open(stdout, 'w')
    if stderr is not None:
        sys.stderr = open(stderr, 'w')

    data_fileName = "./data/DSL-StrongPasswordData.xls"
    events = make_data('s002', 40, 400, data_fileName)
    n_jumps_per_dim = list(map(len, events[0]))
    print('Number of jumps:', len(events) * sum(n_jumps_per_dim))
    print('per node:', n_jumps_per_dim)

    print('\nestimating')
    print('=========')

    param_filename = os.path.join(exp_dir, param_filename)
    if not os.path.exists(param_filename):
        raise FileNotFoundError(
            'Input file `{:s}` not found.'.format(param_filename))
    with open(param_filename, 'r') as param_file:
        param_dict = json.load(param_file)

    result = decision_fun(mu=param_dict['vi_exp']['mu'], W=param_dict['vi_exp']['adjacency'], new_events=events, M=1,
                          param_dict=param_dict_exitation)
    print(result)
    # Log that the run is finished
    print('\n\nFinished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', type=str,
                        # required=True, help="Working directory")
                        required=False, default=".")
    parser.add_argument('-p', '--params', dest='param_filename', type=str,
                        required=False, default='params.json',
                        help="Input parameter file (JSON)")
    args = parser.parse_known_args()[0]
    estimate('.', 'output_varhawkes_train_samples_s002_0-40-.json')