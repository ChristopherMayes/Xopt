import logging
import os
import random
import sys

import pandas as pd
import torch
from botorch.models.transforms.input import Normalize
from botorch.utils.sampling import draw_sobol_samples

from .models.models import create_model
from .utils import standardize, collect_results, sampler_evaluate, get_corrected_outputs, NoValidResultsError
from ..tools import full_path, DummyExecutor

"""
    Bayesian Exploration Botorch

"""

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Logger
logger = logging.getLogger(__name__)


def bayesian_optimize(vocs, evaluate_f,
                      gen_candidate,
                      n_steps=30,
                      executor=None,
                      n_initial_samples=5,
                      custom_model=None,
                      seed=None,
                      output_path=None,
                      verbose=True,
                      restart_data_file=None,
                      initial_x=None,
                      use_gpu=False,
                      eval_args=None,
                      **kwargs):
    """

    Parameters
    ----------
    vocs : dict
        Varabiles, objectives, constraints and statics dictionary, see xopt documentation for detials

    evaluate_f : callable
        Returns dict of outputs after problem has been evaluated

    gen_candidate : callable
        Callable that takes the form f(model, bounds, vocs, **kwargs) and returns 1 or more candidates

    n_steps : int, default: 30
        Number of optimization steps to take

    executor : futures.Executor, default: None
        Executor object to evaluate problem using multiple threads or processors

    n_initial_samples : int, defualt: 5
        Number of initial sobel_random samples to take to start optimization. Ignored if initial_x is not None.

    custom_model : callable, optional
        Function in the form f(train_x, train_y) that returns a botorch model instance

    seed : int, optional
        Seed for random number generation to freeze random number generation.

    output_path : str, optional
        Location to save optimization data and models.

    verbose : bool, default: False
        Specify if the algorithm should print optimization progress.

    restart_data_file : str, optional
        Pickled pandas data frame object containing initial data for restarting optimization.

    initial_x : torch.Tensor, optional
        Initial input points to sample evaluator at, causes sampling to ignore n_initial_samples

    use_gpu : bool, False
        Specify if the algorithm should use GPU resources if available. Only use on large problems!

    eval_args : list, []
        List of positional arguments for evaluation function

    Returns
    -------
    results : dict
        Dictionary object containing optimization points + other info

    """

    if eval_args is None:
        eval_args = []
    random.seed(seed)

    if not use_gpu:
        tkwargs['device'] = 'cpu'

    if output_path is None:
        output_path = ''

    # Verbose print helper
    def vprint(*a, **k):
        # logger.debug(' '.join(a))
        # TODO: use logging instead of print statements
        if verbose:
            print(*a, **k)
            sys.stdout.flush()

    if not executor:
        serial = True
        executor = DummyExecutor()
        vprint('No executor given. Running in serial mode.')

    # Setup saving to file
    if output_path:
        path = full_path(output_path)
        assert os.path.exists(path), f'output_path does not exist {path}'

        def save(pop, prefix, generation):
            # TODO: implement this
            raise NotImplementedError

    else:
        # Dummy save
        def save(pop, prefix, generation):
            pass

    # parse VOCS
    variables = vocs['variables']
    variable_names = list(variables.keys())

    # get initial bounds
    bounds = torch.transpose(
        torch.vstack([torch.tensor(ele, **tkwargs) for _, ele in variables.items()]), 0, 1)

    # create normalization transforms for model inputs
    # inputs are normalized in [0,1]
    input_normalize = Normalize(len(variable_names), bounds)

    sampler_evaluate_args = {'verbose': verbose}

    # generate initial samples if no initial samples are given
    if restart_data_file is None:
        if initial_x is None:
            initial_x = draw_sobol_samples(bounds, 1, n_initial_samples)[0]

        # submit evaluation of initial samples
        initial_y = [executor.submit(sampler_evaluate,
                                     dict(zip(variable_names, x.cpu().numpy())),
                                     evaluate_f, *eval_args,
                                     **sampler_evaluate_args) for x in initial_x]

        train_x, train_y, train_c = collect_results(initial_y, vocs, **tkwargs)

    else:
        df = pd.read_json(restart_data_file)
        train_x = torch.tensor(df[vocs['variables'].keys()].to_numpy())
        train_y = torch.tensor(df[vocs['objectives'].keys()].to_numpy())
        train_c = torch.tensor(df[vocs['constraints'].keys()].to_numpy())

    # do optimization
    for i in range(n_steps):

        # get corrected values
        corrected_train_y, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)

        # standardize y training data - use xopt version to allow for nans
        standardized_train_y = standardize(corrected_train_y)

        # horiz. stack objective and constraint results for training/acq specification
        train_outputs = torch.hstack((standardized_train_y, corrected_train_c))

        model = create_model(train_x, train_outputs, input_normalize, custom_model)

        # get candidate point(s)
        candidates = gen_candidate(model, bounds, vocs, **kwargs)

        if verbose:
            vprint(candidates)

        # observe candidates
        fut = [executor.submit(sampler_evaluate,
                               dict(zip(variable_names, x.cpu().numpy())),
                               evaluate_f, *eval_args,
                               **sampler_evaluate_args) for x in candidates]
        try:
            new_x, new_y, new_c = collect_results(fut, vocs, **tkwargs)

            # add new observations to training data
            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))
            train_c = torch.vstack((train_c, new_c))

            # get feasibility values
            _, corrected_train_c = get_corrected_outputs(vocs, train_y, train_c)
            feas = torch.all(corrected_train_c < 0.0, dim=-1).reshape(-1, 1)
            constraint_status = corrected_train_c < 0.0

            # add data to pandas array for storage
            full_data = torch.hstack((train_x, train_y, train_c, constraint_status, feas))
            labels = list(vocs['variables'].keys()) + \
                     list(vocs['objectives'].keys()) + \
                     list(vocs['constraints'].keys()) + \
                     [ele + '_stat' for ele in vocs['constraints'].keys()] + \
                     ['feasibility']

            df = pd.DataFrame(full_data.numpy(), columns=labels)
            df.to_json(output_path + 'opt_data.json')

        except NoValidResultsError:
            print('No valid results found, skipping to next iteration')
            continue

    # horiz. stack objective and constraint results for training/acq specification
    train_outputs = torch.hstack((train_y, train_c))
    model = create_model(train_x, train_outputs, input_normalize, custom_model)

    results = {'inputs': train_x.cpu(),
               'objectives': train_y.cpu(),
               'constraints': train_c.cpu(),
               'constraint_status': constraint_status.cpu(),
               'feasibility': feas.cpu(),
               'model': model.cpu()}

    return results