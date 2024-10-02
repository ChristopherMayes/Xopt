import time
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from xopt import Evaluator, Xopt
from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer, NumericalOptimizer
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs


def f(X):
    return torch.sum(X**2, dim=-1)


class TestNumericalOptimizers:
    @patch.multiple(NumericalOptimizer, __abstractmethods__=set())
    def test_base(self):
        NumericalOptimizer()

    def test_lbfgs_optimizer(self):
        optimizer = LBFGSOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(f, bounds, ncandidate)
                assert candidates.shape == torch.Size([ncandidate, ndim])

        # test max time
        max_time_optimizer = LBFGSOptimizer(max_time=0.01)
        ndim = 1
        bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
        for ncandidate in [1, 3]:
            start_time = time.time()
            candidates = max_time_optimizer.optimize(f, bounds, ncandidate)
            assert time.time() - start_time < 0.01
            assert candidates.shape == torch.Size([ncandidate, ndim])

    def test_grid_optimizer(self):
        optimizer = GridOptimizer()
        for ndim in [1, 3]:
            bounds = torch.stack((torch.zeros(ndim), torch.ones(ndim)))
            for ncandidate in [1, 3]:
                candidates = optimizer.optimize(f, bounds, ncandidate)
                assert candidates.shape == torch.Size([ncandidate, ndim])

    def test_in_xopt(self):
        vocs = deepcopy(tnk_vocs)

        # can only explore one objective
        vocs.objectives = {}
        vocs.observables = ["y1"]

        generator = BayesianExplorationGenerator(
            vocs=vocs, numerical_optimizer=GridOptimizer(n_grid_points=2)
        )

        evaluator = Evaluator(function=evaluate_TNK)

        X = Xopt(generator=generator, evaluator=evaluator, vocs=vocs)

        X.evaluate_data(
            pd.DataFrame({"x1": [1.0, 0.75, 3.14, 0], "x2": [0.7, 0.95, 0, 3.14]})
        )

        X.step()
        assert np.allclose(
            X.data.iloc[-1][X.vocs.variable_names].to_numpy().astype(float),
            np.zeros(2),
            atol=1e-4,
        )
