from abc import ABC, abstractmethod
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt
from torch import Tensor

from xopt.pydantic import XoptBaseModel


class NumericalOptimizer(XoptBaseModel, ABC):
    name: str = Field("base_numerical_optimizer", frozen=True)
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def optimize(
        self, function: AcquisitionFunction, bounds: Tensor, n_candidates=1, **kwargs
    ):
        """optimize a function to produce a number of candidate points that
        minimize the function"""
        pass


class LBFGSOptimizer(NumericalOptimizer):
    name: str = Field("LBFGS", frozen=True)
    n_restarts: PositiveInt = Field(
        20, description="number of restarts during acquistion function optimization"
    )
    max_iter: PositiveInt = Field(2000)
    max_time: Optional[PositiveFloat] = Field(
        None, description="maximum time for optimizing"
    )

    model_config = ConfigDict(validate_assignment=True)

    def optimize(self, function, bounds, n_candidates=1, **kwargs):
        assert isinstance(bounds, Tensor)
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        candidates, out = optimize_acqf(
            acq_function=function,
            bounds=bounds,
            q=n_candidates,
            raw_samples=self.n_restarts,
            num_restarts=self.n_restarts,
            timeout_sec=self.max_time,
            options={"maxiter": self.max_iter},
            **kwargs,
        )
        return candidates


class GridOptimizer(NumericalOptimizer):
    """
    Numerical optimizer that uses a brute-force grid search to find the optimium.

    Parameters
    ----------
    n_grid_points: PositiveInt, optional
        Number of mesh points per axis to sample. Algorithm time scales as
        `n_grd_points`^`input_dimension`

    """

    name: str = Field("grid", frozen=True)
    n_grid_points: PositiveInt = Field(
        10, description="number of grid points per axis used for optimization"
    )

    def optimize(self, function, bounds, n_candidates=1, **kwargs):
        assert isinstance(bounds, Tensor)
        # create mesh
        if len(bounds) != 2:
            raise ValueError("bounds must have the shape [2, ndim]")

        dim = len(bounds[0])
        # add in a machine eps
        eps = 1e-5
        linspace_list = [
            torch.linspace(
                bounds.T[i][0] + eps, bounds.T[i][1] - eps, self.n_grid_points
            )
            for i in range(dim)
        ]

        xx = torch.meshgrid(*linspace_list, indexing="ij")
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T

        # evaluate the function on grid points
        f_values = function(mesh_pts.unsqueeze(1))

        # get the best n_candidates
        _, indicies = torch.sort(f_values)
        x_min = mesh_pts[indicies.squeeze().flipud()]
        return x_min[:n_candidates]
