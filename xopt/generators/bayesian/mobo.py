from typing import Union

import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.utils import draw_sobol_samples
from pydantic import Field
from torch import Tensor

from xopt.generators.bayesian.bayesian_generator import MultiObjectiveBayesianGenerator

from xopt.generators.bayesian.objectives import create_mobo_objective
from xopt.numerical_optimizer import LBFGSOptimizer


class MOBOGenerator(MultiObjectiveBayesianGenerator):
    name = "mobo"
    supports_batch_generation: bool = True
    use_pf_as_initial_points: bool = Field(
        False,
        description="flag to specify if pf front points are to be used during "
        "optimization of the acquisition function",
    )
    __doc__ = """Implements Multi-Objective Bayesian Optimization using the Expected
            Hypervolume Improvement acquisition function"""

    def _get_objective(self):
        return create_mobo_objective(self.vocs)

    def get_acquisition(self, model):
        """
        Returns a function that can be used to evaluate the acquisition function
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply fixed features if specified in the generator
        if self.fixed_features is not None:
            # get input dim
            dim = len(self.model_input_names)
            columns = []
            values = []
            for name, value in self.fixed_features.items():
                columns += [self.model_input_names.index(name)]
                values += [value]

            acq = FixedFeatureAcquisitionFunction(
                acq_function=acq, d=dim, columns=columns, values=values
            )

        return acq

    def _get_acquisition(self, model):
        inputs = self.get_input_data(self.data)
        sampler = self._get_sampler(model)

        if self.log_transform_acquisition_function:
            acqclass = qLogNoisyExpectedHypervolumeImprovement
        else:
            acqclass = qNoisyExpectedHypervolumeImprovement

        acq = acqclass(
            model,
            X_baseline=inputs,
            constraints=self._get_constraint_callables(),
            ref_point=self.torch_reference_point,
            sampler=sampler,
            objective=self._get_objective(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq

    def _get_initial_conditions(self, n_candidates=1) -> Union[Tensor, None]:
        """
        generate initial candidates for optimizing the acquisition function based on
        the pareto front

        Returns:
            A `num_restarts x q x d` tensor of initial conditions.

        """
        if self.use_pf_as_initial_points:
            if isinstance(self.numerical_optimizer, LBFGSOptimizer):
                bounds = self._get_optimization_bounds()
                num_restarts = self.numerical_optimizer.n_restarts

                pf_locations, _ = self.get_pareto_front()

                # if there is no pareto front just return None to revert back to
                # default behavior
                if pf_locations is None:
                    return None

                initial_points = torch.clone(pf_locations)

                # add the q dimension
                initial_points = initial_points.unsqueeze(1)
                initial_points = initial_points.expand([-1, n_candidates, -1])

                # initial_points must equal the number of restarts
                if len(initial_points) < num_restarts:
                    # add random points to the list inside the bounds
                    sobol_samples = draw_sobol_samples(
                        bounds, num_restarts - len(initial_points), n_candidates
                    )

                    initial_points = torch.cat([initial_points, sobol_samples])
                elif len(initial_points) > num_restarts:
                    # if there are too many select the first `num_restarts` points at
                    # random
                    initial_points = initial_points[
                        torch.randperm(len(initial_points))
                    ][:num_restarts]

                return initial_points
            else:
                raise RuntimeWarning(
                    "cannot use PF as initial optimization points "
                    "for non-LBFGS optimizers, ignoring flag"
                )

        return None
