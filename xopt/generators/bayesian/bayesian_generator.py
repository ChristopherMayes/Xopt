import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, qUpperConfidenceBound
from botorch.models.model import Model
from botorch.sampling import get_sampler
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from gpytorch import Module
from pydantic import Field, field_validator, PositiveInt, SerializeAsAny
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from xopt.errors import XoptError
from xopt.generator import Generator
from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.custom_botorch.constrained_acquisition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.log_acquisition_function import (
    LogAcquisitionFunction,
)
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.objectives import (
    create_constraint_callables,
    create_mc_objective,
    CustomXoptObjective,
)
from xopt.generators.bayesian.turbo import (
    OptimizeTurboController,
    SafetyTurboController,
    TurboController,
)
from xopt.generators.bayesian.utils import (
    interpolate_points,
    rectilinear_domain_union,
    set_botorch_weights,
    validate_turbo_controller_base,
)
from xopt.generators.bayesian.visualize import visualize_generator_model
from xopt.numerical_optimizer import GridOptimizer, LBFGSOptimizer, NumericalOptimizer
from xopt.pydantic import decode_torch_module

logger = logging.getLogger()


# It seems pydantic v2 does not auto-register models anymore
# So one option is to have explicit unions for model subclasses
# The other is to define a descriminated union with name as key, but that stops name field from
# getting exported, and we want to keep it for readability
# See https://github.com/pydantic/pydantic/discussions/5785

# Update: using parent model now seems to work, keeping this just in case
# T_ModelConstructor = Union[StandardModelConstructor, TimeDependentModelConstructor]
# T_NumericalOptimizer = Union[LBFGSOptimizer, GridOptimizer]


class BayesianGenerator(Generator, ABC):
    """Bayesian Generator for Bayesian Optimization.

    Attributes:
    -----------
    name : str
        The name of the Bayesian Generator.

    model : Optional[Model]
        The BoTorch model used by the generator to perform optimization.

    n_monte_carlo_samples : int
        The number of Monte Carlo samples to use in the optimization process.

    turbo_controller : SerializeAsAny[Optional[TurboController]]
        The Turbo Controller for trust-region Bayesian Optimization.

    use_cuda : bool
        A flag to enable or disable CUDA usage if available.

    gp_constructor : SerializeAsAny[ModelConstructor]
        The constructor used to generate the model for Bayesian Optimization.

    numerical_optimizer : SerializeAsAny[NumericalOptimizer]
        The optimizer used to optimize the acquisition function in Bayesian Optimization.

    max_travel_distances : Optional[List[float]]
        The limits for travel distances between points in normalized space.

    fixed_features : Optional[Dict[str, float]]
        The fixed features used in Bayesian Optimization.

    computation_time : Optional[pd.DataFrame]
        A data frame tracking computation time in seconds.

    log_transform_acquisition_function: Optional[bool]
        Flag to determine if final acquisition function value should be
        log-transformed before optimization.

    n_interpolate_samples: Optional[PositiveInt]
        Number of interpolation points to generate between last observation and next
        observation, requires n_candidates to be 1.

    n_candidates : int
        The number of candidates to generate in each optimization step.

    Methods:
    --------
    generate(self, n_candidates: int) -> List[Dict]:
        Generate candidates for Bayesian Optimization.

    add_data(self, new_data: pd.DataFrame):
        Add new data to the generator for Bayesian Optimization.

    train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        Train a Bayesian model for Bayesian Optimization.

    propose_candidates(self, model, n_candidates=1) -> Tensor:
        Propose candidates for Bayesian Optimization.

    get_input_data(self, data: pd.DataFrame) -> torch.Tensor:
        Get input data in torch.Tensor format.

    get_acquisition(self, model) -> AcquisitionFunction:
        Get the acquisition function for Bayesian Optimization.

    """

    name = "base_bayesian_generator"
    model: Optional[Model] = Field(
        None, description="botorch model used by the generator to perform optimization"
    )
    n_monte_carlo_samples: int = Field(
        128, description="number of monte carlo samples to use"
    )
    turbo_controller: SerializeAsAny[Optional[TurboController]] = Field(
        default=None, description="turbo controller for trust-region BO"
    )
    use_cuda: bool = Field(False, description="flag to enable cuda usage if available")
    gp_constructor: SerializeAsAny[ModelConstructor] = Field(
        StandardModelConstructor(), description="constructor used to generate model"
    )
    numerical_optimizer: SerializeAsAny[NumericalOptimizer] = Field(
        LBFGSOptimizer(),
        description="optimizer used to optimize the acquisition " "function",
    )
    max_travel_distances: Optional[List[float]] = Field(
        None,
        description="limits for travel distance between points in normalized space",
    )
    fixed_features: Optional[Dict[str, float]] = Field(
        None, description="fixed features used in Bayesian optimization"
    )
    computation_time: Optional[pd.DataFrame] = Field(
        None,
        description="data frame tracking computation time in seconds",
    )
    log_transform_acquisition_function: Optional[bool] = Field(
        False,
        description="flag to log transform the acquisition function before optimization",
    )
    custom_objective: Optional[CustomXoptObjective] = Field(
        None,
        description="custom objective for optimization, replaces objective specified by VOCS",
    )
    n_interpolate_points: Optional[PositiveInt] = None
    memory_length: Optional[PositiveInt] = None

    n_candidates: int = 1

    @field_validator("model", mode="before")
    def validate_torch_modules(cls, v):
        if isinstance(v, str):
            if v.startswith("base64:"):
                v = decode_torch_module(v)
            elif os.path.exists(v):
                v = torch.load(v)
        return v

    @field_validator("gp_constructor", mode="before")
    def validate_gp_constructor(cls, value):
        constructor_dict = {"standard": StandardModelConstructor}
        if value is None:
            value = StandardModelConstructor()
        elif isinstance(value, ModelConstructor):
            value = value
        elif isinstance(value, str):
            if value in constructor_dict:
                value = constructor_dict[value]()
            else:
                raise ValueError(f"{value} not found")
        elif isinstance(value, dict):
            name = value.pop("name")
            if name in constructor_dict:
                value = constructor_dict[name](**value)
            else:
                raise ValueError(f"{value} not found")

        return value

    @field_validator("numerical_optimizer", mode="before")
    def validate_numerical_optimizer(cls, value):
        optimizer_dict = {"grid": GridOptimizer, "LBFGS": LBFGSOptimizer}
        if value is None:
            value = LBFGSOptimizer()
        elif isinstance(value, NumericalOptimizer):
            pass
        elif isinstance(value, str):
            if value in optimizer_dict:
                value = optimizer_dict[value]()
            else:
                raise ValueError(f"{value} not found")
        elif isinstance(value, dict):
            name = value.pop("name")
            if name in optimizer_dict:
                value = optimizer_dict[name](**value)
            else:
                raise ValueError(f"{value} not found")
        return value

    @field_validator("turbo_controller", mode="before")
    def validate_turbo_controller(cls, value, info: ValidationInfo):
        """note default behavior is no use of turbo"""
        controller_dict = {
            "optimize": OptimizeTurboController,
            "safety": SafetyTurboController,
        }
        value = validate_turbo_controller_base(value, controller_dict, info)

        return value

    @field_validator("computation_time", mode="before")
    def validate_computation_time(cls, value):
        if isinstance(value, dict):
            value = pd.DataFrame(value)

        return value

    def add_data(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data], axis=0)

    def generate(self, n_candidates: int) -> list[dict]:
        """
        Generate candidates using Bayesian Optimization.

        Parameters:
        -----------
        n_candidates : int
            The number of candidates to generate in each optimization step.

        Returns:
        --------
        List[Dict]
            A list of dictionaries containing the generated candidates.

        Raises:
        -------
        NotImplementedError
            If the number of candidates is greater than 1, and the generator does not
            support batch candidate generation.

        RuntimeError
            If no data is contained in the generator, the 'add_data' method should be
            called to add data before generating candidates.

        Notes:
        ------
        This method generates candidates for Bayesian Optimization based on the
        provided number of candidates. It updates the internal model with the current
        data and calculates the candidates by optimizing the acquisition function.
        The method returns the generated candidates in the form of a list of dictionaries.
        """

        self.n_candidates = n_candidates
        if n_candidates > 1 and not self.supports_batch_generation:
            raise NotImplementedError(
                "This Bayesian algorithm does not currently support parallel candidate "
                "generation"
            )

        # if no data exists raise error
        if self.data is None:
            raise RuntimeError(
                "no data contained in generator, call `add_data` "
                "method to add data, see also `Xopt.random_evaluate()`"
            )

        else:
            # dict to track runtimes
            timing_results = {}

            # update internal model with internal data
            start_time = time.perf_counter()
            model = self.train_model(self.get_training_data(self.data))
            timing_results["training"] = time.perf_counter() - start_time

            # propose candidates given model
            start_time = time.perf_counter()
            candidates = self.propose_candidates(model, n_candidates=n_candidates)
            timing_results["acquisition_optimization"] = (
                time.perf_counter() - start_time
            )

            # post process candidates
            result = self._process_candidates(candidates)

            # append timing results to dataframe (if it exists)
            if self.computation_time is not None:
                self.computation_time = pd.concat(
                    (
                        self.computation_time,
                        pd.DataFrame(timing_results, index=[0]),
                    ),
                    ignore_index=True,
                )
            else:
                self.computation_time = pd.DataFrame(timing_results, index=[0])

            if self.n_interpolate_points is not None:
                if self.n_candidates > 1:
                    raise RuntimeError(
                        "cannot generate interpolated points for "
                        "multiple candidate generation"
                    )
                else:
                    assert len(result) == 1
                    result = interpolate_points(
                        pd.concat(
                            (self.data.iloc[-1:][self.vocs.variable_names], result),
                            axis=0,
                            ignore_index=True,
                        ),
                        num_points=self.n_interpolate_points,
                    )

            return result.to_dict("records")

    def train_model(self, data: pd.DataFrame = None, update_internal=True) -> Module:
        """
        Returns a ModelListGP containing independent models for the objectives and
        constraints

        """
        if data is None:
            data = self.get_training_data(self.data)
        if data.empty:
            raise ValueError("no data available to build model")

        # get input bounds
        variable_bounds = deepcopy(self.vocs.variables)

        # if turbo restrict points is true then set the bounds to the trust region
        # bounds
        if self.turbo_controller is not None:
            if self.turbo_controller.restrict_model_data:
                variable_bounds = dict(
                    zip(
                        self.vocs.variable_names,
                        self.turbo_controller.get_trust_region(self).numpy().T,
                    )
                )

        # add fixed feature bounds if requested
        if self.fixed_features is not None:
            # get bounds for each fixed_feature (vocs bounds take precedent)
            for key in self.fixed_features:
                if key not in variable_bounds:
                    if key not in data:
                        raise KeyError(
                            "generator data needs to contain fixed feature "
                            f"column name `{key}`"
                        )
                    f_data = data[key]
                    bounds = [f_data.min(), f_data.max()]
                    if bounds[1] - bounds[0] < 1e-8:
                        bounds[1] = bounds[0] + 1e-8
                    variable_bounds[key] = bounds

        _model = self.gp_constructor.build_model(
            self.model_input_names,
            self.vocs.output_names,
            data,
            {name: variable_bounds[name] for name in self.model_input_names},
            **self._tkwargs,
        )

        if update_internal:
            self.model = _model
        return _model

    def propose_candidates(self, model, n_candidates=1):
        """
        given a GP model, propose candidates by numerically optimizing the
        acquisition function

        """
        # update TurBO state if used with the last `n_candidates` points
        if self.turbo_controller is not None:
            self.turbo_controller.update_state(self, n_candidates)

        # calculate optimization bounds
        bounds = self._get_optimization_bounds()

        # get acquisition function
        acq_funct = self.get_acquisition(model)

        # get initial candidates to start acquisition function optimization
        initial_points = self._get_initial_conditions(n_candidates)

        # get candidates
        candidates = self.numerical_optimizer.optimize(
            acq_funct, bounds, n_candidates, batch_initial_conditions=initial_points
        )
        return candidates

    def get_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get training data used to train the GP model

        If a turbo controller is specified with the flag `restrict_model_data` this
        will return a subset of data that is inside the trust region.

        Parameters:
        -----------
        data : pd.DataFrame
            The data in the form of a pandas DataFrame.

        Returns:
        --------
        data : pd.DataFrame
            A subset of data used to train the model form of a pandas DataFrame.

        """
        if self.turbo_controller is not None:
            if self.turbo_controller.restrict_model_data:
                data = self.turbo_controller.get_data_in_trust_region(data, self)

        return data

    def get_input_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Convert input data to a torch tensor.

        Parameters:
        -----------
        data : pd.DataFrame
            The input data in the form of a pandas DataFrame.

        Returns:
        --------
        torch.Tensor
            A torch tensor containing the input data.

        Notes:
        ------
        This method takes a pandas DataFrame as input data and converts it into a
        torch tensor. It specifically selects columns corresponding to the model's
        input names (variables), and the resulting tensor is configured with the data
        type and device settings from the generator.
        """
        return torch.tensor(data[self.model_input_names].to_numpy(), **self._tkwargs)

    def get_acquisition(self, model):
        """
        Define the acquisition function based on the given GP model.

        Parameters:
        -----------
        model : Model
            The BoTorch model to be used for generating the acquisition function.

        Returns:
        --------
        acqusition_function : AcqusitionFunction

        Raises:
        -------
        ValueError
            If the provided 'model' is None. A valid model is required to create the
            acquisition function.
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)

        # apply constraints if specified in vocs
        # TODO: replace with direct constrainted acquisition function calls
        # see SampleReducingMCAcquisitionFunction in botorch for rationale
        if len(self.vocs.constraints):
            try:
                sampler = acq.sampler
            except AttributeError:
                sampler = self._get_sampler(model)

            acq = ConstrainedMCAcquisitionFunction(
                model, acq, self._get_constraint_callables(), sampler=sampler
            )

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

        if self.log_transform_acquisition_function:
            acq = LogAcquisitionFunction(acq)

        return acq

    def get_optimum(self):
        """select the best point(s) given by the
        model using the Posterior mean"""
        c_posterior_mean = ConstrainedMCAcquisitionFunction(
            self.model,
            qUpperConfidenceBound(
                model=self.model, beta=0.0, objective=self._get_objective()
            ),
            self._get_constraint_callables(),
        )

        result = self.numerical_optimizer.optimize(
            c_posterior_mean, self._get_bounds(), 1
        )

        return self._process_candidates(result)

    def visualize_model(self, **kwargs):
        """displays the GP models"""
        return visualize_generator_model(self, **kwargs)

    def _get_initial_conditions(self, n_candidates=1) -> Union[Tensor, None]:
        """overwrite if algorithm should specifiy initial candidates for optimizing
        the acquisition function"""
        return None

    def _process_candidates(self, candidates: Tensor):
        """process pytorch candidates from optimizing the acquisition function"""
        logger.debug(f"Best candidate from optimize {candidates}")

        if self.fixed_features is not None:
            results = pd.DataFrame(
                candidates.detach().cpu().numpy(), columns=self._candidate_names
            )
            for name, val in self.fixed_features.items():
                results[name] = val

        else:
            results = self.vocs.convert_numpy_to_inputs(
                candidates.detach().cpu().numpy(), include_constants=False
            )

        return results

    def _get_sampler(self, model):
        input_data = self.get_input_data(self.data)
        sampler = get_sampler(
            model.posterior(input_data),
            sample_shape=torch.Size([self.n_monte_carlo_samples]),
        )
        return sampler

    @abstractmethod
    def _get_acquisition(self, model):
        pass

    def _get_objective(self):
        """return default objective (scalar objective) determined by vocs or if
        defined in custom_objective"""
        # check to make sure that if we specify a custom objective that no objectives
        # are specified in vocs
        if self.custom_objective is not None:
            if self.vocs.n_objectives:
                raise RuntimeError(
                    "cannot specify objectives in VOCS "
                    "and a custom objective for the generator at the "
                    "same time"
                )

            return self.custom_objective
        else:
            return create_mc_objective(self.vocs, self._tkwargs)

    def _get_constraint_callables(self):
        """return constratint callable determined by vocs"""
        constraint_callables = create_constraint_callables(self.vocs)
        if len(constraint_callables) == 0:
            constraint_callables = None
        return constraint_callables

    @property
    def _tkwargs(self):
        # set device and data type for generator
        device = "cpu"
        if self.use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                warnings.warn(
                    "Cuda requested in generator options but not found on "
                    "machine! Using CPU instead"
                )

        return {"dtype": torch.double, "device": device}

    @property
    def model_input_names(self):
        """variable names corresponding to trained model"""
        variable_names = self.vocs.variable_names
        if self.fixed_features is not None:
            for name, val in self.fixed_features.items():
                if name not in variable_names:
                    variable_names += [name]
        return variable_names

    @property
    def _candidate_names(self):
        """variable names corresponding to generated candidates"""
        variable_names = self.vocs.variable_names
        if self.fixed_features is not None:
            for name in self.fixed_features:
                if name in variable_names:
                    variable_names.remove(name)
        return variable_names

    def _get_bounds(self):
        """convert bounds from vocs to torch tensors"""
        return torch.tensor(self.vocs.bounds, **self._tkwargs)

    def _get_optimization_bounds(self):
        """
        Get optimization bounds based on the union of several domains.

        Returns:
        --------
        torch.Tensor
            Tensor containing the optimized bounds.

        Notes:
        ------
        This method calculates the optimization bounds based on several factors:

        - If 'max_travel_distances' is specified, the bounds are modified to limit
            the maximum travel distances between points in normalized space.
        - If 'turbo_controller' is not None, the bounds are updated according to the
            trust region specified by the controller.
        - If 'fixed_features' are included in the variable names from the VOCS,
            the bounds associated with those features are removed.

        """
        bounds = deepcopy(self._get_bounds())

        # if specified modify bounds to limit maximum travel distances
        if self.max_travel_distances is not None:
            max_travel_bounds = self._get_max_travel_distances_region(bounds)
            bounds = rectilinear_domain_union(bounds, max_travel_bounds)

        # if using turbo, update turbo state and set bounds according to turbo state
        if self.turbo_controller is not None:
            # set the best value
            turbo_bounds = self.turbo_controller.get_trust_region(self)
            bounds = rectilinear_domain_union(bounds, turbo_bounds)

        # if fixed features key is in vocs then we need to remove the bounds
        # associated with that key
        if self.fixed_features is not None:
            # grab variable name indices that are NOT in fixed features
            indicies = []
            for idx, name in enumerate(self.vocs.variable_names):
                if name not in self.fixed_features:
                    indicies += [idx]

            # grab indexed bounds
            bounds = bounds.T[indicies].T

        return bounds

    def _get_max_travel_distances_region(self, bounds):
        """
        Calculate the region for maximum travel distances based on the current bounds
        and the last observation.

        Parameters:
        -----------
        bounds : torch.Tensor
            The optimization bounds based on the union of several domains.

        Returns:
        --------
        torch.Tensor
            The bounds for the maximum travel distances region.

        Raises:
        -------
        ValueError
            If the length of max_travel_distances does not match the number of
            variables in bounds.

        Notes:
        ------
        This method calculates the region in which the next candidates for
        optimization should be generated based on the maximum travel distances
        specified. The region is centered around the last observation in the
        optimization space. The `max_travel_distances` parameter should be a list of
        maximum travel distances for each variable.

        """
        if len(self.max_travel_distances) != bounds.shape[-1]:
            raise ValueError(
                f"length of max_travel_distances must match the number of "
                f"variables {bounds.shape[-1]}"
            )

        # get last point
        if self.data is None:
            raise ValueError(
                "No data exists to specify max_travel_distances "
                "from, add data first to use during BO"
            )
        last_point = torch.tensor(
            self.data[self.vocs.variable_names].iloc[-1].to_numpy(), **self._tkwargs
        )

        # bound lengths based on vocs for normalization
        lengths = self.vocs.bounds[1, :] - self.vocs.bounds[0, :]

        # get maximum travel distances
        max_travel_distances = torch.tensor(
            self.max_travel_distances, **self._tkwargs
        ) * torch.tensor(lengths, **self._tkwargs)
        max_travel_bounds = torch.stack(
            (last_point - max_travel_distances, last_point + max_travel_distances)
        )

        return max_travel_bounds


class MultiObjectiveBayesianGenerator(BayesianGenerator, ABC):
    name = "multi_objective_bayesian_generator"
    reference_point: dict[str, float] = Field(
        description="dict specifying reference point for multi-objective "
        "optimization",
    )

    supports_multi_objective: bool = True

    @field_validator("reference_point")
    def validate_reference_point(cls, v, info: ValidationInfo):
        objective_names = info.data["vocs"].objective_names
        assert set(v.keys()) == set(objective_names)

        return v

    @property
    def torch_reference_point(self):
        pt = []
        for name in self.vocs.objective_names:
            try:
                ref_val = self.reference_point[name]
            except KeyError:
                raise XoptError(
                    "need to specify reference point for the following "
                    f"objective {name}"
                )
            if self.vocs.objectives[name] == "MINIMIZE":
                pt += [-ref_val]
            elif self.vocs.objectives[name] == "MAXIMIZE":
                pt += [ref_val]
            else:
                raise ValueError(
                    f"objective type {self.vocs.objectives[name]} not\
                    supported"
                )

        return torch.tensor(pt, **self._tkwargs)

    def _get_scaled_data(self):
        """get scaled input/objective data for use with botorch logic which assumes
        maximization for each objective"""
        var_df, obj_df, _, _ = self.vocs.extract_data(
            self.data, return_valid=True, return_raw=True
        )

        variable_data = torch.tensor(var_df[self.vocs.variable_names].to_numpy())
        objective_data = torch.tensor(obj_df[self.vocs.objective_names].to_numpy())
        weights = set_botorch_weights(self.vocs).to(**self._tkwargs)[
            : self.vocs.n_objectives
        ]
        return variable_data, objective_data * weights

    def calculate_hypervolume(self):
        """compute hypervolume given data"""

        # compute hypervolume
        bd = DominatedPartitioning(
            ref_point=self.torch_reference_point, Y=self._get_scaled_data()[1]
        )
        volume = bd.compute_hypervolume().item()

        return volume

    def get_pareto_front(self):
        """compute the pareto front x/y values given data"""
        variable_data, objective_data = self._get_scaled_data()
        obj_data = torch.vstack(
            (self.torch_reference_point.unsqueeze(0), objective_data)
        )
        var_data = torch.vstack(
            (
                torch.full_like(variable_data[0], float("Nan")).unsqueeze(0),
                variable_data,
            )
        )
        non_dominated = is_non_dominated(obj_data)

        weights = set_botorch_weights(self.vocs).to(**self._tkwargs)[
            : self.vocs.n_objectives
        ]

        # note need to undo weights for real number output
        # only return values if non nan values exist
        if torch.all(torch.isnan(var_data[non_dominated])):
            return None, None
        else:
            return var_data[non_dominated], obj_data[non_dominated] / weights


def formatted_base_docstring():
    return "\nBase Generator\n---------------\n" + BayesianGenerator.__doc__
