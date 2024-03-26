import os.path
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Union

import botorch.settings
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.priors import GammaPrior, Prior
from pydantic import ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch.nn import Module

from xopt.generators.bayesian.base_model import ModelConstructor
from xopt.generators.bayesian.models.prior_mean import CustomMean
from xopt.generators.bayesian.utils import get_training_data
from xopt.pydantic import decode_torch_module

DECODERS = {"torch.float32": torch.float32, "torch.float64": torch.float64}
MIN_INFERRED_NOISE_LEVEL = 1e-4


class StandardModelConstructor(ModelConstructor):
    """
    A class for constructing independent models for each objective and constraint.

    Attributes
    ----------
    name : str
        The name of the model (frozen).

    use_low_noise_prior : bool
        Specify if the model should assume a low noise environment.

    covar_modules : Dict[str, Kernel]
        Covariance modules for GP models.

    mean_modules : Dict[str, Module]
        Prior mean modules for GP models.

    trainable_mean_keys : List[str]
        List of prior mean modules that can be trained.


    Methods
    -------
    likelihood
        Get the likelihood for the model, considering the low noise prior.

    build_model(input_names, outcome_names, data, input_bounds, dtype, device)
        Construct independent models for each objective and constraint.

    build_mean_module(name, input_transform, outcome_transform)
        Build the mean module for the output specified by name.

    """

    name: str = Field("standard", frozen=True)
    use_low_noise_prior: bool = Field(
        True, description="specify if model should assume a low noise environment"
    )
    covar_modules: Dict[str, Kernel] = Field(
        {}, description="covariance modules for GP models"
    )
    mean_modules: Dict[str, Module] = Field(
        {}, description="prior mean modules for GP models"
    )
    trainable_mean_keys: List[str] = Field(
        [], description="list of prior mean modules that can be trained"
    )
    transform_inputs: Union[Dict[str, bool], bool] = Field(
        True,
        description="specify if inputs should be transformed inside the gp "
        "model, can optionally specify a dict of specifications",
    )
    custom_noise_prior: Optional[Prior] = Field(
        None,
        description="specify custom noise prior for the GP likelihood, "
        "overwrites value specified by use_low_noise_prior",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("covar_modules", "mean_modules", mode="before")
    def validate_torch_modules(cls, v):
        if not isinstance(v, dict):
            raise ValueError("must be dict")
        else:
            for key, val in v.items():
                if isinstance(val, str):
                    if val.startswith("base64:"):
                        v[key] = decode_torch_module(val)
                    elif os.path.exists(val):
                        v[key] = torch.load(val)

        return v

    @field_validator("trainable_mean_keys")
    def validate_trainable_mean_keys(cls, v, info: ValidationInfo):
        for name in v:
            assert name in info.data["mean_modules"]
        return v

    @property
    def likelihood(self) -> Likelihood:
        """
        Get the likelihood for the model, considering the low noise prior and or a
        custom noise prior.

        Returns
        -------
        Likelihood
            The likelihood for the model.

        """
        if self.custom_noise_prior is not None:
            return GaussianLikelihood(
                noise_prior=self.custom_noise_prior,
            )

        if self.use_low_noise_prior:
            return GaussianLikelihood(
                noise_prior=GammaPrior(1.0, 100.0),
            )
        else:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
            return likelihood

    def build_model(
        self,
        input_names: List[str],
        outcome_names: List[str],
        data: pd.DataFrame,
        input_bounds: Dict[str, List] = None,
        dtype: torch.dtype = torch.double,
        device: Union[torch.device, str] = "cpu",
    ) -> ModelListGP:
        """
        Construct independent models for each objective and constraint.

        Parameters
        ----------
        input_names : List[str]
            Names of input variables.
        outcome_names : List[str]
            Names of outcome variables.
        data : pd.DataFrame
            Data used for training the model.
        input_bounds : Dict[str, List], optional
            Bounds for input variables.
        dtype : torch.dtype, optional
            Data type for the model (default is torch.double).
        device : Union[torch.device, str], optional
            Device on which to perform computations (default is "cpu").

        Returns
        -------
        ModelListGP
            A list of trained botorch models.

        """
        # build model
        tkwargs = {"dtype": dtype, "device": device}
        models = []

        covar_modules = deepcopy(self.covar_modules)
        mean_modules = deepcopy(self.mean_modules)
        for outcome_name in outcome_names:
            input_transform = self._get_input_transform(
                outcome_name, input_names, input_bounds, tkwargs
            )
            outcome_transform = Standardize(1)
            covar_module = self._get_module(covar_modules, outcome_name)
            mean_module = self.build_mean_module(
                outcome_name, mean_modules, input_transform, outcome_transform
            )

            # get training data
            train_X, train_Y, train_Yvar = get_training_data(
                input_names, outcome_name, data
            )
            # collect arguments into a single dict
            kwargs = {
                "input_transform": input_transform,
                "outcome_transform": outcome_transform,
                "covar_module": covar_module,
                "mean_module": mean_module,
            }

            if train_Yvar is None:
                # train basic single-task-gp model
                models.append(
                    self.build_single_task_gp(
                        train_X.to(**tkwargs),
                        train_Y.to(**tkwargs),
                        likelihood=self.likelihood,
                        **kwargs,
                    )
                )
            else:
                # train heteroskedastic single-task-gp model
                # turn off warnings
                models.append(
                    self.build_heteroskedastic_gp(
                        train_X.to(**tkwargs),
                        train_Y.to(**tkwargs),
                        train_Yvar.to(**tkwargs),
                        **kwargs,
                    )
                )
        # check all specified modules were added to the model
        if covar_modules:
            warnings.warn(
                f"Covariance modules for output names {[k for k, v in self.covar_modules.items()]} "
                f"could not be added to the model."
            )
        if mean_modules:
            warnings.warn(
                f"Mean modules for output names {[k for k, v in self.mean_modules.items()]} "
                f"could not be added to the model."
            )

        return ModelListGP(*models)

    def build_mean_module(
        self, name, mean_modules, input_transform, outcome_transform
    ) -> Optional[CustomMean]:
        """
        Build the mean module for the output specified by name.

        Parameters
        ----------
        name : str
            The name of the output.
        mean_modules: dict
            The dictionary of mean modules.
        input_transform : InputTransform
            Transform for input variables.
        outcome_transform : OutcomeTransform
            Transform for outcome variables.

        Returns
        -------
        Optional[CustomMean]
            The mean module for the output, or None if not specified.

        """
        mean_module = self._get_module(mean_modules, name)
        if mean_module is not None:
            fixed_model = False if name in self.trainable_mean_keys else True
            mean_module = CustomMean(
                mean_module, input_transform, outcome_transform, fixed_model=fixed_model
            )
        return mean_module

    @staticmethod
    def _get_module(base, name):
        """
        Get the module for a given name.

        Parameters
        ----------
        base : Union[Module, Dict[str, Module]]
            The base module or a dictionary of modules.
        name : str
            The name of the module.

        Returns
        -------
        Module
            The retrieved module.

        """
        if isinstance(base, Module):
            return deepcopy(base)
        elif isinstance(base, dict):
            return deepcopy(base.pop(name, None))
        else:
            return None

    def _get_input_transform(self, outcome_name, input_names, input_bounds, tkwargs):
        """get input transform based on the supplied bounds and attributes"""
        # get input bounds
        if input_bounds is None:
            bounds = None
        else:
            bounds = torch.vstack(
                [torch.tensor(input_bounds[name], **tkwargs) for name in input_names]
            ).T

        # create transform
        input_transform = Normalize(len(input_names), bounds=bounds)

        # remove input transform if the bool is False or the dict entry is false
        if isinstance(self.transform_inputs, bool):
            if not self.transform_inputs:
                input_transform = None
        if (
            isinstance(self.transform_inputs, dict)
            and outcome_name in self.transform_inputs
        ):
            if not self.transform_inputs[outcome_name]:
                input_transform = None

        # remove warnings if input transform is None
        if input_transform is None:
            botorch.settings.validate_input_scaling(False)

        return input_transform
