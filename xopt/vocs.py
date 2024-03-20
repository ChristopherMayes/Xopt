from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame
from pydantic import ConfigDict, conlist, Field

from xopt.pydantic import XoptBaseModel


# Enums for objectives and constraints
class ObjectiveEnum(str, Enum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"

    # Allow any case
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member


class ConstraintEnum(str, Enum):
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"

    # Allow any case
    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member


class VOCS(XoptBaseModel):
    """
    Variables, Objectives, Constraints, and other Settings (VOCS) data structure
    to describe optimization problems.
    """

    variables: Dict[str, conlist(float, min_length=2, max_length=2)] = Field(
        default={},
        description="input variable names with a list of minimum and maximum values",
    )
    constraints: Dict[
        str, conlist(Union[float, ConstraintEnum], min_length=2, max_length=2)
    ] = Field(
        default={},
        description="constraint names with a list of constraint type and value",
    )
    objectives: Dict[str, ObjectiveEnum] = Field(
        default={}, description="objective names with type of objective"
    )
    constants: Dict[str, Any] = Field(
        default={}, description="constant names and values passed to evaluate function"
    )
    observables: List[str] = Field(
        default=[],
        description="observation names tracked alongside objectives and constraints",
    )

    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, extra="forbid"
    )

    @classmethod
    def from_yaml(cls, yaml_text):
        loaded = yaml.safe_load(yaml_text)
        return cls(**loaded)

    def as_yaml(self):
        return yaml.dump(self.model_dump(), default_flow_style=None, sort_keys=False)

    @property
    def bounds(self):
        """
        Returns a bounds array (mins, maxs) of shape (2, n_variables)
        Arrays of lower and upper bounds can be extracted by:
            mins, maxs = vocs.bounds
        """
        return np.array([v for _, v in sorted(self.variables.items())]).T

    @property
    def variable_names(self):
        """Returns a sorted list of variable names"""
        return list(sorted(self.variables.keys()))

    @property
    def objective_names(self):
        """Returns a sorted list of objective names"""
        return list(sorted(self.objectives.keys()))

    @property
    def constraint_names(self):
        """Returns a sorted list of constraint names"""
        if self.constraints is None:
            return []
        return list(sorted(self.constraints.keys()))

    @property
    def observable_names(self):
        return sorted(self.observables)

    @property
    def output_names(self):
        """
        Returns a list of expected output keys:
            (objectives + constraints + observables)
        Each sub-list is sorted.
        """
        full_list = self.objective_names
        for ele in self.constraint_names:
            if ele not in full_list:
                full_list += [ele]

        for ele in self.observable_names:
            if ele not in full_list:
                full_list += [ele]

        return full_list

    @property
    def constant_names(self):
        """Returns a sorted list of constraint names"""
        if self.constants is None:
            return []
        return list(sorted(self.constants.keys()))

    @property
    def all_names(self):
        """Returns all vocs names (variables, constants, objectives, constraints)"""
        return self.variable_names + self.constant_names + self.output_names

    @property
    def n_variables(self):
        """Returns the number of variables"""
        return len(self.variables)

    @property
    def n_constants(self):
        """Returns the number of constants"""
        return len(self.constants)

    @property
    def n_inputs(self):
        """Returns the number of inputs (variables and constants)"""
        return self.n_variables + self.n_constants

    @property
    def n_objectives(self):
        """Returns the number of objectives"""
        return len(self.objectives)

    @property
    def n_constraints(self):
        """Returns the number of constraints"""
        return len(self.constraints)

    @property
    def n_observables(self):
        """Returns the number of constraints"""
        return len(self.observables)

    @property
    def n_outputs(self):
        """
        Returns the number of outputs
            len(objectives + constraints + observables)
        """
        return len(self.output_names)

    def random_inputs(
        self,
        n: int = None,
        custom_bounds: dict = None,
        include_constants: bool = True,
        seed: int = None,
    ) -> list[dict]:
        """
        Uniform sampling of the variables.

        Returns a dict of inputs.

        If include_constants, the vocs.constants are added to the dict.

        Optional:
            n (integer) to make arrays of inputs, of size n.
            seed (integer) to initialize the random number generator

        """
        inputs = {}
        if seed is None:
            rng_sample_function = np.random.random
        else:
            rng = np.random.default_rng(seed=seed)
            rng_sample_function = rng.random

        # get bounds
        # if custom_bounds is specified then they will be clipped inside
        # vocs variable bounds
        if custom_bounds is None:
            bounds = self.variables
        else:
            variable_bounds = pd.DataFrame(self.variables)
            custom_bounds = pd.DataFrame(custom_bounds)
            custom_bounds = custom_bounds.clip(
                variable_bounds.iloc[0], variable_bounds.iloc[1], axis=1
            )
            bounds = custom_bounds.to_dict()
            for k in bounds.keys():
                bounds[k] = [bounds[k][i] for i in range(2)]

        for key, val in bounds.items():  # No need to sort here
            a, b = val
            n = n if n is not None else 1
            x = rng_sample_function(n)
            inputs[key] = x * a + (1 - x) * b

        # Constants
        if include_constants and self.constants is not None:
            inputs.update(self.constants)

        if n == 1:
            return [inputs]
        else:
            return pd.DataFrame(inputs).to_dict("records")

    def convert_dataframe_to_inputs(
        self, data: pd.DataFrame, include_constants=True
    ) -> pd.DataFrame:
        """
        Extracts only inputs from a dataframe.
        This will add constants if `include_constants` is true.
        """
        # make sure that the df keys only contain vocs variables
        if not set(self.variable_names) == set(data.keys()):
            raise ValueError(
                "input dataframe column set must equal set of vocs variables"
            )

        # only keep the variables
        inner_copy = data.copy()

        # append constants if requested
        if include_constants:
            constants = self.constants
            if constants is not None:
                for name, val in constants.items():
                    inner_copy[name] = val

        return inner_copy

    def convert_numpy_to_inputs(
        self, inputs: np.ndarray, include_constants=True
    ) -> pd.DataFrame:
        """
        convert 2D numpy array to list of dicts (inputs) for evaluation
        Assumes that the columns of the array match correspond to
        `sorted(self.vocs.variables.keys())

        """
        df = pd.DataFrame(inputs, columns=self.variable_names)
        return self.convert_dataframe_to_inputs(df, include_constants)

    # Extract optimization data (in correct column order)
    def variable_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "variable_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing variables according to `vocs.variables` in sorted
        order

        Args:
            data: Data to be processed.
            prefix: Prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_variable_data(self.variables, data, prefix=prefix)

    def objective_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "objective_",
        return_raw=False,
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing objective data transformed according to
        `vocs.objectives` such that we always assume minimization.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_objective_data(self.objectives, data, prefix, return_raw)

    def constraint_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "constraint_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing constraint data transformed according to
        `vocs.constraints` such that values that satisfy each constraint are negative.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_constraint_data(self.constraints, data, prefix)

    def observable_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "observable_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing observable data

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_observable_data(self.observable_names, data, prefix)

    def feasibility_data(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Dict]],
        prefix: str = "feasible_",
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing booleans denoting if a constraint is satisfied or
        not. Returned dataframe also contains a column `feasible` which denotes if
        all constraints are satisfied.

        Args:
            data: data to be processed.
            prefix: prefix added to column names.

        Returns:
            result: processed Dataframe
        """
        return form_feasibility_data(self.constraints, data, prefix)

    def normalize_inputs(self, input_points: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize input data (transform data into the range [0,1]) based on the
        variable ranges defined in the VOCS.

        Parameters
        ----------
        input_points : pd.DataFrame
            A DataFrame containing input data to be normalized.

        Returns
        -------
        pd.DataFrame
            A DataFrame with input data in the range [0,1] corresponding to the
            specified variable ranges. Contains columns equal to the intersection
            between `input_points` and `vocs.variable_names`.

        Notes
        -----

        If the input DataFrame is empty or no variable information is available in
        the VOCS, an empty DataFrame is returned.

        """
        normed_data = {}
        for name in self.variable_names:
            if name in input_points.columns:
                width = self.variables[name][1] - self.variables[name][0]
                normed_data[name] = (
                    input_points[name] - self.variables[name][0]
                ) / width

        if len(normed_data):
            return pd.DataFrame(normed_data)
        else:
            return pd.DataFrame([])

    def denormalize_inputs(self, input_points: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalize input data (transform data from the range [0,1]) based on the
        variable ranges defined in the VOCS.

        Parameters
        ----------
        input_points : pd.DataFrame
            A DataFrame containing normalized input data in the range [0,1].

        Returns
        -------
        pd.DataFrame
            A DataFrame with denormalized input data corresponding to the
            specified variable ranges. Contains columns equal to the intersection
            between `input_points` and `vocs.variable_names`.

        Notes
        -----

        If the input DataFrame is empty or no variable information is available in
        the VOCS, an empty DataFrame is returned.

        """
        denormed_data = {}
        for name in self.variable_names:
            if name in input_points.columns:
                width = self.variables[name][1] - self.variables[name][0]
                denormed_data[name] = (
                    input_points[name] * width + self.variables[name][0]
                )

        if len(denormed_data):
            return pd.DataFrame(denormed_data)
        else:
            return pd.DataFrame([])

    def validate_input_data(self, input_points: pd.DataFrame) -> None:
        """
        Validates input data. Raises an error if the input data does not satisfy
        requirements given by vocs.

        Args:
            input_points: input data to be validated.

        Returns:
            None

        Raises:
            ValueError: if input data does not satisfy requirements.
        """
        validate_input_data(self, input_points)

    def extract_data(self, data: pd.DataFrame, return_raw=False):
        """
        split dataframe into seperate dataframes for variables, objectives and
        constraints based on vocs - objective data is transformed based on
        `vocs.objectives` properties

        Args:
            data: dataframe to be split
            return_raw: if True, return untransformed objective data

        Returns:
            variable_data: dataframe containing variable data
            objective_data: dataframe containing objective data
            constraint_data: dataframe containing constraint data
        """
        variable_data = self.variable_data(data, "")
        objective_data = self.objective_data(data, "", return_raw)
        constraint_data = self.constraint_data(data, "")
        return variable_data, objective_data, constraint_data

    def select_best(self, data: pd.DataFrame, n=1):
        """
        get the best value and point for a given data set based on vocs
        - does not work for multi-objective problems
        - data that violates any constraints is ignored

        Args:
            data: dataframe to select best point from
            n: number of best points to return

        Returns:
            index: index of best point
            value: value of best point
            params: input parameters that give the best point
        """
        if self.n_objectives != 1:
            raise NotImplementedError(
                "cannot select best point when n_objectives is not 1"
            )

        feasible_data = self.feasibility_data(data)
        ascending_flag = {"MINIMIZE": True, "MAXIMIZE": False}
        obj = self.objectives[self.objective_names[0]]
        obj_name = self.objective_names[0]
        res = data[feasible_data["feasible"]].sort_values(
            obj_name, ascending=ascending_flag[obj]
        )[obj_name][:n]

        params = data.iloc[res.index.to_numpy()][self.variable_names].to_dict(
            orient="records"
        )[0]

        return res.index.to_numpy(), res.to_numpy(), params

    def cumulative_optimum(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the cumulative optimum for the given DataFrame.

        Parameters
        ----------
        data: pd.DataFrame
            Data for which the cumulative optimum shall be calculated.

        Returns
        -------
        pd.DataFrame
            Cumulative optimum for the given DataFrame.

        """
        if not self.objectives:
            raise RuntimeError("No objectives defined.")
        if data.empty:
            return pd.DataFrame()
        obj_name = self.objective_names[0]
        obj = self.objectives[obj_name]
        get_opt = np.nanmax if obj == "MAXIMIZE" else np.nanmin
        feasible = self.feasibility_data(data)["feasible"]
        feasible_obj_values = [
            data[obj_name].values[i] if feasible[i] else np.nan
            for i in range(len(data))
        ]
        cumulative_optimum = np.array(
            [get_opt(feasible_obj_values[: i + 1]) for i in range(len(data))]
        )
        return pd.DataFrame({f"best_{obj_name}": cumulative_optimum}, index=data.index)


# --------------------------------
# dataframe utilities

OBJECTIVE_WEIGHT = {"MINIMIZE": 1.0, "MAXIMIZE": -1.0}


def form_variable_data(variables: Union[Dict, DataFrame], data, prefix="variable_"):
    """
    Use variables dict to form a dataframe.
    """
    if not variables:
        return pd.DataFrame([])

    if not isinstance(data, DataFrame):
        data = pd.DataFrame(data)

    # Pick out columns in right order
    variables = sorted(variables)
    vdata = data.loc[:, variables].copy()
    # Rename to add prefix
    vdata.rename({k: prefix + k for k in variables})
    return vdata


def form_objective_data(
    objectives: Dict, data, prefix="objective_", return_raw: bool = False
):
    """
    Use objective dict and data (dataframe) to generate objective data (dataframe)

    Weights are applied to convert all objectives into minimization form unless
    `return_raw` is True

    Returns a dataframe with the objective data intended to be minimized.

    Missing or nan values will be filled with: np.inf

    """
    if not objectives:
        return pd.DataFrame([])

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    objectives_names = sorted(objectives.keys())

    if set(data.columns).issuperset(set(objectives_names)):
        # have all objectives, dont need to fill in missing ones
        weights = np.ones(len(objectives_names))
        for i, k in enumerate(objectives_names):
            operator = objectives[k].upper()
            if operator not in OBJECTIVE_WEIGHT:
                raise ValueError(f"Unknown objective operator: {operator}")

            weights[i] = 1.0 if return_raw else OBJECTIVE_WEIGHT[operator]

        oarr = data.loc[:, objectives_names].to_numpy() * weights
        oarr[np.isnan(oarr)] = np.inf
        odata = pd.DataFrame(
            oarr, columns=[prefix + k for k in objectives_names], index=data.index
        )
    else:
        # have to do this way because of missing objectives, even if slow
        # TODO: pre-allocate 2D array
        length = data.shape[0]
        array_list = []
        for i, k in enumerate(objectives_names):
            if k not in data:
                array_list.append(np.full((length, 1), np.inf))
                continue
            operator = objectives[k].upper()
            if operator not in OBJECTIVE_WEIGHT:
                raise ValueError(f"Unknown objective operator: {operator}")

            weight = 1.0 if return_raw else OBJECTIVE_WEIGHT[operator]
            arr = data.loc[:, [k]].to_numpy() * weight
            arr[np.isnan(arr)] = np.inf
            array_list.append(arr)

        odata = pd.DataFrame(
            np.hstack(array_list),
            columns=[prefix + k for k in objectives_names],
            index=data.index,
        )

    return odata


def form_constraint_data(constraints: Dict, data: pd.DataFrame, prefix="constraint_"):
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe). A
    constraint is satisfied if the evaluation is < 0.

    Args:
        constraints: Dictionary of constraints
        data: Dataframe with the data to be evaluated
        prefix: Prefix to use for the transformed data in the dataframe

    Returns a dataframe with the constraint data.

    Missing or nan values will be filled with: np.inf

    """
    if not constraints:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in sorted(list(constraints)):
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        x = data[k]
        op, d = constraints[k]
        op = op.upper()  # Allow any case

        if op == "GREATER_THAN":  # x > d -> x-d > 0
            cvalues = -(x - d)
        elif op == "LESS_THAN":  # x < d -> d-x > 0
            cvalues = -(d - x)
        else:
            raise ValueError(f"Unknown constraint operator: {op}")

        cdata[prefix + k] = cvalues.fillna(np.inf)  # Protect against nans
    return cdata


def form_observable_data(observables: List, data: pd.DataFrame, prefix="observable_"):
    """
    Use constraint dict and data (dataframe) to generate constraint data (dataframe). A
    constraint is satisfied if the evaluation is < 0.

    Args:
        observables: Dictonary of constraints
        data: Dataframe with the data to be evaluated
        prefix: Prefix to use for the transformed data in the dataframe

    Returns a dataframe with the constraint data.

    Missing or nan values will be filled with: np.inf

    """
    if not observables:
        return pd.DataFrame([])

    data = pd.DataFrame(data)  # cast to dataframe

    cdata = pd.DataFrame(index=data.index)

    for k in observables:
        # Protect against missing data
        if k not in data:
            cdata[prefix + k] = np.inf
            continue

        ovalues = data[k]
        cdata[prefix + k] = ovalues.fillna(np.inf)  # Protect against nans
    return cdata


def form_feasibility_data(constraints: Dict, data, prefix="feasible_"):
    """
    Use constraint dict and data to identify feasible points in the the dataset.

    Returns a dataframe with the feasible data.
    """
    if not constraints:
        df = pd.DataFrame(index=data.index)
        df["feasible"] = True
        return df

    data = pd.DataFrame(data)
    c_prefix = "constraint_"
    cdata = form_constraint_data(constraints, data, prefix=c_prefix)
    fdata = pd.DataFrame()
    for k in sorted(list(constraints)):
        fdata[prefix + k] = cdata[c_prefix + k] <= 0
    # if all row values are true, then the row is feasible
    fdata["feasible"] = fdata.all(axis=1)
    return fdata


def validate_input_data(vocs: VOCS, data: pd.DataFrame) -> None:
    variable_data = data.loc[:, vocs.variable_names].values
    bounds = vocs.bounds

    is_out_of_bounds_lower = variable_data < bounds[0, :]
    is_out_of_bounds_upper = variable_data > bounds[1, :]
    bad_mask = np.logical_or(is_out_of_bounds_upper, is_out_of_bounds_lower)
    any_bad = bad_mask.any()

    if any_bad:
        raise ValueError(
            f"input points at indices {np.nonzero(bad_mask.any(axis=0))} are not valid"
        )
