import logging
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from enum import Enum
from threading import Lock
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import ConfigDict, Field, model_validator

from xopt.errors import XoptError
from xopt.pydantic import NormalExecutor, XoptBaseModel
from xopt.utils import get_function, get_function_defaults, safe_call

logger = logging.getLogger(__name__)


class ExecutorEnum(str, Enum):
    thread_pool_executor = "ThreadPoolExecutor"
    process_pool_executor = "ProcessPoolExecutor"
    normal_executor = "NormalExecutor"


class Evaluator(XoptBaseModel):
    """
    Xopt Evaluator for handling the parallel execution of an evaluate function.

    Parameters
    ----------
    function : Callable
        Function to evaluate.
    function_kwargs : dict, default={}
        Any kwargs to pass on to this function.
    max_workers : int, default=1
        Maximum number of workers.
    executor : NormalExecutor
        NormalExecutor or any instantiated Executor object
    vectorized : bool, default=False
        If true, lists of evaluation points will be sent to the evaluator
        function to be processed in parallel instead of evaluated seperately via
        mapping.
    """

    function: Callable
    max_workers: int = Field(1, ge=1)
    executor: NormalExecutor = Field(exclude=True)  # Do not serialize
    function_kwargs: dict = Field({})
    vectorized: bool = Field(False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def validate_all(cls, values):
        f = get_function(values["function"])
        kwargs = values.get("function_kwargs", {})
        kwargs = {**get_function_defaults(f), **kwargs}
        values["function"] = f
        values["function_kwargs"] = kwargs

        max_workers = values.pop("max_workers", 1)

        executor = values.pop("executor", None)
        if not executor:
            if max_workers > 1:
                executor = ProcessPoolExecutor(max_workers=max_workers)
            else:
                executor = DummyExecutor()

        # Cast as a NormalExecutor
        values["executor"] = NormalExecutor[type(executor)](executor=executor)
        values["max_workers"] = max_workers

        return values

    def evaluate(self, input: Dict, **kwargs):
        """
        Evaluate a single input dict using Evaluator.function with
        Evaluator.function_kwargs.

        Further kwargs are passed to the function.

        Inputs:
            inputs: dict of inputs to be evaluated
            **kwargs: additional kwargs to pass to the function

        Returns:
            function(input, **function_kwargs_updated)

        """
        return self.safe_function(input, **{**self.function_kwargs, **kwargs})

    def evaluate_data(
        self,
        input_data: Union[
            pd.DataFrame,
            List[Dict[str, float]],
            Dict[str, List[float]],
            Dict[str, float],
        ],
    ) -> pd.DataFrame:
        """evaluate dataframe of inputs"""
        if self.vectorized:
            output_data = self.safe_function(input_data, **self.function_kwargs)
        else:
            # This construction is needed to avoid a pickle error
            # translate input data into pandas dataframes
            if not isinstance(input_data, DataFrame):
                try:
                    input_data = DataFrame(input_data)
                except ValueError:
                    input_data = DataFrame(input_data, index=[0])

            inputs = input_data.to_dict("records")

            funcs = [self.function] * len(inputs)
            kwargs = [self.function_kwargs] * len(inputs)

            output_data = self.executor.map(
                safe_function1_for_map,
                funcs,
                inputs,
                kwargs,
            )

        return pd.concat(
            [input_data, DataFrame(output_data, index=input_data.index)], axis=1
        )

    def safe_function(self, *args, **kwargs):
        """
        Safely call the function, handling exceptions.

        Note that this should not be submitted to fuu
        """
        return safe_function(self.function, *args, **kwargs)

    def submit(self, input: Dict):
        """submit a single input to the executor

        Parameters
        ----------
        input : dict

        Returns
        -------
        Future  : Future object
        """
        if not isinstance(input, dict):
            raise ValueError("input must be a dictionary")
        # return self.executor.submit(self.function, input, **self.function_kwargs)
        # Must call a function outside of the classs
        # See: https://stackoverflow.com/questions/44144584/typeerror-cant-pickle-thread-lock-objects
        return self.executor.submit(
            safe_function, self.function, input, **self.function_kwargs
        )

    def submit_data(self, input_data: pd.DataFrame):
        """submit dataframe of inputs to executor"""
        input_data = pd.DataFrame(input_data)  # cast to dataframe for consistency

        if self.vectorized:
            # Single submission, cast to numpy array
            inputs = input_data.to_dict(orient="list")
            for key, value in inputs.items():
                inputs[key] = np.array(value)
            futures = [self.submit(inputs)]  # Single item
        else:
            # Do not use iterrows or itertuples.
            futures = [self.submit(inputs) for inputs in input_data.to_dict("records")]

        return futures


def safe_function1_for_map(function, inputs, kwargs):
    """
    Safely call the function, handling exceptions.
    """
    return safe_function(function, inputs, **kwargs)


def safe_function(function, *args, **kwargs):
    safe_outputs = safe_call(function, *args, **kwargs)
    return process_safe_outputs(safe_outputs)


def process_safe_outputs(outputs: Dict):
    """
    Process the outputs of safe_call, flattening the output.
    """
    o = {}
    error = False
    error_str = ""
    if outputs["exception"]:
        error = True
        error_str = outputs["traceback"]

    result = outputs["result"]
    if isinstance(result, dict):
        o.update(result)
    elif not error:
        o["xopt_non_dict_result"] = result  # result is not a dict, but preserve anyway
        error = True
        error_str = "Non-dict result"

    # Add in error bool
    o["xopt_runtime"] = outputs["runtime"]
    o["xopt_error"] = error
    if error:
        o["xopt_error_str"] = error_str
    return o


def validate_outputs(outputs: DataFrame):
    """
    Looks for Xopt errors in the outputs and raises XoptError if found.
    """

    # Handles dicts or dataframes
    if not np.any(outputs["xopt_error"]):
        return

    if "xopt_non_dict_result" in outputs:
        result = outputs["xopt_non_dict_result"]
        raise XoptError(
            "Xopt evaluator returned a non-dict result, type is: "
            f"{type(result)}, result is: {result}"
        )
    else:
        error_string = "Xopt evaluator caught exception(s):\n\n"
        for i in range(len(outputs["xopt_error_str"])):
            error_string += f"Evaluation index {i}:\n"
            error_string += str(outputs["xopt_error_str"].iloc[i])
            error_string += "\n"

        raise XoptError(error_string)


class DummyExecutor(Executor):
    """
    Dummy executor.

    From: https://stackoverflow.com/questions/10434593/dummyexecutor-for-pythons-futures

    """

    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
