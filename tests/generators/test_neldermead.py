import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.optimize import minimize

from xopt import Xopt
from xopt.generators.scipy.neldermead import NelderMeadGenerator
from xopt.resources.test_functions.ackley_20 import ackley, vocs as ackleyvocs
from xopt.resources.test_functions.rosenbrock import (
    rosenbrock,
    rosenbrock2_vocs as rbvocs,
)
from xopt.resources.testing import TEST_VOCS_BASE


def compare(X, X2):
    """Compare two Xopt objects"""
    y = json.loads(X.json())
    y2 = json.loads(X2.json())
    y.pop("data")
    y2.pop("data")
    assert y == y2
    # For unclear reasons, column order changes on reload....
    data = X.data.drop(["xopt_runtime", "xopt_error"], axis=1)
    data2 = X2.data.drop(["xopt_runtime", "xopt_error"], axis=1)
    # On reload, index is not a range index anymore!
    pd.testing.assert_frame_equal(data, data2, check_index_type=False)


class TestNelderMeadGenerator:
    def test_simplex_generate_multiple_points(self):
        gen = NelderMeadGenerator(vocs=TEST_VOCS_BASE)

        # Try to generate multiple samples
        with pytest.raises(NotImplementedError):
            gen.generate(2)

    def test_simplex_options(self):
        gen = NelderMeadGenerator(vocs=TEST_VOCS_BASE)

        with pytest.raises(ValidationError):
            gen.initial_point = {"x1": None, "x2": 0}

        with pytest.raises(ValidationError):
            gen.initial_simplex = {
                "x1": [0, 1],
                "x2": 0,
            }

        with pytest.raises(ValidationError):
            gen.xatol = None

        with pytest.raises(ValidationError):
            gen.fatol = None

        gen.xatol = 1e-3
        gen.fatol = 1e-3
        assert gen.xatol == 1e-3
        assert gen.fatol == 1e-3

    def test_simplex_agreement(self):
        """Compare between Vanilla Simplex and Xopt Simplex in full auto run mode"""

        # Scipy Simplex
        result = minimize(
            rosenbrock, [-1, -1], method="Nelder-Mead", options={"adaptive": True}
        )
        result = result.x

        # Xopt Simplex
        YAML = """
        generator:
            name: neldermead
            initial_point: {x0: -1, x1: -1}
            adaptive: true
            xatol: 0.0001
            fatol: 0.0001
        evaluator:
            function: xopt.resources.test_functions.rosenbrock.evaluate_rosenbrock
        vocs:
            variables:
                x0: [-5, 5]
                x1: [-5, 5]
            objectives: {y: MINIMIZE}
        """
        X = Xopt.from_yaml(YAML)
        X.run()

        # Results should be the same
        xbest = X.data.iloc[X.data["y"].argmin()]
        assert (
            xbest["x0"] == result[0] and xbest["x1"] == result[1]
        ), "Xopt Simplex does not match the vanilla one"

    @pytest.mark.parametrize(
        "fun,fstring,x0,v",
        [
            (rosenbrock, "rosenbrock.evaluate_rosenbrock", [-1, -1], rbvocs),
            (ackley, "ackley_20.evaluate_ackley_np", [4] * 20, ackleyvocs),
        ],
    )
    def test_simplex_agreement_steps(self, fun, fstring, x0, v):
        """Compare between Vanilla Simplex and Xopt Simplex step by step"""

        inputs = []

        def wrap(x):
            inputs.append(x)
            return fun(x)

        result = minimize(
            wrap, np.array(x0), method="Nelder-Mead", options={"adaptive": True}
        )
        scipy_data = np.array(inputs)
        print(f"Have {scipy_data.shape[0]} steps")

        config = {
            "generator": {
                "name": "neldermead",
                "initial_point": {f"x{i}": x0[i] for i in range(len(x0))},
                "adaptive": True,
                "xatol": 1e-4,
                "fatol": 1e-4,
            },
            "evaluator": {"function": f"xopt.resources.test_functions.{fstring}"},
            "vocs": v.model_dump(),
        }
        X = Xopt.from_dict(config)
        for i in range(scipy_data.shape[0]):
            X.step()
            data = X.vocs.variable_data(X.data).to_numpy()
            if not np.array_equal(data, scipy_data[: i + 1, :]):
                raise Exception

        data = X.vocs.variable_data(X.data).to_numpy()
        assert np.array_equal(data, scipy_data)

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        assert np.array_equal(
            xbest, result.x
        ), "Xopt Simplex does not match the vanilla one"

    @pytest.mark.parametrize(
        "fun,fstring,x0,v,cstep",
        [
            (rosenbrock, "rosenbrock.evaluate_rosenbrock", [-1, -1], rbvocs, 10),
            (ackley, "ackley_20.evaluate_ackley_np", [4] * 20, ackleyvocs, 200),
        ],
    )
    def test_resume_consistency(self, fun, fstring, x0, v, cstep):
        """Compare between Vanilla Simplex and Xopt Simplex while deserializing at every step"""
        inputs = []

        def wrap(x):
            inputs.append(x)
            return fun(x)

        result = minimize(
            wrap, np.array(x0), method="Nelder-Mead", options={"adaptive": True}
        )
        scipy_data = np.array(inputs)

        config = {
            "generator": {
                "name": "neldermead",
                "initial_point": {f"x{i}": x0[i] for i in range(len(x0))},
                "adaptive": True,
                "xatol": 1e-4,
                "fatol": 1e-4,
            },
            "evaluator": {"function": f"xopt.resources.test_functions.{fstring}"},
            "vocs": v.model_dump(),
        }
        X = Xopt.from_dict(config)
        X.step()

        for i in range(scipy_data.shape[0] - 1):
            # For performance, only check some steps
            if i % cstep == 0 or i == scipy_data.shape[0] - 2:
                state = X.json()
                X2 = Xopt.model_validate(json.loads(state))
                compare(X, X2)

                samples = X.generator.generate(1)
                samples2 = X2.generator.generate(1)
                assert samples == samples2
                compare(X, X2)

                state = X.json()
                X3 = Xopt.model_validate(json.loads(state))
                compare(X, X3)

                X.evaluate_data(samples)
                X2.evaluate_data(samples2)
                X3.evaluate_data(samples2)
                compare(X, X2)
                compare(X, X3)
            else:
                samples = X.generator.generate(1)
                X.evaluate_data(samples)

        data = X.vocs.variable_data(X.data).to_numpy()
        assert data.shape == scipy_data.shape
        assert np.array_equal(data, scipy_data)

        data = X2.vocs.variable_data(X2.data).to_numpy()
        assert data.shape == scipy_data.shape
        # Numerical precision issues with using strings for floats, need tolerance
        assert np.allclose(data, scipy_data, rtol=0, atol=1e-10)

        data = X3.vocs.variable_data(X3.data).to_numpy()
        assert data.shape == scipy_data.shape
        assert np.allclose(data, scipy_data, rtol=0, atol=1e-10)

        idx, best, _ = X.vocs.select_best(X.data)
        xbest = X.vocs.variable_data(X.data.loc[idx, :]).to_numpy().flatten()
        assert np.array_equal(
            xbest, result.x
        ), "Xopt Simplex does not match the vanilla one"
