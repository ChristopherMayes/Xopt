import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from xopt.resources.testing import TEST_VOCS_BASE, TEST_VOCS_DATA
from xopt.vocs import ObjectiveEnum, VOCS


class TestVOCS(object):
    def test_init(self):
        from xopt.vocs import VOCS

        # test various configurations
        vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"f": "MINIMIZE"},
        )
        assert vocs.n_inputs == 1
        assert vocs.n_outputs == 1
        assert vocs.n_constraints == 0

    def test_variable_validation(self):
        with pytest.raises(ValidationError):
            VOCS(
                variables={"x": [1, 0]},
            )

    def test_empty_objectives(self):
        VOCS(
            variables={"x": [0, 1]},
        )

    def test_output_names(self):
        test_vocs = VOCS(
            variables={"x": [0, 1]},
            objectives={"y1": "MINIMIZE"},
            constraints={"c1": ["GREATER_THAN", 0], "c2": ["LESS_THAN", 0]},
        )
        assert test_vocs.output_names == ["y1", "c1", "c2"]

    def test_constraint_specification(self):
        good_constraint_list = [
            ["LESS_THAN", 0],
            ["GREATER_THAN", 0],
            ["less_than", 0],
            ["greater_than", 0],
        ]

        for ele in good_constraint_list:
            VOCS(variables={"x": [0, 1]}, constraints={"c": ele})

        bad_constraint_list = [
            ["LESS_THAN"],
            ["GREATER_TAN", 0],
            [0, "less_than"],
            ["greater_than", "ahc"],
        ]

        for ele in bad_constraint_list:
            with pytest.raises(ValidationError):
                VOCS(variables={"x": [0, 1]}, constraints={"c": ele})

    def test_from_yaml(self):
        Y = """
        variables:
          a: [0, 1e3] # Note that 1e3 usually parses as a str with YAML.
          b: [-1, 1]
        objectives:
          c: maximize
          d: minimize
        constraints:
          e: ['Less_than', 2]
          f: ['greater_than', 0]
        constants:
          g: 1234

        """

        vocs = VOCS.from_yaml(Y)
        assert vocs.constraint_names == ["e", "f"]
        assert vocs.variables == {"a": [0, 1e3], "b": [-1, 1]}
        assert vocs.objectives == {"c": "MAXIMIZE", "d": "MINIMIZE"}
        assert vocs.constants == {"g": 1234}

        assert vocs.objectives["c"] == ObjectiveEnum.MAXIMIZE
        assert vocs.objectives["d"] == ObjectiveEnum.MINIMIZE

        assert vocs.constraints["e"] == ["LESS_THAN", 2]
        assert vocs.constraints["f"] == ["GREATER_THAN", 0]

        assert vocs.n_inputs == 3
        assert vocs.n_outputs == 4

    def test_random_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        n_samples = 10
        data = pd.DataFrame(vocs.random_inputs(n_samples))
        assert data.shape == (n_samples, vocs.n_inputs)

        test_inputs = TEST_VOCS_BASE.random_inputs(5, include_constants=False)
        assert len(test_inputs) == 5

        test_inputs = TEST_VOCS_BASE.random_inputs()
        assert isinstance(test_inputs[0]["x1"], float)

    def test_serialization(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.model_dump_json()

        vocs.variables["a"] = np.array([1, 2])
        vocs.model_dump_json()

    def test_properties(self):
        vocs = deepcopy(TEST_VOCS_BASE)

        assert vocs.n_variables == 2
        assert vocs.n_inputs == 3
        assert vocs.n_outputs == 2
        assert vocs.n_constraints == 1
        assert vocs.n_objectives == 1
        assert vocs.variable_names == ["x1", "x2"]
        assert vocs.objective_names == ["y1"]

        # modify the vocs and retest
        vocs.variables = {name: vocs.variables[name] for name in ["x1"]}
        assert vocs.n_variables == 1
        assert vocs.n_inputs == 2
        assert vocs.n_outputs == 2
        assert vocs.variable_names == ["x1"]

    def test_random_sampling_custom_bounds(self):
        vocs = deepcopy(TEST_VOCS_BASE)

        custom_bounds = {"x1": [0.5, 0.75], "x2": [7.5, 15.0]}

        with pytest.warns(RuntimeWarning):
            random_input_data = vocs.random_inputs(100, custom_bounds=custom_bounds)

        random_input_data = pd.DataFrame(random_input_data)
        assert all(random_input_data["x1"] < 0.75)
        assert all(random_input_data["x1"] > 0.5)
        assert all(random_input_data["x2"] > 7.5)
        assert all(random_input_data["x2"] < 10.0)

        # test custom bounds within the vocs domain -- no warnings should be raised
        in_domain_custom_bounds = {"x1": [0.5, 0.75], "x2": [0.5, 0.75]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vocs.random_inputs(100, custom_bounds=in_domain_custom_bounds)

        # test wrong type
        with pytest.raises(TypeError):
            vocs.random_inputs(100, custom_bounds=1)

        # test custom bounds entirely outside the vocs domain or specified incorrectly
        bad_custom_bounds = [
            {"x1": [10.0, 10.75], "x2": [7.5, 15.0]},
            {"x1": [0.75, 0.5], "x2": [7.5, 15.0]},
        ]
        for ele in bad_custom_bounds:
            with pytest.raises(ValueError):
                vocs.random_inputs(100, custom_bounds=ele)

    def test_duplicate_outputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        assert vocs.output_names == ["y1", "c1"]

        vocs.objectives = {"y1": "MAXIMIZE", "d1": "MINIMIZE"}
        vocs.observables = ["y1", "c1"]

        assert vocs.output_names == ["d1", "y1", "c1"]
        assert vocs.n_outputs == 3

    def test_variable_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA

        res = vocs.variable_data(test_data)
        assert np.array_equal(res.to_numpy(), test_data.loc[:, ["x1", "x2"]].to_numpy())

    def test_objective_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA
        test_data["y2"] = 1.0

        res = vocs.objective_data(test_data)
        assert np.array_equal(res.to_numpy(), test_data.loc[:, ["y1"]].to_numpy())

        vocs.objectives.update({"y2": "MAXIMIZE"})
        res = vocs.objective_data(test_data)
        assert np.array_equal(
            res.to_numpy(),
            test_data.loc[:, ["y1", "y2"]].to_numpy() * np.array([1, -1]),
        )

        test_data2 = test_data.drop(columns=["y1"])
        res = vocs.objective_data(test_data2)
        assert np.array_equal(
            res.to_numpy(),
            test_data.loc[:, ["y1", "y2"]].to_numpy() * np.array([np.inf, -1]),
        )

    def test_convert_dataframe_to_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = TEST_VOCS_DATA

        with pytest.raises(ValueError):
            vocs.convert_dataframe_to_inputs(test_data)

        res = vocs.convert_dataframe_to_inputs(test_data[vocs.variable_names])
        assert "constant1" in res

    def test_validate_input_data(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)

        # test good data
        test_vocs.validate_input_data(pd.DataFrame({"x1": 0.5, "x2": 1.0}, index=[0]))

        # test bad data
        with pytest.raises(ValueError):
            test_vocs.validate_input_data(
                pd.DataFrame({"x1": 0.5, "x2": 11.0}, index=[0])
            )

        with pytest.raises(ValueError):
            test_vocs.validate_input_data(
                pd.DataFrame({"x1": [-0.5, 2.5], "x2": [1.0, 11.0]})
            )

    def test_select_best(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.1, 0.1, 0.1],
                "x2": [0.1, 0.1, 0.1, 0.1],
                "c1": [1.0, 0.0, 1.0, 0.0],
                "y1": [0.5, 0.1, 1.0, 1.5],
            }
        )

        # test maximization
        vocs.objectives[vocs.objective_names[0]] = "MAXIMIZE"
        idx, val, _ = vocs.select_best(test_data)
        assert idx == [2]
        assert val == [1.0]

        vocs.constraints = {}
        idx, val, _ = vocs.select_best(test_data)
        assert idx == [3]
        assert val == [1.5]

        # test returning multiple best values -- sorted by best value
        idx, val, _ = vocs.select_best(test_data, 2)
        assert np.allclose(idx, np.array([3, 2]))
        assert np.allclose(val, np.array([1.5, 1.0]))

        # test minimization
        vocs.objectives[vocs.objective_names[0]] = "MINIMIZE"
        vocs.constraints = {"c1": ["GREATER_THAN", 0.5]}
        idx, val, _ = vocs.select_best(test_data)
        assert idx == [0]
        assert val == [0.5]

        vocs.constraints = {}
        idx, val, _ = vocs.select_best(test_data)
        assert idx == 1
        assert val == 0.1

        # test error handling
        with pytest.raises(RuntimeError):
            vocs.select_best(pd.DataFrame())

        vocs.constraints = {"c1": ["GREATER_THAN", 10.5]}
        with pytest.raises(RuntimeError):
            vocs.select_best(pd.DataFrame())

    @pytest.mark.filterwarnings("ignore: All-NaN axis encountered")
    def test_cumulative_optimum(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        obj_name = vocs.objective_names[0]
        test_data = pd.DataFrame(
            {
                obj_name: [np.nan, 0.0, -0.4, 0.6, np.nan, -0.7],
                vocs.constraint_names[0]: [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        vocs.objectives = {}
        with pytest.raises(RuntimeError):
            _ = vocs.cumulative_optimum(test_data)
        vocs.objectives = {obj_name: "MINIMIZE"}
        assert vocs.cumulative_optimum(pd.DataFrame()).empty
        cumulative_minimum = vocs.cumulative_optimum(test_data)
        assert np.array_equal(
            cumulative_minimum[f"best_{obj_name}"].values,
            np.array([np.nan, np.nan, -0.4, -0.4, -0.4, -0.7]),
            equal_nan=True,
        )
        vocs.objectives[obj_name] = "MAXIMIZE"
        cumulative_maximum = vocs.cumulative_optimum(test_data)
        assert np.array_equal(
            cumulative_maximum[f"best_{obj_name}"].values,
            np.array([np.nan, np.nan, -0.4, 0.6, 0.6, 0.6]),
            equal_nan=True,
        )

    def test_normalize_inputs(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = pd.DataFrame({"x1": [0.5, 0.2], "x2": [0.5, 0.75]})
        normed_data = vocs.normalize_inputs(test_data)
        assert normed_data["x1"].to_list() == [0.5, 0.2]
        assert normed_data["x2"].to_list() == [0.05, 0.075]

        assert np.equal(
            vocs.denormalize_inputs(normed_data)[vocs.variable_names].to_numpy(),
            test_data[vocs.variable_names].to_numpy(),
        ).all()

        test_data.pop("x1")
        normed_data = vocs.normalize_inputs(test_data)
        assert normed_data["x2"].to_list() == [0.05, 0.075]

        assert np.equal(
            vocs.denormalize_inputs(normed_data)[test_data.columns].to_numpy(),
            test_data[test_data.columns].to_numpy(),
        ).all()

        test_data = pd.DataFrame({"x1": 0.5}, index=[0])
        normed_data = vocs.normalize_inputs(test_data)
        assert normed_data["x1"].to_list() == [0.5]

        assert np.equal(
            vocs.denormalize_inputs(normed_data)[test_data.columns].to_numpy(),
            test_data[test_data.columns].to_numpy(),
        ).all()

        # test with extra data
        test_data = pd.DataFrame({"x1": [0.5, 0.2], "x2": [0.5, 0.75], "a": [1, 1]})
        normed_data = vocs.normalize_inputs(test_data)
        assert {"x1", "x2"} == set(normed_data.columns)

        assert np.equal(
            vocs.denormalize_inputs(normed_data)[vocs.variable_names].to_numpy(),
            test_data[vocs.variable_names].to_numpy(),
        ).all()

    def test_extract_data(self):
        vocs = deepcopy(TEST_VOCS_BASE)
        test_data = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.4, 0.4],
                "x2": [0.1, 0.2, 0.3, 0.2],
                "c1": [1.0, 0.0, 1.0, 0.0],
                "y1": [0.5, 0.1, 1.0, 1.5],
            }
        )

        # test default functionality
        data = vocs.extract_data(test_data)
        assert set(data[0].keys()) == {"x1", "x2"}
        assert set(data[1].keys()) == {"y1"}
        assert set(data[2].keys()) == {"c1"}
        for ele in data[:-1]:  # ignore observable data
            assert len(ele) == 4

        # test return_valid functionality
        data = vocs.extract_data(test_data, return_valid=True)
        assert data[0]["x1"].to_list() == [0.1, 0.4]
        assert data[1]["y1"].to_list() == [0.5, 1.0]

        for ele in data[:-1]:  # ignore observable data
            assert len(ele) == 2

        # test w/o constraints
        vocs = deepcopy(TEST_VOCS_BASE)
        vocs.constraints = {}
        data = vocs.extract_data(test_data)
        assert len(data[0]) == 4
        assert data[2].empty
