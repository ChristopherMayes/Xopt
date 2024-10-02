import numpy as np

from xopt import VOCS


def construct_zdt(n_dims, problem_index=1):
    """construct Xopt versions of the multiobjective ZDT test problems"""
    vocs = VOCS(
        variables={f"x{i + 1}": [0, 1] for i in range(n_dims)},
        objectives={"f1": "MINIMIZE", "f2": "MINIMIZE"},
    )

    if problem_index == 1:
        # ZDT1
        def evaluate(input_dict):
            x = np.array([input_dict[f"x{i + 1}"] for i in range(n_dims)])

            f1 = x[0]
            g = 1 + (9 / (n_dims - 1)) * np.sum(x[1:])
            h = 1 - np.sqrt(f1 / g)
            f2 = g * h

            return {"f1": f1, "f2": f2, "g": g}
    elif problem_index == 2:

        def evaluate(input_dict):
            x = np.array([input_dict[f"x{i + 1}"] for i in range(n_dims)])

            f1 = x[0]
            g = 1 + (9 / (n_dims - 1)) * np.sum(x[1:])
            h = 1 - (f1 / g) ** 2
            f2 = g * h

            return {"f1": f1, "f2": f2, "g": g}
    elif problem_index == 3:

        def evaluate(input_dict):
            x = np.array([input_dict[f"x{i + 1}"] for i in range(n_dims)])

            f1 = x[0]
            g = 1 + (9 / (n_dims - 1)) * np.sum(x[1:])
            h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
            f2 = g * h

            return {"f1": f1, "f2": f2, "g": g}
    else:
        raise NotImplementedError()

    reference_point = {"f1": 1.0, "f2": 1.0}

    return vocs, evaluate, reference_point
