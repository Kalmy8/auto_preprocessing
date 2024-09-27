from __future__ import annotations

from itertools import product
from typing import Any, Generator


def get_cv2_function_params(function):
    """
    Extract parameter names from the __doc__ string of a cv2 function.
    Returns a list of parameter names.
    """

    doc = function.__doc__
    if not doc:
        return []

    # Use regex to find the function signature
    # The signature is usually in the first or second line
    lines = doc.strip().split("\n")
    signature_line = ""
    for line in lines:
        if line.startswith(function.__name__ + "("):
            signature_line = line
            break

    if not signature_line:
        return []

    # Extract the parameters inside the parentheses
    start_idx = signature_line.find("(") + 1
    end_idx = signature_line.find(")")
    params_str = signature_line[start_idx:end_idx]

    # Remove return annotation if present (e.g., '-> dst')
    params_str = params_str.split("->")[0].strip()

    # Remove square brackets indicating optional parameters
    params_str = params_str.replace("[", "").replace("]", "")

    # Split parameters by commas
    params = [param.strip() for param in params_str.split(",") if param.strip()]

    # Remove default values if present
    params = [param.split("=")[0].strip() for param in params]

    return params


def parameter_combinations(
    param_grid: dict[str, list[Any] | Any]
) -> Generator[dict[str, Any], None, None]:
    """
    Generates and yields all possible parameter combinations from the given parameter_grid.
    Handle cases where parameter values can be lists, single values, or a mix.

    :param param_grid: {'param1': [1, 2, 3],
                        'param2': ['kitten', 'dog'],
                        'param3': 5}
    :return: generator object yielding concrete combinations like
                       {'param1': 1,
                        'param2': 'kitten',
                        'param3': 5}
    """
    # Convert single values to lists for consistent iteration
    for key, value in param_grid.items():
        if not isinstance(value, list):
            param_grid[key] = [value]

    keys = param_grid.keys()
    values = param_grid.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))
