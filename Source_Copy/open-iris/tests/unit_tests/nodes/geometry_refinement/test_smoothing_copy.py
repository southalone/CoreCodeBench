import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/geometry_refinement/smoothing/lhs.tmp'
class VariableTracker:
    def __init__(self, start_line, end_line, target_file):
        self.start_line = start_line
        self.end_line = end_line
        self.target_file = target_file
        self.previous_locals = {}
        
        with open(template_file, 'w') as f:
            f.write('')

    def safe_compare(self, var_name, current_value, previous_value):
        """Safely compare two values, returning True if they differ."""
        try:
            # 首先检查内存地址
            if id(current_value) == id(previous_value):
                return False
            
            # 使用 DeepDiff 进行深度比较
            diff = DeepDiff(current_value, previous_value, ignore_order=True)
            
            # 如果有差异，返回 True
            return bool(diff)
        except Exception as e:
            print(f"Comparison failed for variable '{var_name}': {e}")
            return True

    def trace_func(self, frame, event, arg):
        if (event == "line" and
            frame.f_code.co_filename.endswith(self.target_file) and
            self.start_line <= frame.f_lineno <= self.end_line):
            
            current_locals = copy.deepcopy(frame.f_locals)

            # 初始化时记录所有变量
            if not self.previous_locals:
                self.previous_locals = current_locals

            # 检查哪些变量发生了变化
            changed_vars = {
                var: current_locals[var]
                for var in current_locals
                if (var not in self.previous_locals or
                    self.safe_compare(var, current_locals[var], self.previous_locals[var]))
            }

            # 更新之前的局部变量状态
            self.previous_locals = current_locals

            if changed_vars:
                print('line: ', frame.f_lineno, 'changed_vars: ', changed_vars)
                with open(template_file, 'a') as f:
                    f.write('\n'.join(list(changed_vars.keys())))
                    f.write('\n')

        return self.trace_func

    def track_variable_changes(self, func):
        def wrapper(*args, **kwargs):
            sys.settrace(self.trace_func)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
        return wrapper

# 设置行号范围
tracker = VariableTracker(start_line=248, end_line=252, target_file='./Source_Copy/open-iris/src/iris/nodes/geometry_refinement/smoothing.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from typing import Tuple

import numpy as np
import pytest

from iris.io.errors import GeometryRefinementError
from iris.nodes.geometry_refinement.smoothing import Smoothing
from tests.unit_tests.utils import generate_arc


@pytest.fixture
def algorithm() -> Smoothing:
    return Smoothing(dphi=1, kernel_size=10)


@pytest.mark.parametrize(
    "arc,expected_num_gaps",
    [
        (generate_arc(10, 0.0, 0.0, 0.0, 2 * np.pi), 0),
        (generate_arc(10, 0.0, 0.0, 0.0, np.pi), 1),
        (generate_arc(10, 0.0, 0.0, np.pi, 2.5 * np.pi), 1),
        (np.vstack([generate_arc(10, 0.0, 0.0, 0.0, np.pi / 4), generate_arc(10, 0.0, 0.0, np.pi, 4 / 3 * np.pi)]), 2),
        (
            np.vstack(
                [generate_arc(10, 0.0, 0.0, 0.0, np.pi / 4), generate_arc(10, 0.0, 0.0, 4 / 3 * np.pi, 2 * np.pi)]
            ),
            1,
        ),
    ],
)
def test_cut_into_arcs(algorithm: Smoothing, arc: np.ndarray, expected_num_gaps: int) -> None:
    center_x, center_y = 0.0, 0.0

    _, result_num_gaps = algorithm._cut_into_arcs(arc, (center_x, center_y))

    assert result_num_gaps == expected_num_gaps


@pytest.mark.parametrize(
    "phis, rhos, expected_result",
    [
        (
            np.array([0.0, 0.02621434, 0.05279587, 0.08517275, 0.12059719, 0.15643903]),
            np.array([36.89178243, 36.62426603, 36.38227748, 37.14610941, 36.90603523, 36.71284955]),
            (
                np.array(
                    [
                        0.0,
                        0.01745329,
                        0.03490659,
                        0.05235988,
                        0.06981317,
                        0.08726646,
                        0.10471976,
                        0.12217305,
                        0.13962634,
                    ]
                ),
                np.array(
                    [
                        36.80346909,
                        36.89178243,
                        36.89178243,
                        36.80346909,
                        36.80346909,
                        36.80346909,
                        36.78374777,
                        36.78374777,
                        36.78374777,
                    ]
                ),
            ),
        )
    ],
)
def test_smooth_array(
    algorithm: Smoothing, phis: np.ndarray, rhos: np.ndarray, expected_result: Tuple[np.ndarray, np.ndarray]
) -> None:
    result = algorithm._smooth_array(phis, rhos)
    np.testing.assert_almost_equal(expected_result, result, decimal=0)


def test_smooth_arc(algorithm: Smoothing) -> None:
    center_x, center_y = 0.0, 0.0

    mock_arc = generate_arc(10, center_x, center_y, 0.0, np.pi, 180)
    expected_result = mock_arc[algorithm.kernel_offset : -algorithm.kernel_offset]

    result = algorithm._smooth_arc(mock_arc, (center_x, center_y))

    np.testing.assert_almost_equal(expected_result, result, decimal=0)


def test_smooth_circular_shape(algorithm: Smoothing) -> None:
    center_x, center_y = 0.0, 0.0

    mock_arc = generate_arc(10, center_x, center_y, 0.0, 2 * np.pi, 1000)
    expected_result = generate_arc(10, center_x, center_y, 0.0, 2 * np.pi, 360)

    result = algorithm._smooth_circular_shape(mock_arc, (center_x, center_y))

    np.testing.assert_almost_equal(expected_result, result, decimal=0)


def test_sort_two_arrays(algorithm: Smoothing) -> None:
    first_array = np.array([3.0, 2.0, 1.0])
    second_array = np.array([1.0, 2.0, 3.0])

    first_sorted, second_sorted = algorithm._sort_two_arrays(first_array, second_array)

    np.testing.assert_equal(first_sorted, second_array)
    np.testing.assert_equal(second_sorted, first_array)


def test_rolling_median(algorithm: Smoothing) -> None:
    signal = np.arange(0, 10, 1)
    kernel_offset = 3

    expected_result = np.array([3.0, 4.0, 5.0, 6.0])

    result = algorithm._rolling_median(signal, kernel_offset)

    np.testing.assert_equal(result, expected_result)


def test_rolling_median_raises_an_error_when_not_1D_signal(algorithm: Smoothing) -> None:
    signal = np.arange(0, 10, 1).reshape((5, 2))
    kernel_offset = 3

    with pytest.raises(GeometryRefinementError):
        _ = algorithm._rolling_median(signal, kernel_offset)


def test_find_start_index_raises_an_error_when_phis_not_sorted_ascendingly(algorithm: Smoothing) -> None:
    mock_phis = np.arange(0, 100, 1)
    np.random.shuffle(mock_phis)

    with pytest.raises(GeometryRefinementError):
        _ = algorithm._find_start_index(mock_phis)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)