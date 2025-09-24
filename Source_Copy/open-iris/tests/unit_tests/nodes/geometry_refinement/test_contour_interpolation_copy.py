import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/geometry_refinement/contour_interpolation/lhs.tmp'
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
tracker = VariableTracker(start_line=63, end_line=73, target_file='./Source_Copy/open-iris/src/iris/nodes/geometry_refinement/contour_interpolation.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.geometry_refinement.contour_interpolation import ContourInterpolation


@pytest.fixture
def algorithm() -> ContourInterpolation:
    return ContourInterpolation(max_distance_between_boundary_points=0.01)


def test_constructor() -> None:
    mock_max_distance_between_boundary_points = 0.01

    _ = ContourInterpolation(mock_max_distance_between_boundary_points)


@pytest.mark.parametrize(
    "max_distance_between_boundary_points",
    [(-1.0), (0.0)],
    ids=[
        "wrong max_distance_between_boundary_points < 0",
        "wrong max_distance_between_boundary_points = 0",
    ],
)
def test_constructor_raises_an_exception(max_distance_between_boundary_points: float) -> None:
    with pytest.raises(ValidationError):
        _ = ContourInterpolation(max_distance_between_boundary_points)


@pytest.mark.parametrize(
    "mock_polygon,mock_distance_between_points,expected_result",
    [
        (
            np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]], dtype=np.int32),
            25.0,
            np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [100.0, 0.0]]),
        ),
        (
            np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]], dtype=np.int32),
            50.0,
            np.array(
                [[0.0, 0.0], [0.0, 50.0], [0.0, 100.0], [50.0, 100.0], [100.0, 100.0], [100.0, 50.0], [100.0, 0.0]]
            ),
        ),
        (
            np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 15.0]], dtype=np.int32),
            7.0,
            np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0], [0.0, 15.0]]),
        ),
    ],
    ids=["along line", "complex polygon", "not uniform distance"],
)
def test_interpolate_contour_points(
    algorithm: ContourInterpolation,
    mock_polygon: np.ndarray,
    mock_distance_between_points: float,
    expected_result: np.ndarray,
) -> None:
    result = algorithm._interpolate_polygon_points(
        polygon=mock_polygon, max_distance_between_points_px=mock_distance_between_points
    )

    for point in result:
        assert point in expected_result

    for point in expected_result:
        assert point in result

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)