import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/matcher/utils/lhs.tmp'
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
tracker = VariableTracker(start_line=180, end_line=215, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/matcher/utils.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from typing import Tuple

import numpy as np
import pytest

from iris.nodes.normalization.common import correct_orientation, interpolate_pixel_intensity, to_uint8
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "eye_orientation,pupil_points,iris_points,expected_pupil_points,expected_iris_points",
    [
        (
            -1.0,
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360),
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360), 1, axis=0),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360), 1, axis=0),
        ),
        (
            -1.0,
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720),
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720), 2, axis=0),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720), 2, axis=0),
        ),
    ],
    ids=["1 point rotation", "2 points rotation"],
)
def test_correct_orientation(
    eye_orientation: float,
    pupil_points: np.ndarray,
    iris_points: np.ndarray,
    expected_pupil_points: np.ndarray,
    expected_iris_points: np.ndarray,
) -> None:
    result_pupil_points, result_iris_points = correct_orientation(
        pupil_points=pupil_points,
        iris_points=iris_points,
        eye_orientation=np.radians(eye_orientation),
    )

    assert np.all(result_pupil_points == expected_pupil_points)
    assert np.all(result_iris_points == expected_iris_points)


@pytest.mark.parametrize(
    "pixel_coords,expected_intensity",
    [
        # Corners
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 3.0),
        ((0.0, 2.0), 6.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 4.0),
        ((1.0, 2.0), 7.0),
        ((2.0, 0.0), 2.0),
        ((2.0, 1.0), 5.0),
        ((2.0, 2.0), 8.0),
        # Inside
        ((0.5, 0.5), 2),
        ((0.5, 1.5), 5),
        ((1.5, 0.5), 3),
        # Outside
        ((10.0, 0.5), 0.0),
        ((0.5, 10.0), 0.0),
        ((10.0, 10.0), 0.0),
    ],
)
def test_interpolate_pixel_intensity(pixel_coords: Tuple[float, float], expected_intensity: float) -> None:
    # fmt: off
    test_image = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    # fmt: on

    result = interpolate_pixel_intensity(image=test_image, pixel_coords=pixel_coords)

    assert result == expected_intensity


@pytest.mark.parametrize(
    "input_img",
    [
        (np.ones(shape=(10, 10), dtype=np.uint8)),
        (np.zeros(shape=(10, 10), dtype=np.uint8)),
        (np.random.randn(100).reshape((10, 10))),
    ],
)
def test_to_uint8(input_img: np.ndarray) -> None:
    result = to_uint8(input_img)

    assert result.dtype == np.uint8
    assert np.all(result >= 0) and np.all(result <= 255)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)