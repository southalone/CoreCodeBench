import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/normalization/perspective_normalization/lhs.tmp'
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
tracker = VariableTracker(start_line=213, end_line=239, target_file='./Source_Copy/open-iris/src/iris/nodes/normalization/perspective_normalization.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import NormalizationError
from iris.nodes.normalization.perspective_normalization import PerspectiveNormalization


@pytest.fixture
def algorithm() -> PerspectiveNormalization:
    return PerspectiveNormalization(
        res_in_phi=400,
        res_in_r=100,
        skip_boundary_points=10,
        intermediate_radiuses=np.linspace(0.0, 1.0, 10),
    )


@pytest.mark.parametrize(
    "wrong_param",
    [
        ({"res_in_phi": 0, "res_in_r": 10}),
        ({"res_in_phi": -1, "res_in_r": 10}),
        ({"res_in_phi": 10, "res_in_r": 0}),
        ({"res_in_phi": 10, "res_in_r": -1}),
        ({"skip_boundary_points": -1}),
        ({"intermediate_radiuses": []}),
        ({"intermediate_radiuses": [0.0]}),
        ({"intermediate_radiuses": [-1.0, 0.0, 1.0]}),
        ({"intermediate_radiuses": [0, 0.2, 1.2]}),
    ],
)
def test_constructor_raises_exception(wrong_param: dict) -> None:
    with pytest.raises((NormalizationError, ValidationError)):
        _ = PerspectiveNormalization(**wrong_param)


def test_bbox_coords(algorithm: PerspectiveNormalization) -> None:
    # fmt: off
    norm_dst_points = np.array([
        [0, 0],
        [11, 0],
        [0, 11],
        [11, 11],
    ])
    # fmt: on

    expected_result = (0, 0, 11, 11)

    result = algorithm._bbox_coords(norm_dst_points)

    assert result == expected_result


def test_correspondence_rois_coords(algorithm: PerspectiveNormalization) -> None:
    angle_idx = 0
    ring_idx = 0
    # fmt: off
    # Nones shouldn't be taken
    src_points = np.array([
        [[0.0, 1.0], [1.0, 2.0], [None, None]],
        [[1.0, 1.0], [2.0, 1.0], [None, None]],
        [[None, None], [None, None], [None, None]],
    ])
    dst_points = np.array([
        [[11.0, 11.0], [12.0, 21.0], [None, None]],
        [[22.0, 21.0], [22.0, 22.0], [None, None]],
        [[None, None], [None, None], [None, None]],
    ])

    expected_src_roi = np.array([
        [0.0, 1.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0]
    ])
    expected_dst_roi = np.array([
        [11.0, 11.0], [12.0, 21.0], [22.0, 21.0], [22.0, 22.0],
    ])
    # fmt: on

    result_src_roi, result_dst_roi = algorithm._correspondence_rois_coords(
        angle_idx=angle_idx,
        ring_idx=ring_idx,
        src_points=src_points,
        dst_points=dst_points,
    )

    assert np.all(result_src_roi == expected_src_roi)
    assert np.all(result_dst_roi == expected_dst_roi)


def test_cartesian2homogeneous() -> None:
    cartesian_xs = np.array([1.0, 2.0, 3.0])
    cartesian_ys = np.array([10.0, 20.0, 30.0])

    cartesian_pts = np.array([cartesian_xs, cartesian_ys])
    # fmt: off
    expected_homogeneous_pts = np.array([
        [1.0, 2.0, 3.0],        # xs
        [10.0, 20.0, 30.0],     # ys
        [1.0, 1.0, 1.0]         # ks
    ])
    # fmt: on

    result = PerspectiveNormalization.cartesian2homogeneous(cartesian_pts)

    assert np.all(result == expected_homogeneous_pts)


def test_homogeneous2cartesian() -> None:
    # fmt: off
    homogeneous_pts = np.array([
        [1.0, 2.0, 3.0],        # xs
        [10.0, 20.0, 30.0],     # ys
        [1.0, 2.0, 3.0]         # ks
    ])
    expected_cartesian_pts = np.array([
        [1.0, 1.0, 1.0],        # xs
        [10.0, 10.0, 10.0],     # ys
    ])
    # fmt: on

    result = PerspectiveNormalization.homogeneous2cartesian(homogeneous_pts)

    assert np.all(result == expected_cartesian_pts)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)