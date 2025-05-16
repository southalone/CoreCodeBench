import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/normalization/linear_normalization/lhs.tmp'
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
tracker = VariableTracker(start_line=64, end_line=77, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/normalization/linear_normalization.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import NormalizationError
from iris.nodes.normalization.linear_normalization import LinearNormalization
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "wrong_param",
    [
        ({"res_in_r": -1}),
        ({"res_in_r": 0}),
    ],
)
def test_constructor_raises_exception(wrong_param: dict) -> None:
    with pytest.raises((NormalizationError, ValidationError)):
        _ = LinearNormalization(**wrong_param)


@pytest.mark.parametrize(
    "pupil_points, iris_points, expected_correspondences",
    [
        (
            generate_arc(3.0, 5.0, 5.0, 0.0, 2 * np.pi, 3),
            generate_arc(10.0, 4.8, 5.1, 0.0, 2 * np.pi, 3),
            np.array(
                [
                    [[8, 5], [4, 8], [3, 2]],
                    [[15, 5], [0, 14], [0, -4]],
                ]
            ),
        ),
        (
            generate_arc(50.0, 0.0, 0.0, 0.0, 2 * np.pi, 8),
            generate_arc(100.0, 0.0, 0.0, 0.0, 2 * np.pi, 8),
            np.array(
                [
                    [[50, 0], [35, 35], [0, 50], [-35, 35], [-50, 0], [-35, -35], [0, -50], [35, -35]],
                    [[100, 0], [71, 71], [0, 100], [-71, 71], [-100, 0], [-71, -71], [0, -100], [71, -71]],
                ]
            ),
        ),
    ],
    ids=[
        "test1",
        "test2",
    ],
)
def test_generate_correspondences(
    pupil_points: np.ndarray, iris_points: np.ndarray, expected_correspondences: np.ndarray
) -> None:
    algorithm = LinearNormalization(
        res_in_r=2,
    )
    result = algorithm._generate_correspondences(
        pupil_points=pupil_points,
        iris_points=iris_points,
    )

    np.testing.assert_allclose(result, expected_correspondences, rtol=1e-05)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)