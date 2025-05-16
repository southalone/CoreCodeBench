import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/normalization/common/lhs.tmp'
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
tracker = VariableTracker(start_line=133, end_line=154, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/normalization/common.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from iris.utils import common

MOCK_MASK_SHAPE = (40, 40)


@pytest.fixture
def expected_result() -> np.ndarray:
    expected_result = np.zeros(shape=MOCK_MASK_SHAPE, dtype=bool)
    expected_result[10:21, 10:21] = True
    return expected_result


@pytest.fixture
def expected_result_line() -> np.ndarray:
    expected_result = np.zeros(shape=MOCK_MASK_SHAPE, dtype=bool)
    expected_result[10, 10:21] = True
    return expected_result


@pytest.mark.parametrize(
    "mock_vertices,expected",
    [
        (np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]), "expected_result"),
        (np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0]]), "expected_result"),
        (np.array([[10.0, 10.0], [20.0, 10.0]]), "expected_result_line"),
        (np.array([[10.0, 10.0], [15.0, 10.0], [20.0, 10.0]]), "expected_result_line"),
    ],
    ids=["standard", "loop", "2 vertices line", "3 vertices line"],
)
def test_contour_to_mask(mock_vertices: np.ndarray, expected: str, request: FixtureRequest) -> None:
    expected_result = request.getfixturevalue(expected)
    result = common.contour_to_mask(mock_vertices, MOCK_MASK_SHAPE)

    assert np.all(expected_result == result)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)