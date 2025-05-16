import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/eye_properties_estimation/sharpness_estimation/lhs.tmp'
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
tracker = VariableTracker(start_line=62, end_line=66, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/eye_properties_estimation/sharpness_estimation.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from typing import Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.eye_properties_estimation.sharpness_estimation import SharpnessEstimation


@pytest.mark.parametrize(
    "lap_ksize",
    [
        pytest.param(0),
        pytest.param("a"),
        pytest.param(-10),
        pytest.param(33),
        pytest.param(2),
        pytest.param(np.ones(3)),
    ],
    ids=[
        "lap_ksize should be larger than zero",
        "lap_ksize should be int",
        "lap_ksize should not be negative",
        "lap_ksize should not be larger than 31",
        "lap_ksize should be odd number",
        "lap_ksize should not be array",
    ],
)
def test_sharpness_lap_ksize_raises_an_exception(lap_ksize: int) -> None:
    with pytest.raises(ValidationError):
        _ = SharpnessEstimation(lap_ksize=lap_ksize)


@pytest.mark.parametrize(
    "erosion_ksize",
    [
        pytest.param((0, 5)),
        pytest.param((1, "a")),
        pytest.param((-10, 3)),
        pytest.param((30, 5)),
        pytest.param(np.ones(3)),
    ],
    ids=[
        "erosion_ksize should all be larger than zero",
        "erosion_ksize should all be int",
        "erosion_ksize should not be negative",
        "erosion_ksize should be odd number",
        "erosion_ksize should be a tuple of integer with length 2",
    ],
)
def test_sharpness_erosion_ksize_raises_an_exception(erosion_ksize: Tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        _ = SharpnessEstimation(erosion_ksize=erosion_ksize)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)