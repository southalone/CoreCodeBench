import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/matcher/hamming_distance_matcher/lhs.tmp'
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
tracker = VariableTracker(start_line=74, end_line=85, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/matcher/hamming_distance_matcher.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher


@pytest.mark.parametrize(
    "rotation_shift, norm_mean",
    [
        pytest.param(-0.5, 0.45),
        pytest.param(1.5, None),
        pytest.param(200, "a"),
        pytest.param(100, -0.2),
        pytest.param(10, 1.3),
    ],
    ids=[
        "rotation_shift should not be negative",
        "rotation_shift should not be floating points",
        "norm_mean should be float",
        "norm_mean should not be negative",
        "norm_mean should not be more than 1",
    ],
)
def test_iris_matcher_raises_an_exception1(
    rotation_shift: int,
    norm_mean: float,
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(rotation_shift=rotation_shift, norm_mean=norm_mean)


@pytest.mark.parametrize(
    "rotation_shift, norm_mean, weights",
    [
        pytest.param(5, 0.4, 3),
        pytest.param(15, None, np.zeros((3, 4))),
        pytest.param(200, 0.45, [("a", 13)]),
    ],
    ids=[
        "weights should be a list of arrays",
        "weights should be a list of arrays",
        "n_rows need to be int or float",
    ],
)
def test_iris_matcher_raises_an_exception2(
    rotation_shift: int,
    norm_mean: float,
    weights: List[np.ndarray],
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(rotation_shift=rotation_shift, norm_mean=norm_mean, weights=weights)


@pytest.mark.parametrize(
    "rotation_shift, norm_mean, norm_gradient, separate_half_matching, weights",
    [
        pytest.param(5, 0.4, "a", False, 3),
        pytest.param(15, None, 0.0005, "b", np.zeros((3, 4))),
    ],
    ids=[
        "norm_gradient should be float",
        "separate_half_matching should be bool",
    ],
)
def test_iris_matcher_raises_an_exception2(
    rotation_shift: int,
    norm_mean: float,
    norm_gradient: float,
    separate_half_matching: bool,
    weights: List[np.ndarray],
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(
            rotation_shift,
            norm_mean=norm_mean,
            norm_gradient=norm_gradient,
            separate_half_matching=separate_half_matching,
            weights=weights,
        )

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)