import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/iris_response_refinement/fragile_bits_refinement/lhs.tmp'
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
tracker = VariableTracker(start_line=64, end_line=111, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/iris_response_refinement/fragile_bits_refinement.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from typing import Tuple

import pytest
from pydantic import ValidationError, confloat

from iris.nodes.iris_response_refinement.fragile_bits_refinement import FragileBitRefinement, FragileType


@pytest.mark.parametrize(
    # given
    "value_threshold, fragile_type",
    [
        pytest.param([-0.6, 0, -0.3], FragileType.cartesian),
        pytest.param([-0.2, -0.5, 1], FragileType.polar),
        pytest.param([0, 0, 1], "elliptical"),
    ],
    ids=["error_theshold_cartesian", "error_theshold_polar", "error_fragile_type"],
)
def test_iris_encoder_threshold_raises_an_exception(
    value_threshold: Tuple[confloat(ge=0), confloat(ge=0), confloat(ge=0)],
    fragile_type: FragileType,
) -> None:
    # when
    with pytest.raises((ValidationError)):
        _ = FragileBitRefinement(value_threshold, fragile_type)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)