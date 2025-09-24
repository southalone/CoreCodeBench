import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/utils/base64_encoding/lhs.tmp'
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
tracker = VariableTracker(start_line=31, end_line=36, target_file='./Source_Copy/open-iris/src/iris/utils/base64_encoding.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest

from iris.utils import base64_encoding as be


@pytest.mark.parametrize("mock_shape", [(3, 10, 100), (10, 3, 100), (100, 10, 3)])
def test_base64_array_encode_decode(mock_shape: tuple) -> None:
    mock_array = np.random.choice(2, size=mock_shape).astype(bool)

    result = be.base64_decode_array(be.base64_encode_array(mock_array), array_shape=mock_shape)

    np.testing.assert_equal(result, mock_array)


@pytest.mark.parametrize(
    "plain_str,base64_str", [("test", "dGVzdA=="), ("un:\n  - deux\n  - trois", "dW46CiAgLSBkZXV4CiAgLSB0cm9pcw==")]
)
def test_base64_str_encode_decode(plain_str: str, base64_str: str) -> None:
    # Test base64_encode_str
    encoded_str = be.base64_encode_str(plain_str)
    assert encoded_str == base64_str
    assert isinstance(encoded_str, str)

    # Test base64_decode_str
    decoded_str = be.base64_decode_str(base64_str)
    assert decoded_str == plain_str
    assert isinstance(decoded_str, str)

    # Test that encoding and decoding convolve
    encoded_decoded_str = be.base64_decode_str(be.base64_encode_str(plain_str))
    assert encoded_decoded_str == plain_str

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)