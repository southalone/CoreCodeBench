import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/io/class_configs/lhs.tmp'
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
tracker = VariableTracker(start_line=75, end_line=81, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/io/class_configs.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from typing import Any, Dict

import pytest
from pydantic import Field, ValidationError

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm, ImmutableModel


class ConcreteImmutableModel(ImmutableModel):
    """A concrete implementation of ImmutableModel with parameters"""

    my_param_1: int = Field(..., gt=0)
    my_param_2: str


@pytest.mark.parametrize(
    "parameters",
    [
        ({"my_param_1": 3, "my_param_2": "toto"}),
        ({"my_param_1": 3, "my_param_2": "3.7"}),
    ],
)
def test_immutable_model_constructor(parameters: Dict[str, Any]) -> None:
    cim = ConcreteImmutableModel(**parameters)

    for key, value in parameters.items():
        assert getattr(cim, key) == value


@pytest.mark.parametrize(
    "parameters",
    [
        ({"my_param_1": -4, "my_param_2": "toto"}),
        ({"my_param_1": 3, "my_param_2": "toto", "extra_parameter": "forbidden"}),
    ],
    ids=["pydantic checks", "extra parameter forbidden"],
)
def test_immutable_model_constructor_raises_exception(parameters: Dict) -> None:
    with pytest.raises((ValidationError, TypeError)):
        _ = ConcreteImmutableModel(**parameters)


@pytest.mark.parametrize(
    "parameters,new_parameters",
    [
        pytest.param(
            {"my_param_1": 3, "my_param_2": "toto"},
            {"my_param_1": 6, "my_param_2": "not toto"},
        ),
    ],
    ids=["regular"],
)
def test_immutability_of_immutable_model(parameters: Dict[str, Any], new_parameters: Dict[str, Any]) -> None:
    immutable_obj = ConcreteImmutableModel(**parameters)

    with pytest.raises(TypeError):
        for key, value in new_parameters.items():
            setattr(immutable_obj, key, value)


class MockDummyValidationAlgorithm(Callback):
    CORRECT_MSG = "Worldcoin AI is the best"
    ERROR_MSG = "Incorrect msg returned!"

    def on_execute_end(self, result: str) -> None:
        if result != self.CORRECT_MSG:
            raise RuntimeError(MockDummyValidationAlgorithm.ERROR_MSG)


class MockParametrizedModelWithCallback(Algorithm):
    class Parameters(Algorithm.Parameters):
        ret_msg: str

    __parameters_type__ = Parameters

    def __init__(self, ret_msg: str = "Worldcoin AI is the best") -> None:
        super().__init__(ret_msg=ret_msg, callbacks=[MockDummyValidationAlgorithm()])

    def run(self) -> str:
        return self.params.ret_msg


def test_parametrized_model_validation_hook_not_raising_an_error() -> None:
    mock_model = MockParametrizedModelWithCallback()

    result = mock_model.execute()

    assert result == mock_model.params.ret_msg


def test_parametrized_model_validation_hook_raising_an_error() -> None:
    mock_model = MockParametrizedModelWithCallback(ret_msg="Worldcoin AI isn't the best")

    with pytest.raises(RuntimeError) as err:
        _ = mock_model.execute()

    assert str(err.value) == MockDummyValidationAlgorithm.ERROR_MSG

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)