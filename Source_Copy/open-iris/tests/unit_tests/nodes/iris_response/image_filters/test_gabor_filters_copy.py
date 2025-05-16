import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/testcases/open-iris/src/iris/nodes/iris_response/image_filters/gabor_filters/lhs.tmp'
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
tracker = VariableTracker(start_line=221, end_line=239, target_file='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/open-iris/src/iris/nodes/iris_response/image_filters/gabor_filters.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import math
from typing import Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import ImageFilterError
from iris.nodes.iris_response.image_filters import gabor_filters


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_phi, dc_correction",
    [
        ((19, 21), 2, 4, 45, 10, True),
        ((11, 15), 1.1, 1.5, 0, 3, True),
        ((31, 31), 1, 1, 240.7, 2, True),
    ],
    ids=[
        "regular1",
        "regular2",
        "regular3",
    ],
)
def test_gabor_filter_constructor(
    kernel_size: Tuple[int, int],
    sigma_phi: float,
    sigma_rho: float,
    theta_degrees: float,
    lambda_phi: float,
    dc_correction: bool,
) -> None:
    g_filter = gabor_filters.GaborFilter(
        kernel_size=kernel_size,
        sigma_phi=sigma_phi,
        sigma_rho=sigma_rho,
        theta_degrees=theta_degrees,
        lambda_phi=lambda_phi,
        dc_correction=dc_correction,
    )

    assert np.max(g_filter.kernel_values.real) > np.min(g_filter.kernel_values.real)
    assert np.max(g_filter.kernel_values.imag) > np.min(g_filter.kernel_values.imag)
    assert g_filter.kernel_values.shape[0] == kernel_size[1]
    assert g_filter.kernel_values.shape[1] == kernel_size[0]

    # Gabor filter values are complex numbers
    assert np.iscomplexobj(g_filter.kernel_values)

    # zero DC component
    assert math.isclose(np.mean(g_filter.kernel_values.real), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.mean(g_filter.kernel_values.imag), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.linalg.norm(g_filter.kernel_values.real, ord="fro"), 1.0, rel_tol=1e-03, abs_tol=1e-03)


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_phi, dc_correction",
    [
        (11, 2, 4, 45, 10, True),
        ((20, 21), 2, 4, 45, 10, True),
        ((4.5, 9.78), 2, 4, 45, 10, True),
        (("r", "100"), 2, 4, 45, 10, True),
        ((-1, 0), 2, 4, 45, 10, True),
        ((11, 100), 2, 4, 45, 10, True),
        ((1, 2), 2, 4, 45, 10, True),
        ((11, 15), -2, 4, 45, 10, True),
        ((15, 11), 0, 4, 45, 10, True),
        ((31, 37), 32, 1e-03, 0, 10, True),
        ((11, 15), 3, 0, 45, 10, True),
        ((15, 11), 3, -0.2, 45, 10, True),
        ((31, 21), 3, 25, 0, 10, True),
        ((31, 21), 3, 5, -5, 10, True),
        ((31, 21), 3, 5, 360, 10, True),
        ((31, 21), 3, 5, 30, 1e-03, True),
        ((31, 21), 3, 5, 30, -5, True),
        ((31, 21), 3, 5, 30, 2, "a"),
        ((31, 21), 3, 5, 30, 2, 0.1),
    ],
    ids=[
        "kernel_size is not a single number",
        "kernel_size not odd numbers",
        "kernel_size not integers1",
        "kernel_size not integers2",
        "kernel_size not positive integers",
        "kernel_size size larger than 99",
        "kernel_size size less than 3",
        "sigma_phi not positive interger1",
        "sigma_phi not positive interger2",
        "sigma_phi bigger than kernel_size[0]",
        "sigma_rho not positive interger1",
        "sigma_rho not positive interger2",
        "sigma_rho bigger than kernel_size[1]",
        "theta_degrees is not higher than/equal to 0",
        "theta_degrees is not lower than 360",
        "lambda_phi is not larger than/equal to 2",
        "lambda_phi is not positive",
        "dc_correction is not of boolean type",
        "dc_correction is not of boolean type again",
    ],
)
def test_gabor_filter_constructor_raises_an_exception(
    kernel_size: Tuple[int, int],
    sigma_phi: float,
    sigma_rho: float,
    theta_degrees: float,
    lambda_phi: float,
    dc_correction: bool,
) -> None:
    with pytest.raises((ValidationError, ImageFilterError)):
        _ = gabor_filters.GaborFilter(
            kernel_size=kernel_size,
            sigma_phi=sigma_phi,
            sigma_rho=sigma_rho,
            theta_degrees=theta_degrees,
            lambda_phi=lambda_phi,
            dc_correction=dc_correction,
        )


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_rho",
    [
        ((19, 21), np.pi / np.sqrt(2) / 2, 1, 45, 3),
        ((11, 15), 1.1, 0.5, 0, 3),
    ],
    ids=[
        "regular1",
        "regular2",
    ],
)
def test_log_gabor_filter_constructor(
    kernel_size: Tuple[int, int], sigma_phi: float, sigma_rho: float, theta_degrees: float, lambda_rho: float
) -> None:
    logg_filter = gabor_filters.LogGaborFilter(
        kernel_size=kernel_size,
        sigma_phi=sigma_phi,
        sigma_rho=sigma_rho,
        theta_degrees=theta_degrees,
        lambda_rho=lambda_rho,
    )

    assert np.max(logg_filter.kernel_values.real) > np.min(logg_filter.kernel_values.real)
    assert np.max(logg_filter.kernel_values.imag) > np.min(logg_filter.kernel_values.imag)
    assert logg_filter.kernel_values.shape[0] == kernel_size[1]
    assert logg_filter.kernel_values.shape[1] == kernel_size[0]

    # LogGabor filter values are complex numbers
    assert np.iscomplexobj(logg_filter.kernel_values)

    # zero DC component
    assert math.isclose(np.mean(logg_filter.kernel_values.real), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.mean(logg_filter.kernel_values.imag), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.linalg.norm(logg_filter.kernel_values.real, ord="fro"), 1.0, rel_tol=1e-03, abs_tol=1e-03)


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_rho",
    [
        (11, np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((20, 21), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((4.5, 9.78), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        (("r", "100"), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((-1, 0), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((11, 100), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((1, 2), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((11, 15), -2, 4, 45, 10),
        ((15, 11), 0, 4, 45, 10),
        ((31, 37), 2 * np.pi, 45, 0, 10),
        ((11, 15), 0.8, 0.05, 45, 10),
        ((15, 11), 0.8, -0.2, 45, 10),
        ((31, 21), 0.8, 1.1, 0, 10),
        ((31, 21), 0.8, 0.5, -5, 10),
        ((31, 21), 0.8, 0.5, 360, 10),
        ((31, 21), 0.8, 0.5, 30, 1e-03),
        ((31, 21), 0.8, 0.5, 30, -5),
    ],
    ids=[
        "kernel_size is not a single number",
        "kernel_size not odd numbers",
        "kernel_size not integers1",
        "kernel_size not integers2",
        "kernel_size not positive integers",
        "kernel_size size larger than 99",
        "kernel_size size less than 3",
        "sigma_phi not positive interger1",
        "sigma_phi not positive interger2",
        "sigma_phi bigger than np.pi",
        "sigma_rho not positive interger1",
        "sigma_rho not positive interger2",
        "sigma_rho bigger than 1",
        "theta_degrees is not higher than/equal to 0",
        "theta_degrees is not lower than 360",
        "lambda_phi is not larger than/equal to 2",
        "lambda_phi is not positive",
    ],
)
def test_log_gabor_filter_constructor_raises_an_exception(
    kernel_size: Tuple[int, int], sigma_phi: float, sigma_rho: float, theta_degrees: float, lambda_rho: float
) -> None:
    with pytest.raises((ValidationError, ImageFilterError)):
        _ = gabor_filters.LogGaborFilter(
            kernel_size=kernel_size,
            sigma_phi=sigma_phi,
            sigma_rho=sigma_rho,
            theta_degrees=theta_degrees,
            lambda_rho=lambda_rho,
        )

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)