import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/eye_properties_estimation/moment_of_area/lhs.tmp'
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
tracker = VariableTracker(start_line=55, end_line=65, target_file='./Source_Copy/open-iris/src/iris/nodes/eye_properties_estimation/moment_of_area.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest

from iris.io.dataclasses import GeometryPolygons
from iris.io.errors import EyeOrientationEstimationError
from iris.nodes.eye_properties_estimation.moment_of_area import MomentOfArea
from tests.unit_tests.utils import rotated_elliptical_contour


@pytest.mark.parametrize(
    "input_contour,eccentricity_threshold",
    [(rotated_elliptical_contour(a=1, b=1, theta=0), 0.5)],
    ids=["eccentricity < threshold"],
)
def test_first_order_area_fail_eccentricity_threhsold(input_contour: np.ndarray, eccentricity_threshold: float) -> None:
    triangle = np.array([[0, 0], [0, 1], [1, 0]])
    input_geometry_polygon = GeometryPolygons(pupil_array=triangle, iris_array=triangle, eyeball_array=input_contour)

    with pytest.raises(EyeOrientationEstimationError):
        moments_of_area = MomentOfArea(eccentricity_threshold=eccentricity_threshold)
        moments_of_area(input_geometry_polygon)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)