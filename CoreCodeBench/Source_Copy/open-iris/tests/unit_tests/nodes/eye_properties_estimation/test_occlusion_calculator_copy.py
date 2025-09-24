import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/eye_properties_estimation/occlusion_calculator/lhs.tmp'
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
tracker = VariableTracker(start_line=112, end_line=138, target_file='./Source_Copy/open-iris/src/iris/nodes/eye_properties_estimation/occlusion_calculator.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import math

import numpy as np
import pytest

from iris.io.dataclasses import EyeCenters, EyeOrientation, GeometryPolygons, NoiseMask
from iris.nodes.eye_properties_estimation.occlusion_calculator import OcclusionCalculator
from tests.unit_tests.utils import area_of_circular_segment, generate_arc, rotated_asymmetric_rectangle


@pytest.fixture
def algorithm() -> OcclusionCalculator:
    return OcclusionCalculator(quantile_angle=30.0)


@pytest.mark.parametrize(
    "quantile_angle,eye_orientation_angle,expected_result",
    [
        (
            90.0,
            np.radians(10.0),
            generate_arc(1.0, 0.0, 0.0, from_angle=0.0, to_angle=2 * np.pi, num_points=360),
        ),
        (
            30.0,
            np.radians(10.0),
            np.concatenate(
                [
                    generate_arc(1.0, 0.0, 0.0, from_angle=np.radians(0), to_angle=np.radians(40), num_points=40),
                    generate_arc(1.0, 0.0, 0.0, from_angle=np.radians(340), to_angle=np.radians(360), num_points=20),
                    generate_arc(1.0, 0.0, 0.0, from_angle=np.radians(160), to_angle=np.radians(220), num_points=60),
                ]
            ),
        ),
    ],
    ids=["90 degrees", "30 degrees"],
)
def test_get_quantile_points(
    quantile_angle: float,
    eye_orientation_angle: np.float64,
    expected_result: np.ndarray,
) -> None:
    mock_iris_coords = generate_arc(
        radius=1.0, center_x=0.0, center_y=0.0, from_angle=0.0, to_angle=2 * np.pi, num_points=360
    )

    algorithm = OcclusionCalculator(quantile_angle=quantile_angle)

    result_xs, result_ys = algorithm._get_quantile_points(
        mock_iris_coords,
        EyeOrientation(angle=eye_orientation_angle),
        EyeCenters(pupil_x=0.0, pupil_y=0.0, iris_x=0.0, iris_y=0.0),
    )
    result = np.vstack([result_xs, result_ys]).T

    assert np.mean(np.abs(np.sort(result) - np.sort(expected_result))) < 0.5


@pytest.mark.parametrize(
    "quantile_angle,upper_noise_distance,lower_noise_distance,upper_eyelid_distance,lower_eyelid_distance,eye_orientation",
    [
        (90, 200, 200, 200, 200, 0),
        (30, 200, 200, 200, 200, 0),
        (90, 200, 200, 200, 200, np.pi / 4),
        (30, 200, 200, 200, 200, np.pi / 4),
        (90, 100, 200, 200, 200, 0),
        (90, 100, 200, 200, 200, np.pi / 6),
        (30, 200, 100, 200, 200, -np.pi / 6),
        (90, 0, 200, 200, 200, np.pi / 6),
        (90, 100, 100, 200, 200, np.pi / 6),
        (90, 0, 0, 200, 200, -np.pi / 6),
        (30, 0, 0, 200, 200, -np.pi / 6),
        (30, 50, 200, 200, 200, -np.pi / 6),
        (90, 200, 200, 100, 100, -np.pi / 6),
        (30, 200, 200, 0, 100, -np.pi / 6),
        (30, 200, 200, 0, 0, -np.pi / 6),
        (0, 200, 200, 0, 0, -np.pi / 6),
        (45, 80, 10, 60, 50, -np.pi / 2),
    ],
    ids=[
        "occ90 - no occlusion - 0 degrees",
        "occ30 - no occlusion - 0 degrees",
        "occ90 - no occlusion - 45 degrees",
        "occ30 - no occlusion - 45 degrees",
        "occ90 - upper eyelashes half closed - 0 degrees",
        "occ90 - upper eyelashes half closed - 30 degrees",
        "occ30 - lower eyelashes half closed - -30 degrees",
        "occ90 - upper eyelashes closed - 30 degrees",
        "occ90 - both eyelashes half closed",
        "occ90 - eye completely closed (eyelashes)",
        "occ30 - eye completely closed (eyelashes)",
        "occ30 - upper eyelashes half occluded",
        "occ90 - both eyelids half occluded",
        "occ30 - upper eyelid occluded",
        "occ30 - eye completely closed (eyelids)",
        "occ0",
        "occ45 - some eyelash and eyelid occlusion - 90 degrees",
    ],
)
def test_occlusion_calculation(
    quantile_angle: float,
    upper_noise_distance: int,
    lower_noise_distance: int,
    upper_eyelid_distance: int,
    lower_eyelid_distance: int,
    eye_orientation: float,
) -> None:
    """This function tests the occlusion_calculator in an exhaustive number of eye configurations.

    Args:
        quantile_angle (float): quantile of the occlusion, e.g. 90, in degrees.
        upper_noise_distance (int): distance between the center of the iris and the upper eyelashes in pixels.
        lower_noise_distance (int): distance between the center of the iris and the lower eyelashes in pixels.
        upper_eyelid_distance (int): distance between the center of the iris and the upper eyelid in pixels.
        lower_eyelid_distance (int): distance between the center of the iris and the lower eyelid in pixels.
        eye_orientation (float): eye orientation in radians.
    """
    # Extra hardcoded parameters
    img_w, img_h = 1440, 1080
    img_center_x, img_center_y = img_w / 2, img_h / 2
    iris_radius = 200
    pupil_radius = 50

    # Mathematically computing the expected occlusion fraction
    theta_occlusion = 2 * (np.pi / 2 - quantile_angle * 2 * np.pi / 360)
    quantile_area_removed = iris_radius**2 / 2 * (theta_occlusion - np.sin(theta_occlusion))
    area_upper_eyelashes = area_of_circular_segment(iris_radius, upper_noise_distance)
    area_lower_eyelashes = area_of_circular_segment(iris_radius, lower_noise_distance)
    area_upper_eyelid = area_of_circular_segment(iris_radius, upper_eyelid_distance)
    area_lower_eyelid = area_of_circular_segment(iris_radius, lower_eyelid_distance)
    pupil_area_not_included_in_masks = (
        np.pi * pupil_radius**2
        - max(
            area_of_circular_segment(pupil_radius, upper_noise_distance),
            area_of_circular_segment(pupil_radius, upper_eyelid_distance),
        )
        - max(
            area_of_circular_segment(pupil_radius, lower_noise_distance),
            area_of_circular_segment(pupil_radius, lower_eyelid_distance),
        )
    )
    expected_visible_fraction = (
        np.pi * iris_radius**2
        - pupil_area_not_included_in_masks
        - max(quantile_area_removed, area_upper_eyelid, area_upper_eyelashes)
        - max(quantile_area_removed, area_lower_eyelid, area_lower_eyelashes)
    ) / (np.pi * iris_radius**2 - np.pi * pupil_radius**2 - 2 * quantile_area_removed)
    if np.isnan(expected_visible_fraction):
        expected_visible_fraction = 0.0

    # Constructing the mock data
    mock_eye_orientation = EyeOrientation(angle=eye_orientation)
    mock_eye_centers = EyeCenters(pupil_x=img_center_x, pupil_y=img_center_y, iris_x=img_center_x, iris_y=img_center_y)

    mock_pupil = generate_arc(
        radius=pupil_radius,
        center_x=img_center_x,
        center_y=img_center_y,
        from_angle=0.0,
        to_angle=2 * np.pi,
        num_points=360,
    )
    mock_iris = generate_arc(
        radius=iris_radius,
        center_x=img_center_x,
        center_y=img_center_y,
        from_angle=0.0,
        to_angle=2 * np.pi,
        num_points=360,
    )
    mock_eyeball = rotated_asymmetric_rectangle(
        img_center_x, img_center_y, 1.5 * iris_radius, upper_eyelid_distance, lower_eyelid_distance, eye_orientation
    )
    mock_extrapolated_polygons = GeometryPolygons(
        pupil_array=mock_pupil, iris_array=mock_iris, eyeball_array=mock_eyeball
    )

    x, y = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))
    x, y = x.flatten(), y.flatten()
    # (y > Ax + b) or (y < Ax + b')  with A = tan(eye_orientation)
    z = np.logical_or(
        y
        > np.tan(eye_orientation) * x
        + upper_noise_distance / np.cos(eye_orientation)
        + img_center_y
        - img_center_x * np.tan(eye_orientation),
        y
        < np.tan(eye_orientation) * x
        - lower_noise_distance / np.cos(eye_orientation)
        + img_center_y
        - img_center_x * np.tan(eye_orientation),
    )
    mock_noise_mask = NoiseMask(mask=z.reshape(img_h, img_w))

    algorithm = OcclusionCalculator(quantile_angle=quantile_angle)
    result = algorithm(mock_extrapolated_polygons, mock_noise_mask, mock_eye_orientation, mock_eye_centers)

    assert math.isclose(result.visible_fraction, expected_visible_fraction, abs_tol=0.01)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)