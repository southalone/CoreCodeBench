import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/validators/object_validators/lhs.tmp'
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
tracker = VariableTracker(start_line=254, end_line=264, target_file='./Source_Copy/open-iris/src/iris/nodes/validators/object_validators.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

import numpy as np
import pytest

import iris.io.errors as E
import iris.nodes.validators.object_validators as obj_v
from iris.io.dataclasses import EyeOcclusion, GeometryPolygons, IrisTemplate, Offgaze, PupilToIrisProperty, Sharpness
from tests.unit_tests.utils import generate_arc, generate_multiple_arcs


@pytest.fixture
def mock_polygons_arcs_of_circle() -> GeometryPolygons:
    return GeometryPolygons(
        pupil_array=generate_arc(100, 500, 500, -np.pi / 4, -3 * np.pi / 4, num_points=25000),
        iris_array=generate_multiple_arcs(
            [
                {
                    "radius": 300,
                    "center_x": 500,
                    "center_y": 500,
                    "from_angle": np.pi / 4,
                    "to_angle": -np.pi / 4,
                    "num_points": 25000,
                },
                {
                    "radius": 300,
                    "center_x": 500,
                    "center_y": 500,
                    "from_angle": -3 * np.pi / 4,
                    "to_angle": -5 * np.pi / 4,
                    "num_points": 25000,
                },
            ]
        ),
        eyeball_array=generate_arc(500, 500, 500, 0, 2 * np.pi, num_points=25000),
    )


@pytest.mark.parametrize(
    "p2i_property",
    [
        PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.3, pupil_to_iris_center_dist_ratio=0.1),
        PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.2, pupil_to_iris_center_dist_ratio=0.5),
        PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.5, pupil_to_iris_center_dist_ratio=0.0),
    ],
    ids=["simple", "edge case: left boundary", "edge case: right boundary"],
)
def test_pupil_to_iris_property_validator(p2i_property: float) -> None:
    validator = obj_v.Pupil2IrisPropertyValidator(
        min_allowed_diameter_ratio=0.2, max_allowed_diameter_ratio=0.5, max_allowed_center_dist_ratio=0.5
    )

    try:
        validator(p2i_property)
        assert True
    except E.PupilIrisPropertyEstimationError:
        assert False, "E.PupilToIrisRatioEstimationError exception raised."


@pytest.mark.parametrize(
    "p2i_property, expected_error",
    [
        (
            PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.9, pupil_to_iris_center_dist_ratio=0.1),
            E.Pupil2IrisValidatorErrorDilation,
        ),
        (
            PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.19, pupil_to_iris_center_dist_ratio=0.1),
            E.Pupil2IrisValidatorErrorConstriction,
        ),
        (
            PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.51, pupil_to_iris_center_dist_ratio=0.1),
            E.Pupil2IrisValidatorErrorDilation,
        ),
        (
            PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.2, pupil_to_iris_center_dist_ratio=0.8),
            E.Pupil2IrisValidatorErrorOffcenter,
        ),
    ],
    ids=["simple", "edge case: left boundary", "edge case: right boundary", "center distance too big"],
)
def test_pupil_to_iris_property_validator_raise_exception1(p2i_property: float, expected_error: Exception) -> None:
    validator = obj_v.Pupil2IrisPropertyValidator(
        min_allowed_diameter_ratio=0.2, max_allowed_diameter_ratio=0.5, max_allowed_center_dist_ratio=0.5
    )

    with pytest.raises(expected_error):
        validator(p2i_property)


@pytest.mark.parametrize("offgaze", [Offgaze(score=0.3), Offgaze(score=0.5)], ids=["simple", "edge case: boundary"])
def test_offgaze_score_validator(offgaze: float) -> None:
    validator = obj_v.OffgazeValidator(max_allowed_offgaze=0.5)

    try:
        validator(offgaze)
        assert True
    except E.OffgazeEstimationError:
        assert False, "E.OffgazeEstimationError exception raised."


@pytest.mark.parametrize("offgaze", [Offgaze(score=0.8), Offgaze(score=0.51)], ids=["simple", "edge case: boundary"])
def test_offgaze_score_validator_raise_exception(offgaze: float) -> None:
    validator = obj_v.OffgazeValidator(max_allowed_offgaze=0.5)

    with pytest.raises(E.OffgazeEstimationError):
        validator(offgaze)


@pytest.mark.parametrize(
    "eye_occlusion",
    [EyeOcclusion(visible_fraction=0.8), EyeOcclusion(visible_fraction=0.5)],
    ids=["simple", "edge case: boundary"],
)
def test_occlusion_visible_fraction_validator(eye_occlusion: EyeOcclusion) -> None:
    validator = obj_v.OcclusionValidator(min_allowed_occlusion=0.5)

    try:
        validator(eye_occlusion)
        assert True
    except E.OcclusionError:
        assert False, "E.OcclusionError exception raised."


@pytest.mark.parametrize(
    "eye_occlusion",
    [EyeOcclusion(visible_fraction=0.2), EyeOcclusion(visible_fraction=0.49)],
    ids=["simple", "edge case: boundary"],
)
def test_occlusion_visible_fraction_validator_raise_exception(eye_occlusion: float) -> None:
    validator = obj_v.OcclusionValidator(min_allowed_occlusion=0.5)

    with pytest.raises(E.OcclusionError):
        validator(eye_occlusion)


@pytest.mark.parametrize(
    "geometry_polygons",
    [
        GeometryPolygons(
            pupil_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(200.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(50.0, 450.0, 400.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(50.0, 350.0, 400.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(50.0, 400.0, 450.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(50.0, 400.0, 350.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
    ],
    ids=[
        "simple",
        "pupil and iris boundaries aligned",
        "pupil and iris boundaries partially aligned (right)",
        "pupil and iris boundaries partially aligned (left)",
        "pupil and iris boundaries partially aligned (down)",
        "pupil and iris boundaries partially aligned (up)",
    ],
)
def test_is_pupil_inside_iris_validator(geometry_polygons: GeometryPolygons) -> None:
    validator = obj_v.IsPupilInsideIrisValidator()

    try:
        validator(geometry_polygons)
        assert True
    except E.IsPupilInsideIrisValidatorError:
        assert False, "E.IsPupilInsideIrisValidatorError exception raised."


@pytest.mark.parametrize(
    "geometry_polygons",
    [
        GeometryPolygons(
            pupil_array=generate_arc(10.0, 40.0, 40.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(100.0, 350.0, 400.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(100.0, 450.0, 400.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(100.0, 400.0, 350.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(100.0, 400.0, 450.0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(100.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(300.0, 400.0, 400.0, 0.0, 2 * np.pi, 360),
        ),
    ],
    ids=[
        "entire pupil outside iris",
        "not all pupil points inside iris (left)",
        "not all pupil points inside iris (right)",
        "not all pupil points inside iris (up)",
        "not all pupil points inside iris (down)",
    ],
)
def test_is_pupil_inside_iris_validator_raise_exception(geometry_polygons: GeometryPolygons) -> None:
    validator = obj_v.IsPupilInsideIrisValidator()

    with pytest.raises(E.IsPupilInsideIrisValidatorError):
        validator(geometry_polygons)


@pytest.mark.parametrize(
    "input_polygons,min_pupil_length,min_iris_length",
    [
        ("mock_polygons_arcs_of_circle", 200, 1500),
        ("mock_polygons_arcs_of_circle", 200, 10),
        ("mock_polygons_arcs_of_circle", 10, 1500),
    ],
    ids=["All too small polygons", "Pupil too small", "Iris too small"],
)
def test_polygon_length_validator_raise_exception(
    input_polygons: str, min_iris_length: int, min_pupil_length: int, request
) -> None:
    validator = obj_v.PolygonsLengthValidator(min_iris_length=min_iris_length, min_pupil_length=min_pupil_length)
    with pytest.raises(E.GeometryEstimationError):
        validator(request.getfixturevalue(input_polygons))


@pytest.mark.parametrize(
    "input_polygons,min_pupil_length,min_iris_length",
    [
        ("mock_polygons_arcs_of_circle", 140, 800),
    ],
    ids=["All polygons long enough"],
)
def test_polygon_length_validator(input_polygons: str, min_iris_length: int, min_pupil_length: int, request) -> None:
    validator = obj_v.PolygonsLengthValidator(min_iris_length=min_iris_length, min_pupil_length=min_pupil_length)

    validator(request.getfixturevalue(input_polygons))


@pytest.mark.parametrize(
    "min_sharpness",
    [
        100.0,
        300.0,
        461.0,
    ],
    ids=["low", "medial", "high"],
)
def test_sharpness_validator(min_sharpness: float) -> None:
    sharpness = Sharpness(score=500.0)
    validator = obj_v.SharpnessValidator(min_sharpness=min_sharpness)

    try:
        validator(sharpness)
        assert True
    except E.SharpnessEstimationError:
        assert False, "E.SharpnessEstimationError exception raised."


@pytest.mark.parametrize(
    "code_height,code_width,num_filters,min_maskcodes_size",
    [
        (16, 200, 2, 5120),
        (32, 200, 1, 5000),
    ],
    ids=["good1", "good2"],
)
def test_is_mask_too_small_validator(
    code_height: int, code_width: int, num_filters: int, min_maskcodes_size: int, request
) -> None:
    rng = np.random.default_rng(seed=1)
    mock_iris_template = IrisTemplate(
        iris_codes=[rng.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)],
        mask_codes=[
            rng.choice(2, size=(code_height, code_width, 2), p=[0.1, 0.9]).astype(bool) for _ in range(num_filters)
        ],
        iris_code_version="v3.0",
    )
    validator = obj_v.IsMaskTooSmallValidator(min_maskcodes_size=min_maskcodes_size)
    try:
        validator(mock_iris_template)
        assert True
    except E.MaskTooSmallError:
        assert False, "E.MaskTooSmallError exception raised."


@pytest.mark.parametrize(
    "code_height,code_width,num_filters,min_maskcodes_size",
    [
        (5, 5, 4, 100),
        (50, 10, 2, 2000),
        (10, 100, 2, 4000),
    ],
    ids=["toosmall1", "toosmall2", "toosmall3"],
)
def test_is_mask_too_small_validator_raise_exception(
    code_height: int, code_width: int, num_filters: int, min_maskcodes_size: int
) -> None:
    rng = np.random.default_rng(seed=1)
    mock_iris_template = IrisTemplate(
        iris_codes=[rng.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)],
        mask_codes=[
            rng.choice(2, size=(code_height, code_width, 2), p=[0.5, 0.5]).astype(bool) for _ in range(num_filters)
        ],
        iris_code_version="v3.0",
    )
    validator = obj_v.IsMaskTooSmallValidator(min_maskcodes_size=min_maskcodes_size)

    with pytest.raises(E.MaskTooSmallError):
        validator(mock_iris_template)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)