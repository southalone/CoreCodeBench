import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = './testcases/open-iris/src/iris/nodes/vectorization/contouring/lhs.tmp'
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
tracker = VariableTracker(start_line=106, end_line=119, target_file='./Source_Copy/open-iris/src/iris/nodes/vectorization/contouring.py')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

from functools import partial
from typing import Callable, List

import numpy as np
import pytest
from pydantic import NonNegativeFloat, ValidationError

from iris.io.dataclasses import GeometryMask
from iris.nodes.vectorization.contouring import ContouringAlgorithm, filter_polygon_areas


@pytest.mark.parametrize(
    "mock_polygons,rel_tr,abs_tr,expected_result",
    [
        (
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
                np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),  # area = 0.25
            ],
            0.5,
            0.0,
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
            ],
        ),
        (
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
                np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),  # area = 0.25
            ],
            0.0,
            0.5,
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
            ],
        ),
    ],
    ids=["smaller than abs_tr", "smaller than rel_tr"],
)
def test_filter_polygon_areas(
    mock_polygons: List[np.ndarray],
    rel_tr: NonNegativeFloat,
    abs_tr: NonNegativeFloat,
    expected_result: List[np.ndarray],
) -> None:
    result = filter_polygon_areas(mock_polygons, rel_tr=rel_tr, abs_tr=abs_tr)

    np.testing.assert_equal(result, expected_result)


@pytest.fixture
def algorithm() -> ContouringAlgorithm:
    return ContouringAlgorithm(contour_filters=[filter_polygon_areas])


def test_geometry_raster_constructor() -> None:
    mock_pupil_mask = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )

    mock_iris_mask = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    mock_eyeball_mask = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    result = GeometryMask(pupil_mask=mock_pupil_mask, iris_mask=mock_iris_mask, eyeball_mask=mock_eyeball_mask)

    np.testing.assert_equal(result.filled_eyeball_mask, mock_eyeball_mask + mock_iris_mask + mock_pupil_mask)
    np.testing.assert_equal(result.filled_iris_mask, mock_iris_mask + mock_pupil_mask)


@pytest.mark.parametrize(
    "mock_pupil_mask,mock_iris_mask,mock_eyeball_mask",
    [
        (
            np.array([0, 0, 0], dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
        ),
        (
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.array([0, 0, 0], dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
        ),
        (
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.array([0, 0, 0], dtype=np.uint8),
        ),
    ],
    ids=["wrong dimension of pupil mask", "wrong dimension of iris mask", "wrong dimension of eyeball mask"],
)
def test_geometry_raster_constructor_raises_an_exception(
    mock_pupil_mask: np.ndarray, mock_iris_mask: np.ndarray, mock_eyeball_mask: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _ = GeometryMask(pupil_mask=mock_pupil_mask, iris_mask=mock_iris_mask, eyeball_mask=mock_eyeball_mask)


@pytest.mark.parametrize(
    "contour_filters",
    [
        ([]),
        ([filter_polygon_areas]),
        ([filter_polygon_areas, partial(filter_polygon_areas, atol=0.1, rtol=0.05)]),
    ],
    ids=["empty filter list", "single element list", "more elements list"],
)
def test_constructor(contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]]) -> None:
    _ = ContouringAlgorithm(contour_filters=contour_filters)


@pytest.mark.parametrize(
    "contour_filters",
    [(None), (filter_polygon_areas)],
    ids=["None value", "func not in a list"],
)
def test_constructor_raises_an_exception(contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]]) -> None:
    with pytest.raises(ValidationError):
        _ = ContouringAlgorithm(contour_filters=contour_filters)


def test_eliminate_tiny_contours() -> None:
    algorithm = ContouringAlgorithm(contour_filters=[partial(filter_polygon_areas, abs_tr=0.5)])

    mock_contours = [
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
        np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),  # area = 0.25
    ]

    expected_result = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]  # area = 1.0

    # condition: area > 0.5
    result = algorithm._filter_contours(mock_contours)

    np.testing.assert_equal(result, expected_result)

if  __name__ == "__main__":
    unittest.main()
# 取消跟踪
sys.settrace(None)