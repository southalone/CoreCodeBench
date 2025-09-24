import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import SelectComplete, SelectNonExpiring
from skfolio.preprocessing import prices_to_returns


def generate_prices(n: int) -> list[float]:
    # Just for example purposes
    return list(100 * np.cumprod(1 + np.random.normal(0, 0.01, n)))


@pytest.fixture
def X_df():
    X_df = pd.DataFrame(
        {
            "asset1": [1, 2, 3, 4],
            "asset2": [2, 3, 4, 5],
            "asset3": [3, 4, 5, 6],
            "asset4": [4, 5, np.nan, 7],
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D"),
    )
    return X_df


@pytest.fixture
def prices():
    prices = pd.DataFrame(
        {
            "inception": [np.nan] * 3 + generate_prices(10),
            "defaulted": generate_prices(6) + [0.0] + [np.nan] * 6,
            "expired": generate_prices(10) + [np.nan] * 3,
            "complete": generate_prices(13),
        },
        index=pd.date_range(start="2024-01-03", end="2024-01-19", freq="B"),
    )
    return prices


@pytest.mark.parametrize(
    "expiration_dates,expected",
    [
        (
            {
                "asset1": pd.Timestamp("2023-01-10"),
                "asset2": pd.Timestamp("2023-01-02"),
                "asset3": pd.Timestamp("2023-01-06"),
                "asset4": dt.datetime(2023, 5, 1),
            },
            pd.DataFrame(
                {"asset1": [1, 2, 3, 4], "asset4": [4, 5, np.nan, 7]},
                index=pd.date_range("2023-01-01", periods=4, freq="D"),
            ),
        ),
    ],
)
def test_select_non_expiring(X_df, expiration_dates, expected):
    with config_context(transform_output="pandas"):
        selector = SelectNonExpiring(
            expiration_dates=expiration_dates,
            expiration_lookahead=pd.DateOffset(days=5),
        )
        res = selector.fit_transform(X_df)
        pd.testing.assert_frame_equal(res, expected)


def test_pipeline(prices):
    X = prices_to_returns(prices, drop_inceptions_nan=False, fill_nan=False)

    with config_context(transform_output="pandas"):
        model = Pipeline(
            [
                ("select_complete_assets", SelectComplete()),
                (
                    "select_non_expiring_assets",
                    SelectNonExpiring(
                        expiration_dates={"expired": dt.datetime(2024, 1, 16)},
                        expiration_lookahead=pd.offsets.BusinessDay(4),
                    ),
                ),
                ("zero_imputation", SimpleImputer(strategy="constant", fill_value=0)),
                ("optimization", EqualWeighted()),
            ]
        )
        pred = cross_val_predict(model, X, cv=WalkForward(train_size=4, test_size=4))
        expected = pd.DataFrame(
            {
                "EqualWeighted": {
                    "defaulted": 0.3333333333333333,
                    "expired": 0.3333333333333333,
                    "complete": 0.3333333333333333,
                    "inception": 0.0,
                },
                "EqualWeighted_1": {
                    "defaulted": 0.0,
                    "expired": 0.0,
                    "complete": 0.5,
                    "inception": 0.5,
                },
            },
        )
        expected.index.name = "asset"
        pd.testing.assert_frame_equal(pred.composition, expected)
        assert len(pred.returns) == 8
        assert np.all(~np.isnan(pred.returns))


if __name__ == "__main__":
    # 事先要设置PYTHONPATH = transformers src 的路径
    from pycallgraph import PyCallGraph, Config
    from pycallgraph import GlobbingFilter
    from pycallgraph.output import GraphvizOutput

    from pycallgraph.util import CallNode, CallNodeEncoder
    import os
    import sys
    print(sys.path)
    import json
    import skfolio
    import pytest

    # 定义要测试的文件路径
    project_root = os.path.dirname(os.path.dirname(skfolio.__file__))
    print(project_root)
    print(sys.path)
    sys.path.append(project_root)

    config = Config(project_root=project_root)
    config.trace_filter = GlobbingFilter(
        include=[
            'skfolio.*',   # 包括项目中的所有模块
            'skfolio.*',
        ],
        exclude=[
            'pycallgraph.*', # 排除 pycallgraph 自身
            'os.*',          # 排除 os 模块
            'sys.*',         # 排除 sys 模块
            '*.<listcomp>*','*.<dictcomp>*','*.<setcomp>*','*.<genexpr>*','*.<module>*','*.<locals>*','*.<lambda>*'
        ]
    )

    # 使用 PyCallGraph 进行调用跟踪
    with PyCallGraph(output=GraphvizOutput(),config=config) as pycg:
        pytest.main([__file__, '-s'])
    
    # 获取调用树
    call_tree = pycg.get_call_tree()
    
    def serialize_tree(node, depth):
        if depth == 6:
            return None
        return {
            'name': node.name,
            'source_dir': node.source_dir,
            'call_position': node.call_position,
            'children': [serialize_tree(child, depth+1) for child in node.children],
        }

    def merge_trees(root):
        # 使用字典存储合并后的节点
        node_dict = {}
        def merge_node(node):
            if node.name not in node_dict:
                # 如果节点名称不存在，则创建一个新的节点并添加到字典中
                new_node = CallNode(node.name, node.source_dir, node.call_position)
                node_dict[node.name] = new_node
            else:
                # 如果节点名称已经存在，则获取现有节点
                new_node = node_dict[node.name]
            
            for child in node.children:
                merged_child = merge_node(child)
                # 检查子节点是否已经存在于当前节点的子节点列表中
                if merged_child not in new_node.children and merged_child.name != new_node.name:
                    new_node.add_child(merged_child)

            return new_node
        # 从根节点开始合并
        new_root = merge_node(root)
        return new_root
    
    def get_2level_tree(root, level):
        # 把原来的树剪枝，只保留2层
        if level == 2:
            root.children = []
            return
        else:
            # level = 0/1
            for child in root.children:
                get_2level_tree(child, level+1)


    merged_tree = merge_trees(call_tree)
    get_2level_tree(call_tree, 0)
    
    # 保存调用树到JSON文件
    with open('./func_testcases/skfolio/tests/test_pre_selection/test_select_non_expiring/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('./func_testcases/skfolio/tests/test_pre_selection/test_select_non_expiring/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
