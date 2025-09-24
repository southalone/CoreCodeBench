import numpy as np
import pytest

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.preprocessing import prices_to_returns


@pytest.fixture
def prices():
    prices = load_sp500_dataset()
    return prices


@pytest.fixture(scope="module")
def factor_prices():
    factor_prices = load_factors_dataset()
    return factor_prices


def test_returns(prices, factor_prices):
    # insert random nan
    for col in prices.columns:
        prices.loc[prices.sample(frac=0.1).index, col] = np.nan
    p = prices.ffill().dropna()
    X = prices_to_returns(X=prices)
    np.testing.assert_almost_equal(X.to_numpy(), p.pct_change().iloc[1:].to_numpy())

    X = prices_to_returns(X=prices, log_returns=True)
    np.testing.assert_almost_equal(
        X.to_numpy(), np.log(p / p.shift()).iloc[1:].to_numpy()
    )

    X = prices_to_returns(X=prices, nan_threshold=0.01)
    assert X.shape[0] < prices.shape[0] - 2

    X, y = prices_to_returns(X=prices, y=factor_prices)
    assert np.all(X.columns == prices.columns)
    assert np.all(y.columns == factor_prices.columns)
    assert np.all(X.index == y.index)


def test_returns_drop_inceptions_nan(prices):
    # Test index_intersect by making the first column mostly 0's
    prices.loc[: prices.index[-10], "AAPL"] = np.nan

    X = prices_to_returns(X=prices)
    assert X.shape[0] == 8

    X = prices_to_returns(X=prices, drop_inceptions_nan=False)
    assert X.shape[0] == prices.shape[0] - 1


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
    with open('./func_testcases/skfolio/tests/test_preprocessing/test_returns/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('./func_testcases/skfolio/tests/test_preprocessing/test_returns/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
