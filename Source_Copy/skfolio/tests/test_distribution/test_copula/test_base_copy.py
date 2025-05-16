import numpy as np
import pytest

from skfolio.distribution import BaseBivariateCopula, GaussianCopula


@pytest.fixture
def random_data():
    """Fixture that returns a random numpy array in [0,1] of shape (100, 2)."""
    rng = np.random.default_rng(seed=42)
    return rng.random((100, 2))


def test_base_bivariate_copula_is_abstract():
    """Check that BaseBivariateCopula cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseBivariateCopula()


def test_validate_X_correct_shape(random_data):
    """Check _validate_X passes with correct input shape and range."""

    # We'll create a minimal subclass that just implements abstract methods.
    cop = GaussianCopula()
    X_validated = cop._validate_X(random_data, reset=True)
    assert X_validated.shape == (100, 2)
    # Check the data remain in [0,1] (they might be clipped slightly)
    assert np.all(X_validated >= 1e-8) and np.all(X_validated <= 1 - 1e-8)


def test_validate_X_wrong_shape():
    """Check _validate_X raises error if not exactly 2 columns."""
    cop = GaussianCopula()

    # 3 columns -> should fail
    with pytest.raises(ValueError, match="X must contains two columns"):
        data_3cols = np.random.rand(10, 3)
        cop._validate_X(data_3cols, reset=True)


def test_validate_X_out_of_bounds():
    """Check _validate_X raises error if values are out of [0,1]."""
    cop = GaussianCopula()
    data_negative = np.array([[0.2, -0.1], [0.3, 0.4]])
    with pytest.raises(ValueError, match="X must be in the interval"):
        cop._validate_X(data_negative, reset=True)


def test_n_params():
    """Check _validate_X raises error if values are out of [0,1]."""
    cop = GaussianCopula()
    assert cop.n_params == 1


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
    with open('/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/func_testcases/skfolio/tests/test_distribution/test_copula/test_base/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/func_testcases/skfolio/tests/test_distribution/test_copula/test_base/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
