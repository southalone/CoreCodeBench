import numpy as np
import pytest
from scipy.stats import norm

from skfolio.distribution import BaseUnivariateDist


class DummyUnivariate(BaseUnivariateDist):
    """Dummy univariate estimator using the standard normal distribution."""

    _scipy_model = norm

    def __init__(self, random_state):
        super().__init__(random_state=random_state)

    @property
    def _scipy_params(self) -> dict[str, float]:
        # Standard normal: mean 0, std 1.
        return {"loc": self.loc_, "scale": self.scale_}

    def fit(self):
        self.loc_ = 0
        self.scale_ = 1
        return self

    @property
    def fitted_repr(self) -> str:
        return "Standard Normal"


@pytest.fixture
def dummy_model():
    """Fixture for creating a fitted DummyUnivariate instance."""
    model = DummyUnivariate(random_state=42).fit()
    # "Fitting" in our context just means that scikit-learn's check_is_fitted will pass.
    # We simulate fitting by calling _validate_X once.
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    model._validate_X(X, reset=True)
    return model


def test_n_params(dummy_model):
    assert dummy_model.n_params == 2


def test_score_samples(dummy_model):
    """Test that score_samples returns log-density values."""
    X = np.array([[-1.0], [0.0], [1.0]])
    log_dens = dummy_model.score_samples(X)
    # For standard normal, logpdf at 0 should be approx -0.9189
    np.testing.assert_almost_equal(log_dens[1], norm.logpdf(0), decimal=5)
    assert log_dens.shape[0] == X.shape[0]


def test_score(dummy_model):
    """Test that score returns the sum of log-likelihoods."""
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    total_log_likelihood = dummy_model.score(X)
    np.testing.assert_almost_equal(
        total_log_likelihood, np.sum(dummy_model.score_samples(X)), decimal=5
    )


def test_sample(dummy_model):
    """Test that sample returns an array of the correct shape."""
    samples = dummy_model.sample(n_samples=10)
    assert samples.shape == (10, 1)
    # Check that samples are roughly in the range for a standard normal
    assert np.all(samples > -5) and np.all(samples < 5)


def test_cdf_ppf(dummy_model):
    """Test that cdf and ppf are inverses of each other."""
    probabilities = np.linspace(0.1, 0.9, 5)
    quantiles = dummy_model.ppf(probabilities)
    computed_probabilities = dummy_model.cdf(quantiles.reshape(-1, 1))
    np.testing.assert_allclose(
        computed_probabilities.flatten(), probabilities, atol=1e-5
    )


def test_plot_pdf(dummy_model):
    """Test that plot_pdf_2d returns a Plotly Figure with expected data."""
    fig = dummy_model.plot_pdf(title="Test PDF")
    # Check that figure has at least one trace
    assert len(fig.data) >= 1
    # Check that layout title matches
    assert "Test PDF" in fig.layout.title.text


def test_qq_plot(dummy_model):
    """Test that plot_pdf_2d returns a Plotly Figure with expected data."""
    fig = dummy_model.qq_plot(
        X=np.array([1, 2, 3, 4]).reshape(-1, 1), title="Test Q-Q Plot"
    )
    # Check that figure has at least one trace
    assert len(fig.data) >= 1
    # Check that layout title matches
    assert "Test Q-Q Plot" in fig.layout.title.text


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
    with open('./func_testcases/skfolio/tests/test_distribution/test_univariate/test_base/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('./func_testcases/skfolio/tests/test_distribution/test_univariate/test_base/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
