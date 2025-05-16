import numpy as np
import pytest
from scipy.stats import johnsonsu

from skfolio.distribution import JohnsonSU


@pytest.fixture(scope="module")
def johnsonsu_model():
    """
    Fixture for creating and fitting a JohnsonSU estimator.

    Synthetic data is generated from a Johnson SU distribution with known parameters.
    For instance, we set a=1.0, b=1.5, loc=0.0, scale=1.0.
    """
    # Known parameters for the synthetic Johnson SU distribution.
    a_true = 1.0
    b_true = 1.5
    loc_true = 0.0
    scale_true = 1.0

    # Generate synthetic data.
    np.random.seed(42)
    X = johnsonsu.rvs(
        a_true, b_true, loc=loc_true, scale=scale_true, size=50000, random_state=42
    )
    X = X.reshape(-1, 1)

    # Create an estimator (with both loc and scale free for fitting)
    model = JohnsonSU(random_state=123)
    model.fit(X)
    return model


def test_fit_estimates_parameters(johnsonsu_model):
    """
    Test that the fit method estimates parameters reasonably close to the true values.

    Since parameter estimation for Johnson SU can be challenging,
    we test that the fitted parameters are finite and that the fitted loc is close
    to the synthetic data's location.
    """
    # We know that synthetic data was generated with loc = 0.0 and scale = 1.0.
    np.testing.assert_allclose(johnsonsu_model.a_, 1.0, atol=0.2)
    np.testing.assert_allclose(johnsonsu_model.b_, 1.5, atol=0.2)
    np.testing.assert_allclose(johnsonsu_model.loc_, 0.0, atol=0.2)
    np.testing.assert_allclose(johnsonsu_model.scale_, 1.0, atol=0.2)


def test_scipy_params(johnsonsu_model):
    """
    Test that scipy_params returns the correct dictionary of fitted parameters.
    """
    params = johnsonsu_model._scipy_params
    np.testing.assert_allclose(params["loc"], johnsonsu_model.loc_)
    np.testing.assert_allclose(params["scale"], johnsonsu_model.scale_)
    np.testing.assert_allclose(params["a"], johnsonsu_model.a_)
    np.testing.assert_allclose(params["b"], johnsonsu_model.b_)


def test_fitted_repr(johnsonsu_model):
    """
    Test that fitted_repr returns a string representation including parameter values.
    """
    repr_str = johnsonsu_model.fitted_repr
    assert isinstance(repr_str, str)
    assert "JohnsonSU(" in repr_str


def test_score_samples(johnsonsu_model):
    """
    Test that score_samples returns log-density values matching SciPy's logpdf.
    """
    X_test = np.array([[-1.0], [0.0], [1.0]])
    log_densities = johnsonsu_model.score_samples(X_test)
    expected = johnsonsu.logpdf(X_test, **johnsonsu_model._scipy_params).ravel()
    np.testing.assert_allclose(log_densities, expected, rtol=1e-5)
    assert log_densities.shape[0] == X_test.shape[0]


def test_score(johnsonsu_model):
    """
    Test that score returns the sum of log-likelihoods from score_samples.
    """
    X_test = np.linspace(-2, 2, 50).reshape(-1, 1)
    total_log_likelihood = johnsonsu_model.score(X_test)
    np.testing.assert_almost_equal(
        total_log_likelihood, np.sum(johnsonsu_model.score_samples(X_test)), decimal=5
    )


def test_sample(johnsonsu_model):
    """
    Test that sample generates an array of the correct shape and reasonable values.
    """
    samples = johnsonsu_model.sample(n_samples=20)
    assert samples.shape == (20, 1)
    # For a reasonable range check, ensure samples are finite.
    assert np.all(np.isfinite(samples))


def test_cdf_ppf(johnsonsu_model):
    """
    Test that cdf and ppf are (approximately) inverses for the fitted model.
    """
    probabilities = np.linspace(0.1, 0.9, 5)
    quantiles = johnsonsu_model.ppf(probabilities)
    computed_probs = johnsonsu_model.cdf(quantiles.reshape(-1, 1))
    np.testing.assert_allclose(computed_probs.flatten(), probabilities, atol=1e-5)


def test_plot_pdf(johnsonsu_model):
    """
    Test that plot_pdf_2d returns a Plotly Figure with expected layout and data.
    """
    fig = johnsonsu_model.plot_pdf(title="Johnson SU PDF")
    # Check that the figure contains at least one trace and title text is correct.
    assert len(fig.data) >= 1
    assert "Johnson SU PDF" in fig.layout.title.text


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
    with open('/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/func_testcases/skfolio/tests/test_distribution/test_univariate/test_johnson_su/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/func_testcases/skfolio/tests/test_distribution/test_univariate/test_johnson_su/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
