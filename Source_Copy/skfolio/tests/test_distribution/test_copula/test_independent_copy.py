import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import IndependentCopula


@pytest.fixture
def X():
    # Using same convention as other libraries for Benchmark
    X = np.array(
        [
            [0.05, 0.1],
            [0.4, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.2, 0.3],
        ]
    )
    return X


@pytest.fixture
def fitted_model(X):
    # Using same convention as other libraries for Benchmark
    fitted_model = IndependentCopula(random_state=42).fit(X)
    return fitted_model


def test_independent_copula_init():
    """Test initialization of IndependentCopula with default parameters."""
    _ = IndependentCopula()


def test_independent_cdf_shape(random_data):
    """Test cdf() returns correct shape."""
    model = IndependentCopula().fit(random_data)
    cdf = model.cdf(random_data)
    assert cdf.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(cdf >= 0) and np.all(cdf <= 1)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_independent_partial_derivative_shape(random_data, first_margin):
    """Test partial_derivative() returns correct shape."""
    model = IndependentCopula().fit(random_data)
    h = model.partial_derivative(random_data, first_margin=first_margin)
    assert h.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(h >= 0) and np.all(h <= 1)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_independent_inverse_partial_derivative_shape(random_data, first_margin):
    """Test inverse_partial_derivative() returns correct shape."""
    model = IndependentCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data, first_margin=first_margin)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_independent_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = IndependentCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_independent_score(random_data):
    """Test the total log-likelihood via score()."""
    model = IndependentCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_independent_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = IndependentCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_independent_sample():
    """Test sample() method for shape and range."""
    model = IndependentCopula(random_state=123).fit(np.random.rand(100, 2))
    samples = model.sample(n_samples=50)
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_independent_score_exact(X, fitted_model):
    assert np.isclose(fitted_model.score(X), 0.0)
    np.testing.assert_almost_equal(
        fitted_model.score_samples(X),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_independent_aic_bic_exact(X, fitted_model):
    np.isclose(fitted_model.aic(X), 0.0)
    np.isclose(fitted_model.bic(X), 0.0)


def test_cdf_exact(
    X,
    fitted_model,
):
    np.testing.assert_almost_equal(
        fitted_model.cdf(X),
        np.array([0.005, 0.08, 0.12, 0.3, 0.06]),
    )


@pytest.mark.parametrize(
    "first_margin,expected",
    [
        (True, np.array([0.1, 0.2, 0.4, 0.6, 0.3])),
        (False, np.array([0.05, 0.4, 0.3, 0.5, 0.2])),
    ],
)
def test_independent_partial_derivative_exact(X, fitted_model, first_margin, expected):
    h = fitted_model.partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(h, expected)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_independent_partial_derivative_numeric(X, fitted_model, first_margin):
    h = fitted_model.partial_derivative(X, first_margin=first_margin)

    delta = 1e-6
    i = 0 if first_margin else 1
    X1 = X.copy()
    X1[:, i] += delta
    X2 = X.copy()
    X2[:, i] -= delta

    h_num = (fitted_model.cdf(X1) - fitted_model.cdf(X2)) / delta / 2

    np.testing.assert_almost_equal(h, h_num)


@pytest.mark.parametrize(
    "first_margin,expected",
    [
        (True, np.array([0.1, 0.2, 0.4, 0.6, 0.3])),
        (False, np.array([0.05, 0.4, 0.3, 0.5, 0.2])),
    ],
)
def test_independent_inverse_partial_derivative_exact(
    X, fitted_model, first_margin, expected
):
    p = fitted_model.inverse_partial_derivative(X, first_margin=first_margin)
    np.testing.assert_almost_equal(p, expected)


@pytest.mark.parametrize(
    "first_margin",
    [
        True,
        False,
    ],
)
def test_independent_partial_derivative_inverse_partial_derivative(
    fitted_model, X, first_margin
):
    """h(u | v) = p => h^-1(p | v) = u"""
    p = fitted_model.partial_derivative(X, first_margin=first_margin)
    if first_margin:
        PV = np.stack([X[:, 0], p], axis=1)
    else:
        PV = np.stack([p, X[:, 1]], axis=1)

    u = fitted_model.inverse_partial_derivative(PV, first_margin=first_margin)
    if first_margin:
        np.testing.assert_almost_equal(X[:, 1], u)
    else:
        np.testing.assert_almost_equal(X[:, 0], u)


def test_tail_concentration(fitted_model):
    quantiles = np.linspace(0.01, 0.99, 50)
    tc = fitted_model.tail_concentration(quantiles)
    assert tc.shape == quantiles.shape, "tail_concentration output shape mismatch"
    assert np.all(tc >= 0), "tail_concentration contains negative values"


def test_plot_tail_concentration(fitted_model):
    fig = fitted_model.plot_tail_concentration(title="Test Tail Concentration")
    assert isinstance(fig, go.Figure), "plot_tail_concentration did not return a Figure"
    # Check that the title is set
    assert "Tail Concentration" in fig.layout.title.text, (
        "plot_tail_concentration title missing"
    )


def test_plot_pdf_2d(fitted_model):
    fig = fitted_model.plot_pdf_2d(title="Test PDF 2D")
    assert isinstance(fig, go.Figure), "plot_pdf_2d did not return a Figure"


# Test plot_pdf_3d
def test_plot_pdf_3d(fitted_model):
    fig = fitted_model.plot_pdf_3d(title="Test PDF 3D")
    assert isinstance(fig, go.Figure), "plot_pdf_3d did not return a Figure"


def test_lower_tail_dependence(fitted_model):
    result = fitted_model.upper_tail_dependence
    assert result == 0.0


def test_upper_tail_dependence(fitted_model):
    result = fitted_model.upper_tail_dependence
    assert result == 0.0


def test_fitted_repr(fitted_model):
    rep = fitted_model.fitted_repr
    assert "IndependentCopula" in rep, "fitted_repr does not contain class name"


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
    with open('/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/func_testcases/skfolio/tests/test_distribution/test_copula/test_independent/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/func_testcases/skfolio/tests/test_distribution/test_copula/test_independent/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
