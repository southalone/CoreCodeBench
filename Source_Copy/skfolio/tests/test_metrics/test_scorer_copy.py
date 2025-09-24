"""Test Scorer module."""

import numpy as np
import sklearn.model_selection as sks

from skfolio import RatioMeasure, RiskMeasure
from skfolio.metrics import make_scorer
from skfolio.optimization import MeanRisk, ObjectiveFunction


def test_default_score(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)
    grid_search = sks.GridSearchCV(
        estimator=model, cv=cv, n_jobs=-1, param_grid={"l2_coef": l2_coefs}
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.sharpe_ratio
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_ratio(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)

    # ratio measure
    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(RatioMeasure.CDAR_RATIO),
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cdar_ratio
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_risk_measure(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)
    # risk measure
    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(RiskMeasure.CVAR),
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cvar
        res[f"split{i}_test_score"] = -d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_custom(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)

    # Custom
    def custom(prediction):
        return prediction.cvar - 2 * prediction.cdar

    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(custom),
    )

    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cvar - 2 * pred.cdar
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


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
    with open('./func_testcases/skfolio/tests/test_metrics/test_scorer/funcCallTree_new.json', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('./func_testcases/skfolio/tests/test_metrics/test_scorer/funcCallTree2level.json', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
