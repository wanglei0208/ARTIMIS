import torch
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def get_boxes_rf(model: RandomForestClassifier, x: np.ndarray):
    from sklearn.tree import _tree
    all_boxes = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        boxes = [{} for _ in range(tree.node_count)]

        def recurse(node, bounds):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                f = tree.feature[node]
                thresh = tree.threshold[node]
                left_bounds = bounds.copy()
                left_bounds[f] = (bounds.get(f, (-np.inf, np.inf))[0], min(thresh, bounds.get(f, (-np.inf, np.inf))[1]))
                right_bounds = bounds.copy()
                right_bounds[f] = (max(thresh, bounds.get(f, (-np.inf, np.inf))[0]), bounds.get(f, (-np.inf, np.inf))[1])
                recurse(tree.children_left[node], left_bounds)
                recurse(tree.children_right[node], right_bounds)
            else:
                boxes[node] = bounds

        recurse(0, {})
        leaf_boxes = {i: boxes[i] for i in range(tree.node_count) if tree.children_left[i] == -1}
        all_boxes.append(leaf_boxes)
    return all_boxes

def get_leaf_ids_rf(model, x_np):
    return np.array([estimator.apply(x_np) for estimator in model.estimators_]).T

def get_prediction_rf(model, x_np):
    return model.predict(x_np.reshape(1, -1))[0]

def get_prediction_xgb(model, x_np, feature_names):
    dmatrix = xgb.DMatrix(x_np, feature_names=feature_names)
    return int(model.predict(dmatrix)[0] >= 0.5)

def get_boxes_xgb(model: xgb.Booster, feature_names: list):
    dump = model.get_dump(dump_format='json')
    import json
    all_boxes = []
    for tree_str in dump:
        tree = json.loads(tree_str)
        boxes = []

        def recurse(node, bounds):
            if 'leaf' in node:
                boxes.append(bounds)
            else:
                f_name = node['split']
                f_idx = feature_names.index(f_name)
                thresh = node['split_condition']
                left_bounds = bounds.copy()
                left_bounds[f_idx] = (bounds.get(f_idx, (-np.inf, np.inf))[0], min(thresh, bounds.get(f_idx, (-np.inf, np.inf))[1]))
                right_bounds = bounds.copy()
                right_bounds[f_idx] = (max(thresh, bounds.get(f_idx, (-np.inf, np.inf))[0]), bounds.get(f_idx, (-np.inf, np.inf))[1])
                recurse(node['children'][0], left_bounds)
                recurse(node['children'][1], right_bounds)

        recurse(tree, {})
        leaf_boxes = {i: box for i, box in enumerate(boxes)}
        all_boxes.append(leaf_boxes)
    return all_boxes

def project_to_box(x, box, n_features):
    proj = x.copy()
    for f, (lb, ub) in box.items():
        proj[0, f] = np.clip(proj[0, f], lb, ub)
    return proj

def leaf_tuple_attack(model, x_input, model_type, feature_names=None, max_iters=10):
    device = x_input.device
    x_np = x_input.detach().cpu().numpy()
    x_adv = x_np.copy()
    n_features = x_np.shape[1]

    if model_type == 'sklearn':
        leaf_ids = get_leaf_ids_rf(model, x_np)
        leaf_boxes = get_boxes_rf(model, x_np)
        original_label = get_prediction_rf(model, x_np)

        for step in range(max_iters):
            found = False
            for t_idx, tree in enumerate(model.estimators_):
                tree_leaf_id = tree.apply(x_np)[0]
                for node in leaf_boxes[t_idx]:
                    if node != tree_leaf_id:
                        box = leaf_boxes[t_idx][node]
                        x_try = project_to_box(x_adv.copy(), box, n_features)
                        x_pred = model.predict(x_try.reshape(1, -1))[0]
                        if x_pred != original_label:
                            x_adv = x_try
                            found = True
                            break
                if found:
                    break
            if not found:
                break

    elif model_type == 'xgboost':
        leaf_boxes = get_boxes_xgb(model, feature_names)
        original_label = get_prediction_xgb(model, x_np, feature_names)

        for step in range(max_iters):
            found = False
            for t_idx, tree_boxes in enumerate(leaf_boxes):
                for node, box in tree_boxes.items():
                    x_try = project_to_box(x_adv.copy(), box, n_features)
                    x_pred = get_prediction_xgb(model, x_try, feature_names)
                    if x_pred != original_label:
                        x_adv = x_try
                        found = True
                        break
                if found:
                    break
            if not found:
                break

    else:
        raise NotImplementedError("Currently, only sklearn.RandomForest and xgboost.Booster are supported")

    delta = x_adv - x_np
    return torch.tensor(delta, dtype=torch.float32, device=device)