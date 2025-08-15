import os
import yaml
import torch
import joblib
import xgboost as xgb
import pickle
import sys
from utils.AverageMeter import AccuracyMeter
from utils.model import LeNet1D,AlexNet1D,MLP, DeepNet, AlertNet, LSTMModel,IdsNet
from kitnet_model.KitNET import KitNET
from model_mampf import EnhancedClassifier
from model_diff_rf import DiFF_TreeEnsemble

def get_models(args, device):
    models = {}
    metrics = {}

    # The hardcoded yaml_path is kept as per your original script
    yaml_path = "configs/checkpoint2018_brute_force.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print('🛠 The NIDS model is being constructed...')
    for name, entry in config.items():
        arch = entry['arch']
        if arch == 'kitnet_rf':
            # MODIFIED: Removed os.path.join(args.root_path, ...)
            ckpt_kitnet_path = entry['ckpt']['kitnet']
            ckpt_rf_path = entry['ckpt']['rf']
            import pickle
            with open(ckpt_kitnet_path, "rb") as f:
                kitnet_model = pickle.load(f)
            with open(ckpt_rf_path, "rb") as f:
                rf_model = pickle.load(f)
            def kitnet_rf_predict(batch_x):
                rmse_vectors = [kitnet_model.extract_rmse_vector(xi) for xi in batch_x]
                preds = rf_model.predict(rmse_vectors)
                return preds
            models[name] = {'model': kitnet_rf_predict, 'type': 'function'}
            metrics[name] = AccuracyMeter()
            print(f'✅ The model has been successfully loaded:{name}（arch: {arch}）')

        elif arch == 'mampf':
            # MODIFIED: Removed os.path.join(args.root_path, ...)
            ckpt_mampf_path = entry['ckpt']['mampf']
            ckpt_mlp_path = entry['ckpt']['mlp']

            mampf_extractor = joblib.load(ckpt_mampf_path)
            model = EnhancedClassifier(input_dim=62)
            model.load_state_dict(torch.load(ckpt_mlp_path, map_location=device))
            model.to(device)
            model.eval()
            def build_mampf_pipeline(extractor, classifier, device):
                def pipeline(batch_x):
                    feats = extractor.transform(batch_x)
                    feats = torch.tensor(feats, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits = classifier(feats)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                    return preds
                return pipeline
            mampf_pipeline = build_mampf_pipeline(mampf_extractor, model, device)
            models[name] = {'model': mampf_pipeline, 'type': 'function'}
            metrics[name] = AccuracyMeter()
            print(f'✅ The model has been successfully loaded: {name}（arch: {arch}）')

        elif arch == 'diff_rf':
            # MODIFIED: Removed os.path.join(args.root_path, ...)
            ckpt_diff_path = entry['ckpt']['diffrf']
            thresh_path = entry['ckpt']['thresh_path']
            diff_rf_model = joblib.load(ckpt_diff_path)
            with open(thresh_path, 'r') as f:
                diff_thresh = float(f.read().strip())
            def diff_rf_predict(batch_x):
                scores, _, scDF = diff_rf_model.anomaly_score(batch_x)
                preds = (scDF > diff_thresh).astype(int)
                return preds
            models[name] = {'model': diff_rf_predict, 'type': 'function'}
            metrics[name] = AccuracyMeter()
            print(f'✅ The model has been successfully loaded: {name}（arch: {arch}）')

        else:
            # MODIFIED: Removed os.path.join(args.root_path, ...)
            ckpt_path = entry['ckpt']

            if arch == 'lenet':
                model = LeNet1D(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            # elif arch == 'fsnet':
            #     # Assuming FSNetForStatFeatures0 is defined elsewhere
            #     model = FSNetForStatFeatures0(input_dim=60, num_classes=2)
            #     model.load_state_dict(torch.load(ckpt_path, map_location=device))
            #     model.to(device)
            #     model.eval()
            #     models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'alexnet':
                model = AlexNet1D(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch in ['mlp1', 'mlp2', 'mlp3']:
                model = MLP(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}
                
            elif arch == 'idsnet':
                model = IdsNet(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'deepnet':
                model = DeepNet(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'alertnet':
                model = AlertNet(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'lstm':
                model = LSTMModel(input_size=60, hidden_size=120, num_layers=3, output_size=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch in ['rf1', 'rf2', 'rf3', 'rf4']:
                model = joblib.load(ckpt_path)
                models[name] = {'model': model, 'type': 'sklearn'}

            elif arch in ['xgboost1', 'xgboost2', 'xgboost3', 'xgboost4']:
                model = xgb.Booster()
                model.load_model(ckpt_path)
                models[name] = {'model': model, 'type': 'xgboost'}

            # else:
            #     raise NotImplementedError(f"Unknown model architecture: {arch}")

            metrics[name] = AccuracyMeter()
            print(f'✅ The model has been successfully loaded: {name}（arch: {arch}）')

    return models, metrics