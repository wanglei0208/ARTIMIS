import os
import yaml
import torch
import joblib
import xgboost as xgb
import pickle
import sys
sys.path.append('ARTIMIS/artimis')
from utils.AverageMeter import AccuracyMeter
from utils.model import LeNet1D,AlexNet1D,MLP, DeepNet, AlertNet, LSTMModel,IdsNet
from kitnet_model.KitNET import KitNET
from model_mampf import EnhancedClassifier
from model_diff_rf import DiFF_TreeEnsemble
def get_models(args, device):
    models = {}
    metrics = {}

    #yaml_path = os.path.join(args.root_path, 'configs/checkpoint2017_bot.yaml')
    #yaml_path = os.path.join(args.root_path, 'configs/checkpoint2018_bot.yaml')
    yaml_path = os.path.join(args.root_path, 'configs/checkpoint2018_brute_force.yaml')
    #yaml_path = os.path.join(args.root_path, 'configs/checkpoint2018_dos.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print('🛠 The NIDS model is being constructed...')
    for name, entry in config.items():
        arch = entry['arch']
        if arch == 'kitnet_rf':
            ckpt_kitnet_path = os.path.join(args.root_path, entry['ckpt']['kitnet'])
            ckpt_rf_path = os.path.join(args.root_path, entry['ckpt']['rf'])
            import pickle
            with open(ckpt_kitnet_path, "rb") as f:
                kitnet_model = pickle.load(f)
            with open(ckpt_rf_path, "rb") as f:
                rf_model = pickle.load(f)
            def kitnet_rf_predict(batch_x):
                # batch_x: numpy array of shape (batch_size, num_features)
                rmse_vectors = [kitnet_model.extract_rmse_vector(xi) for xi in batch_x]
                preds = rf_model.predict(rmse_vectors)  # shape: (batch_size,)
                return preds
            models[name] = {'model': kitnet_rf_predict, 'type': 'function'}
            metrics[name] = AccuracyMeter()
            print(f'✅ The model has been successfully loaded:{name}（arch: {arch}）')

        elif arch == 'mampf':
            ckpt_mampf_path = os.path.join(args.root_path, entry['ckpt']['mampf'])
            ckpt_mlp_path = os.path.join(args.root_path, entry['ckpt']['mlp'])

            # ✅ load MaMPF extractor（pickle）
            import pickle
            mampf_extractor = joblib.load(ckpt_mampf_path)
            # ✅ load MLP classfier
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
            ckpt_diff_path = os.path.join(args.root_path, entry['ckpt']['diffrf'])
            thresh_path = os.path.join(args.root_path, entry['ckpt']['thresh_path'])
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
            ckpt_path = os.path.join(args.root_path, entry['ckpt'])

            if arch == 'lenet':
                model = LeNet1D(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'fsnet':
                model = FSNetForStatFeatures0(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'alexnet':
                model = AlexNet1D(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}

            elif arch == 'mlp1':
                model = MLP(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}
            elif arch == 'mlp2':
                model = MLP(input_dim=60, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()
                models[name] = {'model': model, 'type': 'logit'}
            elif arch == 'mlp3':
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

            # elif arch == 'logistic':
            #     model = LogisticRegressionModel(input_dim=53)
            #     model.load_state_dict(torch.load(ckpt_path, map_location=device))
            #     model.to(device)
            #     model.eval()
            #     models[name] = {'model': model, 'type': 'sigmoid'}

            elif arch == 'rf1':
                model = joblib.load(ckpt_path)
                models[name] = {'model': model, 'type': 'sklearn'}
            elif arch == 'rf2':
                model = joblib.load(ckpt_path)
                models[name] = {'model': model, 'type': 'sklearn'}
            elif arch == 'rf3':
                model = joblib.load(ckpt_path)
                models[name] = {'model': model, 'type': 'sklearn'}
            elif arch == 'rf4':
                model = joblib.load(ckpt_path)
                models[name] = {'model': model, 'type': 'sklearn'}

            elif arch == 'xgboost1':
                model = xgb.Booster()
                model.load_model(ckpt_path)
                models[name] = {'model': model, 'type': 'xgboost'}

            elif arch == 'xgboost2':
                model = xgb.Booster()
                model.load_model(ckpt_path)
                models[name] = {'model': model, 'type': 'xgboost'}

            elif arch == 'xgboost3':
                model = xgb.Booster()
                model.load_model(ckpt_path)
                models[name] = {'model': model, 'type': 'xgboost'}

            elif arch == 'xgboost4':
                model = xgb.Booster()
                model.load_model(ckpt_path)
                models[name] = {'model': model, 'type': 'xgboost'}

            else:
                raise NotImplementedError(f"未知模型架构: {arch}")

            metrics[name] = AccuracyMeter()
            print(f'✅ The model has been successfully loaded: {name}（arch: {arch}）')

    return models, metrics
