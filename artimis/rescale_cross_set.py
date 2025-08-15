import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

def rescale_with_new_scaler(orig_path, adv_path, scaler_2017_path, scaler_2018_path, output_dir):
    orig_scaled = pd.read_csv(orig_path, header=None).values
    adv_scaled = pd.read_csv(adv_path, header=None).values

    print(f"✅ load orig: {orig_scaled.shape}, adv: {adv_scaled.shape}")

    scaler_2017: MinMaxScaler = joblib.load(scaler_2017_path)
    scaler_2018: MinMaxScaler = joblib.load(scaler_2018_path)

    orig_raw = scaler_2017.inverse_transform(orig_scaled)
    adv_raw = scaler_2017.inverse_transform(adv_scaled)

    orig_2018_scaled = scaler_2018.transform(orig_raw)
    adv_2018_scaled = scaler_2018.transform(adv_raw)

    # === save ===
    os.makedirs(output_dir, exist_ok=True)
    # pd.DataFrame(orig_raw).to_csv(os.path.join(output_dir, 'orig_raw_2017.csv'), index=False, header=False)
    # pd.DataFrame(adv_raw).to_csv(os.path.join(output_dir, 'adv_raw_2017.csv'), index=False, header=False)
    # pd.DataFrame(orig_2018_scaled).to_csv(os.path.join(output_dir, 'orig_rescaled_2018.csv'), index=False, header=False)
    # pd.DataFrame(adv_2018_scaled).to_csv(os.path.join(output_dir, 'adv_rescaled_2018.csv'), index=False, header=False)

    # pd.DataFrame(orig_raw).to_csv(os.path.join(output_dir, 'orig_2018_1.csv'), index=False, header=False)
    # pd.DataFrame(adv_raw).to_csv(os.path.join(output_dir, 'adv_2018_1.csv'), index=False, header=False)
    pd.DataFrame(orig_2018_scaled).to_csv(os.path.join(output_dir, 'orig_rescaled.csv'), index=False, header=False)
    pd.DataFrame(adv_2018_scaled).to_csv(os.path.join(output_dir, 'adv_rescaled.csv'), index=False, header=False)

    print(f"📦 Four files have been saved to {output_dir}")

if __name__ == '__main__':
    rescale_with_new_scaler(
        orig_path='',
        adv_path='',
        scaler_2017_path='',
        scaler_2018_path='',
        output_dir=''
    )
