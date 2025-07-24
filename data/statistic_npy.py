import numpy as np
def analyze_npy_features(npy_path, output_txt_path):
    data = np.load(npy_path)
    features = data[:, :-1]
    labels = data[:, -1]

    with open(output_txt_path, 'w') as f:
        f.write("Feature Analysis Report\n")
        f.write("=============================\n\n")
        for i, col in enumerate(features.T):
            max_val = np.max(col)
            min_val = np.min(col)
            max_label = labels[np.argmax(col)]
            min_label = labels[np.argmin(col)]

            f.write(f"feature {i + 1}:\n")
            f.write(f"  max: {max_val} (class: {max_label})\n")
            f.write(f"  min: {min_val} (class: {min_label})\n")
            f.write("\n")

        f.write("=============================\n")
        f.write("finished\n")

    print(f"The analysis report has been saved to {output_txt_path}")

npy_file = 'ARTIMIS/data/2018/brute_force/X_train.npy'
output_file = 'ARTIMIS/data/2018/brute_force/train_feature_analysis.txt'
analyze_npy_features(npy_file, output_file)
