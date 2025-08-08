import pandas as pd
def analyze_csv_features(csv_path, output_txt_path):
    df = pd.read_csv(csv_path)

    feature_columns = df.columns[:-1]
    label_column = df.columns[-1]

    with open(output_txt_path, 'w') as f:
        f.write("Feature column analysis report\n")
        f.write("====================\n\n")
        for col in feature_columns:
            max_val = df[col].max()
            min_val = df[col].min()
            max_label = df[df[col] == max_val][label_column].iloc[0]
            min_label = df[df[col] == min_val][label_column].iloc[0]

            f.write(f"feature: {col}\n")
            f.write(f"  max: {max_val} (class: {max_label})\n")
            f.write(f"  min: {min_val} (class: {min_label})\n")
            f.write("\n")

        f.write("====================\n")
        f.write("finished\n")

csv_file = "ARTIMIS/data/brute_force/bf.csv"
output_file = "ARTIMIS/data/brute_force/bf_feature_analysis.txt"
analyze_csv_features(csv_file, output_file)
