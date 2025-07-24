# ARTIMIS:Adaptive Reweighing for Transferable Evasion via Meta-learning in Zero-Query Network Intrusion Detection Systems
![ARTIMIS Demo](framework.pdf)

This is the official repository for the ARTIMIS framework, designed for generating and evaluating adversarial attacks against Network Intrusion Detection Systems (NIDS).

## Quick Start

### 1. Environment Setup

Clone this repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

1.  **Download Datasets**: Obtain the raw network traffic data (e.g., BCCC-CIC-IDS-2017/2018) from a public source such as the [Canadian Institute for Cybersecurity Datasets](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/).

2.  **Feature Extraction**: Process the raw `.pcap` files and extract features to create tabular data (`.csv` files).

3.  **Data Preprocessing**: After feature extraction, use `minmax_scaler.py` to normalize the data and prepare it for the models.

    ```bash
    python data/minmax_scaler.py
    ```

### 3. Model Training

Use the `train.py` scripts located in the `NIDS_Classfier/` subdirectories to train the various NIDS models.

```bash
# Example: Train an AlexNet model
python NIDS_Classfier/CNN/train.py --model AlexNet --base_dir ARTIMIS/CNN/2018/brute_force/AlexNet
```

### 4. Attack Generation

Use `attack.py` to generate adversarial samples. You will need to configure the surrogate and target models within the script or via command-line arguments.

```bash
python artimis/attack.py
```

### 5. Performance Evaluation

Use `evaluate_adv.py` to assess the performance of the models against the generated adversarial samples and calculate metrics like Attack Success Rate (ASR).

```bash
python artimis/evaluate_adv.py
```

## License

This project is licensed under the MIT License.
