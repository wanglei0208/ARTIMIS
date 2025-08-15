# ARTIMIS:Adaptive Reweighing for Transferable Evasion via Meta-learning in Zero-Query Network Intrusion Detection Systems
![ARTIMIS Demo](./framework.pdf)

This is the official repository for the ARTIMIS framework, designed for generating and evaluating adversarial attacks against Network Intrusion Detection Systems (NIDS).

## Quick Start

### 1. Environment Setup

Clone this repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

### 2. Data Preparation(The csv file after data preprocessing has been placed in the "data" directory.)

1.  **Download Datasets**: Obtain the raw network traffic data (e.g., BCCC-CIC-IDS-2017/2018) from a public source such as the [Canadian Institute for Cybersecurity Datasets](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/).

2.  **Feature Extraction**: Process the raw `.pcap` files and extract features to create tabular data (`.csv` files).

3.  **CSV Processing**: Run `malicious_sample.ipynb` to sample malicious records and `benign_sample.ipynb` to sample benign records. Afterwards, run `merge_final.ipynb` and `clean.ipynb` to merge and clean the data.


### 3. Scaler

```bash
python data/minmax_scaler.py
```

### 4. Model Training

Use the `train.py` scripts located in the `NIDS/` subdirectories to train the various NIDS models.

```bash
# Example: Train an AlexNet model
cd NIDS/CNN 
python train.py --model AlexNet
```

### 5. Attack Generation

Use `attack.py` to generate adversarial samples. You will need to configure the surrogate and target models within the script or via command-line arguments.

```bash
cd ..
cd ..
cd artimis
python attack.py
```

## License

This project is licensed under the MIT License.
