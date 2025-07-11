# Spatio-temporal Fusion Flow with Dynamic Graph Leaning for Unsupervised Multivariate Time Series Anomaly Detection
## Abstract
Detecting anomalies in multivariate time series (MTS) is essential for maintaining system safety in industrial environments. Due to the challenges associated with acquiring labeled data, unsupervised approaches have attracted increasing attention for anomaly detection. However, existing methods face challenges in capturing temporal dependencies within individual entities and spatial correlations among different entities. Moreover, traditional approaches that rely solely on point-wise or periodic criteria often fail to detect structurally complex anomalous patterns. To address these challenges, we propose an unsupervised anomaly detection framework, namely, Spatio-temporal Fusion Flow with Dynamic Graph Learning (STF²-DGL), which jointly models long-short-term temporal patterns and evolving inter-variable dependencies. The framework leverages an encoder to capture multi-scale temporal features, generates dynamic graph structures via spatio-temporal fusion, and integrates these representations into a structure-aware normalizing flow optimized with a graph-enhanced loss. This unified design enables STF²-DGL to effectively distinguish subtle and structurally complex patterns in multivariate time series.

This repository is based on [`USD`](https://github.com/zangzelin/code_USD.git).

## Framework
![Framework](./asset/method1.png)

## Requirements
```plaintext
torch==2.4.1
numpy==1.24.4
torchvision==0.19.1
scipy==1.6.1
scikit-learn==0.24.1
matplotlib==3.7.5
pillow==10.4.0
wandb==0.18.7
pandas==2.0.3
```

```sh
pip install -r requirements.txt
```

## Data
We test our method for five public datasets, e.g., ```SWaT```, ```WADI```, ```PSM```, ```MSL```, and ```SMD```.

[`SWaT`](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat)
[`WADI`](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi)
[`PSM`](https://github.com/tuananhphamds/MST-VAE)
[`MSL`](https://github.com/khundman/telemanom)
[`SMD`](https://github.com/NetManAIOps/OmniAnomaly)

```sh
mkdir Dataset
cd Dataset
mkdir input
```
Download the dataset in ```Data/input```.

## Run the code
For example, run the USD for the dataset of  ```PSM```
```
bash ./run/run_PSM_mymodel.sh
```

## BibTex Citation

