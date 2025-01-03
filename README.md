# MVA Deep Learning Course - Project

This repository contains the code for the project of the Deep Learning course of the MVA master at ENS Paris-Saclay (2024).

## Setup the project
To setup the project, do:
```bash
python -m venv dl_project_phydnet
source dl_project_phydnet/bin/activate
```

To install the requirements:
```bash
pip install -r requirements.txt
```

## Training

First, you need to get access to the datasets. 
Either you download the CIKM dataset: https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y, and the MNIST and MovingMNIST datasets (using the file `./data/download.sh`) (not recommended)
Or you use the datasets created on Kaggle for the sake of this project (recommended):
1. To train PhyDNet on the CIKM dataset, copy paste the file `./kaggle/phyd_lstm.ipynb` on this dataset project: https://www.kaggle.com/datasets/lounsmh/weather-data
2. To train the double-LSTM PhyDNet model, copy paste the file `./kaggle/phyd_double_lstm.ipynb` on this dataset project:  https://www.kaggle.com/datasets/lounsmh/mnist-dataset

## Usage

First, download MNIST and MovingMNIST datasets:
```bach
cd data/
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
wget https://github.com/hamlinzheng/mnist/raw/master/dataset/train-images-idx3-ubyte.gz
```

To generate predictions, you can use the provided pertained models in the folder `./save`.

To make predictions on MovingMNIST using PhyDNet:
```bash
python main.py
```

To make predictions on MovingMNIST using the double-LSTM PhyDNet model:
```bash
python main_lstm.py
```

To make predictions on the CIKM dataset using PhyDNet:
```bash
python main_weather.py
```

## Acknowledgements

We would like to thank:
* The authors of the paper [Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction](https://arxiv.org/abs/2003.01460) from which we are basing this project.

## Contact and Contributors

This project is conducted by: [Lounès Meddahi]().