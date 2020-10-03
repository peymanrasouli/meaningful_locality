# MeaningfulSampling

This repository contains the implementation source code of the following paper:

Peyman Rasouli and Ingrid Chieh Yu, ["Meaningful Data Sampling for a Faithful Local Explanation Method,"](https://link.springer.com/chapter/10.1007/978-3-030-33607-3_4) in 20th International Conference on Intelligent Data Engineering and Automated Learning (IDEAL 2019), LNCS, vol. 11871, pp. 28–38. Springer, 2019.

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanras/MeaningfulSampling
```
2- Create a conda virtual environment:
```
conda create -n MeaningfulSampling python=3.6
```
3- Activate the conda environment: 
```
conda activate MeaningfulSampling
```
4- Standing in MeaningfulSampling directory, install the requirements:
```
pip install -r requirements.txt
```

# Reproducing the results
To reproduce the results of MeaningfulSampling method on LIME with:

1- Linear Regression as interpretable model run:
```
python test_lime_lr.py
```
2- Decision Tree as interpretable model run:
```
python test_lime_dt.py
```
