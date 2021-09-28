# FaithfulXAI

This repository contains the implementation source code of the following paper:

[Meaningful Data Sampling for a Faithful Local Explanation Method](https://link.springer.com/chapter/10.1007/978-3-030-33607-3_4)

Bibtex:

    @inproceedings{rasouli2019meaningful,
                   title={Meaningful Data Sampling for a Faithful Local Explanation Method},
                   author={Rasouli, Peyman and Yu, Ingrid Chieh},
                   booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
                   pages={28--38},
                   year={2019},
                   organization={Springer}
    }

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanrasouli/FaithfulXAI
```
2- Create a conda virtual environment:
```
conda create -n FaithfulXAI python=3.6
```
3- Activate the conda environment: 
```
conda activate FaithfulXAI
```
4- Standing in FaithfulXAI directory, install the requirements:
```
pip install -r requirements.txt
```

# Reproducing the results
To reproduce the results of FaithfulXAI method on LIME with:

1- Linear Regression as interpretable model run:
```
python test_lime_lr.py
```
2- Decision Tree as interpretable model run:
```
python test_lime_dt.py
```
