# SynPathy: SYNergy prediction using Pathway

## a. required packages
1. Python=3.6.12
2. keras=2.4.3
3. tensorflow=2.2.0
4. SHAP=0.37.0
5. for other packages, see requirements.txt

## b. example dataset
dataset for one cross validation run is available at: https://zenodo.org/record/5016011#.YNKNHBGSkaY

# How to install SynPathy?

```{python}

# download SynPathy via git clone
$ git clone https://github.com/TangYiChing/SynPathy

# install SynPathy to a conda environment 
$ pip install -r requirements.txt
```
# How to run SynPathy?

```{python}
# case 1: for one cross validation run: optimze MSE through training, validate with Valid.pkl data and finally test on the hold-out Test.pkl data
$ python ./script_run_model/PathComb.CHEM-CHEM-DGNet-DGNet-EXP.py -train ./dataset/train/TRAIN.pkl -valid ./dataset/valid/VALID.pkl -test ./dataset/test/TEST.pkl -norm tanh -g 1 -m 1 -o model.cv0

# case 2: for making prediction with the pre-trained model
$ python ./script_run_model/PathComb.CHEM-CHEM-DGNet-DGNet-EXP.py -train ./dataset/train/TRAIN.pkl -valid ./dataset/valid/VALID.pkl -test ./dataset/test/TEST.pkl -norm tanh -g 1 -m 0 -pretrained model.cv0.saved_model.h5 -o model.cv0
```
# Reference

1.Tang, Y.-C. & Gottlieb, A. SynPathy: Predicting Drug Synergy through Drug-Associated Pathways Using Deep Learning. Mol Cancer Res (2022) doi:10.1158/1541-7786.MCR-21-0735.
