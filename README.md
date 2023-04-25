# User Guide

**Contact:**

Shuzheng Zhang (18353291726@139.com)

Baoshan Ma (mabaoshan@dlmu.edu.cn)

Yu Liu(ly1392570280@163.com)

Yiwen Shen

Di Li

Shuxin Liu

Fengju Song

**Citation:** Predicting locus-specific DNA methylation levels in cancer and paracancer tissues

## 1.Introduction

This repository contains a python program to build a cross-tissue prediction model for DNA methylation levels across a pair of cancer and paracancer tissues. The developed prediction model can be applied to a new DNA methylation dataset where only one of the tissue is available and output predicted DNA methylation value of the interested tissue. If you find the program useful, please cite the above reference.

## 2.The version of Python and packages

Python version=3.7

XGBoost version=0.82

scikit-learn version=0.24.2

numpy version=1.20.1

pandas version=1.2.2

## 3.The datasets of the program

The DNA methylation data used in this research were collected from The Cancer Genome Atlas (TCGA) project and that are publicly available at https://portal.gdc.cancer.gov. The 'Data' folder contains three files: 'surrogate_feature.txt','surrogate_tissue.txt' and 'target_tissue.txt' files. We matched DNA methylation data of cancer and paracancer tissues from the same patient in advance. Rows and columns of the files are corresponding to CpG sites and samples respectively.

## 4. How to use our program and obtain output predicted DNA methylation values

(1) Organize three data files: 'surrogate_feature.txt', 'surrogate_tissue.txt', and 'target_tissue.txt' files.

(2) Define file names of output predicted DNA methylation values for target tissue. For example, 'LR-single_target.txt' is the predicted DNA methylation values of target tissue based on single-CpG-based linear regression model, 'SVM-single_target.txt' is the predicted DNA methylation values of target tissue based on single-CpG-based SVM model, 'XGBoost-single_target.txt' is the predicted DNA methylation values of target tissue based on single-CpG-based XGBoost model.'LR-multiple_target.txt' is the predicted DNA methylation values of target tissue based on multiple-CpG-based linear regression model, 'SVM-multiple_target.txt' is the predicted DNA methylation values of target tissue based on multiple-CpG-based SVM model, 'XGBoost-multiple_target.txt' is the predicted DNA methylation values of target tissue based on multiple-CpG-based XGBoost model.

(3) Now, you can run the python program under 'model_construction' folder. The 'model_construction' folder contains two subfolders: 'single-CpG-based_model' and 'multiple-CpG-based_model'. The proposed cross-tissue prediction models based on a single CpG site were included in 'single-CpG-based_model' folder. In each python file, two input parameters are required, corresponding to DNA methylation data file names for surrogate and target tissue datasets, which can be found in 'surrogate_tissue.txt' and 'target_tissue.txt' files under 'Data' folder. The proposed cross-tissue prediction models based on multiple correlating CpG sites were included in 'multiple-CpG-based_model' folder. Three input parameters are required in each python file, corresponding to DNA methylation data file names for feature selection, surrogate, target tissue datasets, which can be found in 'surrogate_feature.txt', 'surrogate_tissue.txt' and 'target_tissue,txt' files under 'Data' folder.

(4) After the program is excuted, you can obtain the predicted DNA methylation values of these CpG sites provided by us under the 'output_DNAm_data' folder. 
