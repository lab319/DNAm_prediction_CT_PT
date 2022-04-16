# User Guide

**Contact:**

**Citation:**

## 1.Introduction

This repository contains a python program to build a cross-tissue prediction model for DNA methylation levels across a pair of cancer and paracancer tissues. The developed prediction model can be applied to a new DNA methylation dataset where only one of the tissue is available and output predicted DNA methylation value of the interested tissue. If you find the program useful, please cite the above reference.

## 2.The version of Python and packages

Python version=3.7
XGBoost version=0.82
scikit-learn version=0.24.2
numpy version=1.20.1
pandas version=1.2.2

## 3.The datasets of the program

The DNA methylation data used in this research were collected from The Cancer Genome Atlas (TCGA) project and that are publicly available at https://portal.gdc.cancer.gov. The 'Data' folder contains two subfolders: 'surrogate_tissue_data' and 'target_tissue_data' folders. We matched DNA methylation data of cancer and paracancer tissues from the same patient in advance. Rows and columns of the files are corresponding to CpG sites and samples respectively.

## 4. How to use our program and obtain output predicted DNA methylation values

(1) 
