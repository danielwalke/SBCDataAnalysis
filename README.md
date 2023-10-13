# SBCDataAnalysis

## Table of Contents 
|content                          |
|---------------------------------|
|[1. Abstract](#overview)     |
|[2. Installation](#installation) |
|[3. Description](#description) |
|[4. Fundings](#fundings)           |
|[5. Competing intrests](#competingIntrests) |

<a name="overview"/>

## 1. Abstract
Artificial intelligence (AI) is currently revolutionizing countless domains in our
daily lives. In many domains like medicine data required for building AI is interconnected (e.g., sequential
measurements). However, current AI algorithms cannot utilize connections between data which limits their
learning capabilities. A promising technology for exploiting those connections is Graph Neural Networks. In
this study, we evaluated when Graph Neural Networks represent a valuable alternative to current AI algorithms
and what limitations this new technology has exemplified on the classification of blood measurements
as septic or not. Finally, we reveal the underlying mechanisms of Graph Neural Networks and current AI
approaches for the prediction.

<a name="installation"/>

## 2. Installation
1) Unzip the CSV in extdata
2) Install packages using conda:
   ```bash
   conda create -n myenv --file package-list.txt
   ```
3) Some packages were easier to install using pip (e.g., sklearn), so why those are included in the requirements.txt. Install them using
   ```bash
   pip install -r requirements.txt
   ```
Note: We have used Conda version 11.7 with the following hardware setup:
•	Mainboard Supermicro X12SPA-TF
•	CPU: Intel® Xeon® Scalable Processor “Ice Lake” Gold 6338, 2.0 GHz, 32- Core
•	GPU: NVIDIA® RTX A6000 (48 GB GDDR6)
•	RAM: 8x32 GB DDR4-3200
•	ROM: 2TB Samsung SSD 980 PRO, M.2  

If you have any issues or problems in reproducing, do not hesitate to create an Issue or directly contact me using the following e-mail:
daniel.walke@ovgu.de

<a name="description"/>

## 2. Description
I have created multiple directories each containing different parts relevant for the SBC analysis

1) extdata - containing the original dataset from Steinbach et al. (https://github.com/ampel-leipzig/sbcdata/tree/main)
2) dataAnalysis - Contains all python scripts for pre-processing the data (reading, filtering and transforming the data) (DataAnalysis.py) and some scripts for metric evaluations (Metrics.py) and constructing feature importance and slope (FeatureImportance.py)
3) noise - contains scripts for writing noisy features to the original dataset
4) machine_learning - contains all jupyter notebooks for analyzing the dataset using machine learning algorithms (logistic regression, decision tree, random forest, XGBoost, RUSBoost) and creates/writes the feature variation graphs
5) neural_network - contains the jupyter notebook for analyzing the dataset using the proposed neural network
6) graph_learning - contains all jupyter notebooks for analyzing the dataset as similarity graphs (heterogeneous & homogeneous) and as patient-centric graphs (patient_centric) using graph learning and evaluations for the resulting attention scores
7) feature_variation - contains jupyter notebooks for writing feature variation graphs for the graph learning algorithms and the neural network and the jupyter notebook for visualizing all graphs for each algorithm in feature variation graphs
8) cuda_test - checks the cuda version and availability
9) keep_ssh - to ensure that the ssh tunnel does not break for a defined inactive period 

If you have any questions regarding the structure or specific implementation details, do not hesitate to contact me using the following e-mail:
daniel.walke@ovgu.de

<a name="fundings"/>

## 4. Fundings
This work was supported by the German Federal Ministry of Education and Research (de.NBI network. project MetaProtServ. grant no. 031L0103). We highly appreciate their funding.


<a name="competingIntrests"/>

## 5. Competing interests
The authors declare that they have no competing interests.
