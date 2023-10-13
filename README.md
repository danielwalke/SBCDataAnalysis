# SBCDataAnalysis

## Table of Contents 
|content                          |
|---------------------------------|
|[1. Abstract](#overview)     |
|[2. Installation](#installation) |
|[3. Fundings](#fundings)           |
|[4. Competing intrests](#competingIntrests) |

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

<a name="fundings"/>

## 3. Fundings
This work was supported by the German Federal Ministry of Education and Research (de.NBI network. project MetaProtServ. grant no. 031L0103). We highly appreciate their funding.


<a name="competingIntrests"/>

## 4. Competing intrests
The authors declare that they have no competing interests.
