## Create Persona dialogue dataset from Reddit dataset

This repository provides the data process scripts.


## Installation
```
pip install -r requirements.txt
```
## 1.Data Preparation
Before running, you should download Reddit datasets.
Link:
https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/

## 2.How to Run
There are python scripts(.py scripts) and jupyter notebook(.ipynb).
Both scripts process is same.
You can use which you like.

* **training data processing**
```
python PersonaGeneration.py
```
Default input path is ./reddit_data/*/*.json
If you wanna change input file path, you can change INPUT_PATH.
Default output path is ./outputs/.


* **test data processing ** 
```
python PreprocessOfTest.py
```
Input file is output of PersonaGeneration.py.


## 3.Reference
Definition of persona.
https://arxiv.org/abs/1809.01984