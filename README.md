## Create Persona dialogue dataset from Reddit dataset

This repository provides the data process scripts.

## 1.Data Preparation
Before running, you should download Reddit datasets.
Link:
https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/

## 2.How to Run
* **training data processing**
```
python PersonaGeneration.py
```
PATH indicate input data location.
Output data locate ./outputs/

* **test data processing ** 
```
python PreprocessOfTest.py
```
Input file is output of PersonaGeneration.py