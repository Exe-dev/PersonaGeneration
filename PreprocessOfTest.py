#!/usr/bin/env python
# coding: utf-8

# # Rules of persona
# + Each sentence must contain between 4 and 20 words or punctuation marks.
# + It contains either the word I or my.
# + At least one verb, and (iv) at least one noun, pronoun or adjective.

# # Example Dialogue with persona
# - Persona: [“I like sport”, “I work a lot”]
# - Context: “I love running.”
# - Response: “Me too! But only on weekends.”

# {"dialog": [["没有 钱   万万 不行 ！ ~"], ["现实 就是 如此"]], "profile": [{"tag": ["漫画;旅遊;星座"], "loc": "广东 广州", "gender": "male"}, {"tag": [""], "loc": "", "gender": ""}], "uid": [0, 1]}

# {'dialog': ["For what it's worth, I don't have a problem with it.", 'My apologies.  I did not have any problems with it, but I will be more careful in the future.'], 'profile': [{'tag': ["for what it's worth, i don't have a problem with it."], 'loc': '', 'gender': ''}, {'tag': [' i did not have any problems with it, but i will be more careful in the future.'], 'loc': '', 'gender': ''}], 'uid': [0, 1]}

# # test_data Example
# {
#     
#     "uid": [0, 1, 2], 
#     
#     "dialog": [["剧烈运动 是 吧"], ["各种 剧烈运动"], ["... 姐 最近 有点 寂寞 过头 了 ..."]], 
#     
#     "responder_profile": {"loc": "海南", "gender": "female", "tag": "美食;宅;80后"}, 
#     
#     "profile": [{"loc": "天津 滨海新区", "gender": "male", "tag": ""}, {"loc": "海南", "gender": "female", "tag": "美食;宅;80后"}, {"loc": "安徽 合肥","gender": "male", "tag": "游戏动漫;双子座;宅;音乐;90后;WOW台服众"}], 
#     
#     "golden_response": ["可不是 ， 我 又 不 像 你 ， 有 女神 。"]
# 
# }

# # Output Example
# {
#     
#     "uid": [0], 
#     
#     "dialog": ["[\"For what it's worth, I don't have a problem with it.\"]"], 
#     
#     "responder_profile": {"loc": "", "gender": "", "tag": "['i did not have any problems with it, but i will be more careful in the future.']"}, 
#     
#     "profile": [{"loc": "", "gender": "", "tag": "[]"}], 
#     
#     "golden_response": "['My apologies.  I did not have any problems with it, but I will be more careful in the future.']"
#     
# }

# # Constant Value

# In[44]:


NPARTITIONS = 1000
INPUT_PATH = "./outputs/persona1.csv"
SCHEDULER = "threads"


# # Imports

# In[64]:


import pandas as pd 
import json
import bz2
from tqdm import tqdm
import glob
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import spacy
import os
import ast
#import neuralcoref


# In[46]:


tqdm.pandas()
ProgressBar().register()
nlp = spacy.load('en_core_web_sm')
#neuralcoref.add_to_pipe(nlp)


# In[56]:


df_input = pd.read_csv(INPUT_PATH)
df_input.head(5)


# In[48]:


def create_json(row):
    return {
        "uid":[0],
        "dialog":row["body"],
        "responder_profile":{
            "loc":"",
            "gender":"",
            "tag":row["parent_persona"]
        },
        "profile":[
            {
                "loc":"",
                "gender":"",
                "tag":row["persona"]
            },
        ],
        "golden_response":row["parent_body"]  
    }


# In[66]:


df_input["body"] = [ast.literal_eval(d) for d in df_input["body"]]
df_input


# In[67]:


df_input["body"] = df_input["body"].progress_apply(lambda x: [x])
df_input["json"] = df_input.progress_apply(create_json, axis=1)
df_input.head(5)


# In[68]:


list_json = df_input["json"].tolist()
with open(f"./outputs/test_data.json", "wt", encoding="utf-8") as file:
    for dic in list_json:
        file.write(str(json.dumps(dic))+"\n")


# In[51]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'script', '*.ipynb'])

