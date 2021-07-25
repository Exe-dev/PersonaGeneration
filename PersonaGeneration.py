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

# # Constant Value

# In[82]:


NPARTITIONS = 1000
PATH = "./reddit_data/*/*.json"
SCHEDULER = "threads"


# # Imports

# In[83]:


import pandas as pd 
import json
import bz2
from tqdm import tqdm
import glob
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import spacy
#import neuralcoref


# In[84]:


tqdm.pandas()
ProgressBar().register()
nlp = spacy.load('en_core_web_sm')
#neuralcoref.add_to_pipe(nlp)


# # reddit_data下の全てのjsonファイルを読み込む

# In[85]:


list_bz2_file = glob.glob(PATH)
list_reddit_conversation = []
list_bz2_file


# In[86]:


for i in range(0,len(list_bz2_file)):
    with open(list_bz2_file[i]) as f:
        for line in f.readlines():
            dic=json.loads(line)
            list_reddit_conversation.append(dic)


# In[87]:


df_reddit_conversation = pd.DataFrame(list_reddit_conversation)
df_reddit_conversation.to_csv("AllConversation.csv")
df_reddit_conversation


# # 会話ペアの作成

# In[88]:


df_reddit_conversation = pd.DataFrame(list_reddit_conversation)
df_reddit_conversation = df_reddit_conversation[df_reddit_conversation["body"]!="[deleted]"]
df_reddit_conversation["body"] = df_reddit_conversation["body"].replace(["&lt","&gt","&amp"],["","",""])
df_reddit_conversation["removed_prefix_parent_id"] = df_reddit_conversation["parent_id"].str.replace("t\d_","")
df_reddit_conversation["parent_body"] = df_reddit_conversation[df_reddit_conversation["removed_prefix_parent_id"]==df_reddit_conversation["id"]]["body"]
df_reddit_conversation["body"] = df_reddit_conversation["body"].str.replace('\"','’')
df_reddit_conversation["parent_body"] = df_reddit_conversation["parent_body"].str.replace('\"','’')
df_reddit_conversation = pd.merge(df_reddit_conversation,df_reddit_conversation[["id","body"]].rename(columns={"id":"parent_id","body":"parent_body"}),left_on="removed_prefix_parent_id",right_on="parent_id").drop(columns=["parent_body_x","parent_id_y"]).rename(columns={"parent_body_y":"parent_body"})
df_reddit_conversation = df_reddit_conversation.dropna(subset=["parent_body"]).sort_values(["author"]).reset_index(drop=True)
df_reddit_conversation["original_body"] = df_reddit_conversation["body"]
df_reddit_conversation["original_parent_body"] = df_reddit_conversation["parent_body"]
df_reddit_conversation = df_reddit_conversation[["body","parent_body","original_body","original_parent_body","ups","author"]]
df_reddit_conversation


# In[89]:


def CreatePersona(body: str):
    doc = nlp(body.lower())
    # 文ごとに分割
    persona = [str(sentence) for sentence in doc.sents if IsPersona(str(sentence))]
    return persona


# In[90]:


def IsPersona(sentence: str):
    # 以下の3つの条件を満たすものをペルソナとする
    # 1.文の単語数が4-20の間
    # 2.I か my　が含まれている
    # 3.少なくとも1つの動詞と，名詞，代名詞，形容詞のいずれかが含まれている
    words = [str(word) for word in nlp(sentence.strip())]
    poses = [token.pos_ for token in nlp(sentence.strip())]
    return (
        (4 <= len(words) <= 20)&
        (not set(["i","my"]).isdisjoint(set(words)))&
        (("VERB" in poses)&(not set(["NOUN", "ADJ", "PROPN"]).isdisjoint(set(poses))))
    )


# In[91]:


def create_json(row):
    return {
        "dialog":row["dialog"],
        "profile":[
            {"tag":row["persona"],
            "loc":"",
            "gender":""},
            {"tag":row["parent_persona"],
            "loc":"",
            "gender":""}
        ],
        "uid":[0,1]
    }


# In[92]:


def reference_resolution(sentence):
    return nlp(sentence)._.coref_resolved


# In[93]:


ddf_reddit_conversation = dd.from_pandas(data=df_reddit_conversation, npartitions=NPARTITIONS)
ddf_reddit_conversation["persona"] = ddf_reddit_conversation["original_body"].map(CreatePersona)
ddf_reddit_conversation["parent_persona"] = ddf_reddit_conversation["original_parent_body"].map(CreatePersona)
ddf_reddit_conversation = ddf_reddit_conversation.query("persona.notnull() & parent_persona.notnull()")
#ddf_reddit_conversation["body"] = ddf_reddit_conversation["body"].map(reference_resolution)
#ddf_reddit_conversation["parent_body"] = ddf_reddit_conversation["parent_body"].map(reference_resolution)
df_reddit_conversation = ddf_reddit_conversation.compute(scheduler=SCHEDULER)


# # ペルソナの作成

# In[94]:


df_reddit_conversation = df_reddit_conversation[(df_reddit_conversation.astype(str)["persona"] !="[]")|(df_reddit_conversation.astype(str)["parent_persona"] !="[]")].reset_index(drop=True)
df_reddit_conversation["body"] = df_reddit_conversation["body"].progress_map(lambda x: [x] )
df_reddit_conversation["parent_body"] = df_reddit_conversation["parent_body"].progress_map(lambda x: [x] )
df_reddit_conversation["dialog"] = [list(x) for x in zip(df_reddit_conversation["body"].tolist(),df_reddit_conversation["parent_body"].tolist())]
df_reddit_conversation


# # Json形式の作成

# In[95]:


df_reddit_conversation["json"] = df_reddit_conversation.progress_apply(create_json, axis=1)
df_reddit_conversation


# # Outputs

# In[96]:


df_reddit_conversation.to_csv("persona.csv")


# In[97]:


list_json = df_reddit_conversation["json"].tolist()
with open("created_dialogues.json", "wt", encoding="utf-8") as file:
    for dic in list_json:
        file.write(str(json.dumps(dic))+"\n")


# In[1]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'script', '*.ipynb'])

