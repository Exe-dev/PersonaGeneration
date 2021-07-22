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

# In[275]:


import pandas as pd 
import json
import bz2
from tqdm import tqdm
import glob
import dask.dataframe as dd
import spacy
#import neuralcoref


# In[276]:


tqdm.pandas()


# # reddit_data下の全てのjsonファイルを読み込む

# In[277]:


list_bz2_file = glob.glob("./reddit_data/*/*.json")
list_reddit_conversation = []
list_bz2_file


# In[278]:


for i in range(0,len(list_bz2_file)):
    with open(list_bz2_file[i]) as f:
        for line in f.readlines():
            dic=json.loads(line)
            list_reddit_conversation.append(dic)


# In[279]:


df_reddit_conversation = pd.DataFrame(list_reddit_conversation)
df_reddit_conversation


# In[280]:
print(df_reddit_conversation.head(5))
print(df_reddit_conversation.tail(5))
df_reddit_conversation = pd.DataFrame(list_reddit_conversation)
df_reddit_conversation = df_reddit_conversation[df_reddit_conversation["body"]!="[deleted]"]
df_reddit_conversation["body"] = df_reddit_conversation["body"].replace(["&lt","&gt","&amp"],["","",""])
df_reddit_conversation["removed_prefix_parent_id"] = df_reddit_conversation["parent_id"].str.replace("t\d_","")
df_reddit_conversation["parent_body"] = df_reddit_conversation[df_reddit_conversation["removed_prefix_parent_id"]==df_reddit_conversation["id"]]["body"]
df_reddit_conversation["body"] = df_reddit_conversation["body"].str.replace('\"','’')
df_reddit_conversation["parent_body"] = df_reddit_conversation["parent_body"].str.replace('\"','’')
df_reddit_conversation = pd.merge(df_reddit_conversation,df_reddit_conversation[["id","body"]].rename(columns={"id":"parent_id","body":"parent_body"}),left_on="removed_prefix_parent_id",right_on="parent_id").drop(columns=["parent_body_x","parent_id_y"]).rename(columns={"parent_body_y":"parent_body"})
df_reddit_conversation = df_reddit_conversation.dropna(subset=["parent_body"]).sort_values(["author"]).reset_index(drop=True)
df_reddit_conversation = df_reddit_conversation[["body","parent_body","ups","author"]]
df_reddit_conversation


# In[281]:


nlp = spacy.load("en_core_web_sm")


# In[282]:


def CreatePersona(body: str):
    doc = nlp(body.lower())
    # 文ごとに分割
    persona = [str(sentence) for sentence in doc.sents if IsPersona(str(sentence))]
    return persona


# In[283]:


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


# In[284]:


def create_json(dialog, persona0, persona1):
    return {
        "dialog":dialog,
        "profile":[
            {"tag":persona0,
            "loc":"",
            "gender":""},
            {"tag":persona1,
            "loc":"",
            "gender":""}
        ],
        "uid":[0,1]
    }


# In[285]:


df_reddit_conversation["persona"] = df_reddit_conversation["body"].progress_apply(CreatePersona)
df_reddit_conversation["parent_persona"] = df_reddit_conversation["parent_body"].progress_apply(CreatePersona)
df_reddit_conversation = df_reddit_conversation[(df_reddit_conversation.astype(str)["persona"] !="[]")|(df_reddit_conversation.astype(str)["parent_persona"] !="[]")].reset_index(drop=True)
df_reddit_conversation["body"] = df_reddit_conversation["body"].progress_apply(lambda x: [x] )
df_reddit_conversation["parent_body"] = df_reddit_conversation["parent_body"].progress_apply(lambda x: [x] )
df_reddit_conversation["dialog"] = [list(x) for x in zip(df_reddit_conversation["body"].tolist(),df_reddit_conversation["parent_body"].tolist())]
df_reddit_conversation


# In[286]:


list_json = []
for column_name, item in tqdm(df_reddit_conversation.iterrows()):
    list_json.append(create_json(item["dialog"],item["persona"],item["parent_persona"]))
df_reddit_conversation["json"] = list_json
df_reddit_conversation


# In[287]:


df_reddit_conversation.to_csv("persona.csv")


# In[288]:


with open("created_dialogues.json", "wt", encoding="utf-8") as file:
    for dic in list_json:
        file.write(str(json.dumps(dic))+"\n")


# {"dialog": [["没有 钱   万万 不行 ！ ~"], ["现实 就是 如此"]], "profile": [{"tag": ["漫画;旅遊;星座"], "loc": "广东 广州", "gender": "male"}, {"tag": [""], "loc": "", "gender": ""}], "uid": [0, 1]}

# {'dialog': ["For what it's worth, I don't have a problem with it.", 'My apologies.  I did not have any problems with it, but I will be more careful in the future.'], 'profile': [{'tag': ["for what it's worth, i don't have a problem with it."], 'loc': '', 'gender': ''}, {'tag': [' i did not have any problems with it, but i will be more careful in the future.'], 'loc': '', 'gender': ''}], 'uid': [0, 1]}
