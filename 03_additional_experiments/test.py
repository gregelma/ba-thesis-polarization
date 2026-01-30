import torch

# print("CUDA verfügbar:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU Name:", torch.cuda.get_device_name(0))

# # Setze Threads für PyTorch Operationen
# torch.set_num_threads(16)          # max Threads für Matrix-Operationen
# torch.set_num_interop_threads(16)

import pandas as pd
import re
import numpy as np

df = pd.read_csv("../04_results/training_history/history_subtask1.csv")


langs = ["amh","arb","ben","deu","eng","fas","hau","hin","ita","khm","mya",
         "nep","ori","pan","pol","rus","spa","swa","tel","tur","urd","zho"]

models = ["bert-base-multilingual-cased", "microsoft_mdeberta-v3-base", "xlm-roberta-base"]

dict_bert, dict_mdeberta, dict_xlm = {}, {}, {}

for m in models:
    out = {}
    for l in langs:
        scores = []
        for s in [1, 2, 3]:
            name = f"mono_{l}_{m}_{s}"
            rows = df[df["model"] == name]
            if len(rows) >= 6:
                scores.append(rows.iloc[5]["val_f1_macro"])      # 6th row (index 5)
        if len(scores) == 3:
            out[l] = sum(scores) / 3

    if m == "bert-base-multilingual-cased":
        dict_bert = out
    elif m == "microsoft_mdeberta-v3-base":
        dict_mdeberta = out
    elif m == "xlm-roberta-base":
        dict_xlm = out

print("bert:", dict_bert)
print("mdeberta:", dict_mdeberta)
print("xlm-r:", dict_xlm)

import json

with open("bert-base-multilingual-cased.json", "w", encoding="utf-8") as f:
    json.dump(dict_bert, f, indent=2, ensure_ascii=False)

with open("microsoft_mdeberta-v3-base.json", "w", encoding="utf-8") as f:
    json.dump(dict_mdeberta, f, indent=2, ensure_ascii=False)

with open("xlm-roberta-base.json", "w", encoding="utf-8") as f:
    json.dump(dict_xlm, f, indent=2, ensure_ascii=False)