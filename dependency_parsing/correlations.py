import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == "__main__":
    gold_df = pd.read_csv("gold_standard/dependencies_adversarial.csv")
    entire_df = pd.read_json("json/gold_standard_2024-2025_adversarial.json")
    test_df = pd.read_csv("spacy_dependency_parses/spacy_transformers_gold_standard_adversarial.csv")
    x = list()
    y = list()
    stats = defaultdict(lambda: {"y_true": list(), "y_pred": list()})
    for s_id in test_df["sentence_id"].unique():
        gold_sentence = gold_df[gold_df["sentence_id"] == s_id]
        test_sentence = test_df[test_df["sentence_id"] == s_id]
        
        attacked_word = entire_df["new_word"].values[s_id]
       # print(s_id, attacked_word)
       # print(gold_sentence["word"].tolist())
        attacked_word_index = gold_sentence["word"].tolist().index(attacked_word)
        attacked_deprel = gold_sentence["deprel"].values[attacked_word_index]
        y_true = gold_sentence["head"].values
        y_true_rel = gold_sentence["deprel"].values
        for i in range(len(y_true)):
            if y_true[i] == -1:
                y_true[i] = i
        
        y_pred = test_sentence["head_id"].values
        y_pred_rel = test_sentence["deprel"].values
        for i in range(len(y_pred)):
            if y_pred[i] == -1:
                y_pred[i] = i
        stats[attacked_deprel]["y_true"].extend(y_true)
        stats[attacked_deprel]["y_pred"].extend(y_pred)
    for key in stats:
        y_true = stats[key]["y_true"]
        y_pred = stats[key]["y_pred"]
        print(key, accuracy_score(y_true, y_pred))