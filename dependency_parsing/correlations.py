import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.express as px

def get_mean_dd_vs_uas_and_las(gold_df, entire_df, test_df):
    x = list()
    y_uas = list()
    y_las = list()
    x_dist = list()
    y_uas_bin = list()
    y_las_bin = list()
    y_true_labels = list()
    y_pred_labels = list()
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
        las_score = 0.0
    
        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]) and (y_pred_rel[i] == y_true_rel[i]):
                las_score += 1
                y_las_bin.append(1)
                
            else:
                if (y_pred_rel[i] == "ROOT") and abs(y_pred[i] - y_true[i]) >=6:
                    print("Sentence", s_id)
                y_las_bin.append(0)
        x.append(np.mean(np.abs(y_true - np.arange(len(y_true)))))
        x_dist.extend(np.abs(y_true - np.arange(len(y_true))))
        y_uas_bin.extend((y_true == y_pred).astype(int))
        y_uas.append(accuracy_score(y_true, y_pred))
        y_las.append(las_score/(len(y_pred)))
        y_true_labels.extend(y_true_rel)
        y_pred_labels.extend(y_pred_rel)
       

    return x, y_uas, y_las, x_dist, y_uas_bin, y_true_labels, y_las_bin, y_pred_labels
if __name__ == "__main__":
    gold_df = pd.read_csv("gold_standard/dependencies_adversarial.csv")
    entire_df = pd.read_json("json/gold_standard_2024-2025_adversarial.json")
    test_trf_df = pd.read_csv("spacy_dependency_parses/spacy_transformers_gold_standard_adversarial.csv")
    
    test_lg_df = pd.read_csv("spacy_dependency_parses/spacy_lg_gold_standard_adversarial.csv")
    lg_results = get_mean_dd_vs_uas_and_las(gold_df, entire_df, test_lg_df)
    trf_results = get_mean_dd_vs_uas_and_las(gold_df, entire_df, test_trf_df)
    lg_stats = pd.DataFrame({"dist":lg_results[3], "uas":lg_results[4],"true_rel":lg_results[5], "pred_rel":lg_results[7],"las": lg_results[6]})
    trf_stats = pd.DataFrame({"dist":trf_results[3], "uas":trf_results[4],"true_rel":trf_results[5], "pred_rel":trf_results[7], "las":trf_results[6]})

    rows = list()
    trf_stats_grouped = trf_stats.groupby("dist")
    for dist in trf_stats_grouped:
        y_true = dist[1]["true_rel"]
        y_pred = dist[1]["pred_rel"]
        if len(y_true) >= 100:
            row = dict()
            row["Dependency Distance"] = dist[0]
            print(dist[0])
            print(dist[0], dist[1]["true_rel"].value_counts())
            print(sorted(dist[1]["pred_rel"].unique()))
            print(f1_score(y_true, y_pred, average=None))
            print("-"*30)
            row["Macro-F1 Score"] = f1_score(y_true, y_pred, average="macro")
            row["Model"] = "RoBERTa-base"
            row["UAS"] = dist[1]["uas"].mean()
            row["LAS"] = dist[1]["las"].mean()
            rows.append(row)

    lg_stats_grouped = lg_stats.groupby("dist")
    for dist in lg_stats_grouped:
        y_true = dist[1]["true_rel"]
        y_pred = dist[1]["pred_rel"]
        if len(y_true) >= 100:
            row = dict()
            row["Dependency Distance"] = dist[0]
            print(dist[0])
            print(dist[0], dist[1]["true_rel"].value_counts())
            print(sorted(dist[1]["pred_rel"].unique()))
            print(f1_score(y_true, y_pred, average=None))
            print("-"*30)
            row["Macro-F1 Score"] = f1_score(y_true, y_pred, average="macro")
            row["Model"] = "Large"
            row["UAS"] = dist[1]["uas"].mean()
            row["LAS"] = dist[1]["las"].mean()
            rows.append(row)
   # fig = px.bar(pd.DataFrame(rows),template="plotly_white", x="Dependency Distance",y="UAS",color="Model",barmode="group")
    
    #fig.update_layout(legend=dict(
    #orientation="h",
    #))
    #fig.show()
   # fig.write_image("plots/f1_score_vs_dd.png",scale=3)
    #plt.scatter(lg_results[0],lg_results[1], label="Large UAS")
    #plt.scatter(lg_results[0],lg_results[2], label="Large LAS")
    #plt.scatter(trf_results[0],trf_results[1], label="RoBERTa-base UAS")
    #plt.scatter(trf_results[0],trf_results[2], label="RoBERTa-base LAS")
   
    
    #plt.xlabel("Sentence Mean Dependency Distance")
    #plt.ylabel("Sentence UAS")
    #plt.legend()
    #plt.show()
