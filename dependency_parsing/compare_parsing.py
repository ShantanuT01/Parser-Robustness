import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from collections import defaultdict, Counter

matplotlib.rcParams["figure.dpi"] = 300


def plot_confusion_matrix(df, title, save_file):
    plt.figure(figsize=(15,15))  
    s = sns.heatmap(df, annot=True, fmt='g',cmap="Blues",linewidths=0.5)
    s.set_xlabel("Predicted")
    s.set_ylabel("Actual")
    s.set_title(title)
    plt.savefig(save_file)
    plt.close()


if __name__ == "__main__":
    pred_df = pd.read_csv("spacy_dependency_parses/spacy_lg_gold_standard_adversarial.csv")
    true_df = pd.read_csv("gold_standard/dependencies_adversarial.csv")
    true_df = true_df[true_df["sentence_id"].isin(pred_df["sentence_id"].unique())]
    pred_sentences = pred_df.groupby("sentence_id")
    true_sentences = true_df.groupby("sentence_id")
    print(len(true_sentences))
    y_true = list()
    y_pred = list()
    y_true_rel = list()
    y_pred_rel = list()
    sentence_level_uas = list()
    sentence_level_las = list()
    rows = list()
    for i in pred_sentences.groups.keys():
        pred_dep = pred_sentences.get_group(i)
        true_dep = true_sentences.get_group(i)
        row = dict()
        row["sentence_id"] = i
        y_pred.extend(pred_dep["head"].to_list())
        y_true.extend(true_dep["head_word"].to_list())
        y_true_rel.extend(true_dep["deprel"].to_list())
        y_pred_rel.extend(pred_dep["deprel"].to_list())
        uas_score = 0.0
        las_score = 0.0
        for j in range(len(pred_dep)):
            if (pred_dep["head"].values[j] == true_dep["head_word"].values[j]) and (pred_dep["deprel"].values[j] == true_dep["deprel"].values[j]):
                las_score += 1
                uas_score += 1
            elif (pred_dep["head"].values[j] == true_dep["head_word"].values[j]):
                uas_score += 1
        sentence_level_uas.append(uas_score/len(true_dep))
        sentence_level_las.append(las_score/len(true_dep))
        row["total_deps"] = len(true_dep)
        row["uas_correct"] = uas_score
        row["uas"] = uas_score/len(true_dep)
        row["las_correct"] = las_score
        row["las"] = las_score/len(true_dep)
        rows.append(row)
    uas = accuracy_score(y_true, y_pred)
    count = 0.0
    for i in range(len(y_pred)):
        if (y_pred[i] == y_true[i]) and (y_true_rel[i] == y_pred_rel[i]):
            count += 1
    las = count/len(y_pred)
    label_f1 = f1_score(y_true_rel, y_pred_rel,average="macro")
    print("UAS:",uas)
    print("LAS:",las)
    print("Label-F1:", label_f1)
    print("Perfect Sentences UAS:", sentence_level_uas.count(1)/len(sentence_level_uas))
    print("Perfect Sentences LAS:", sentence_level_las.count(1)/len(sentence_level_las))
    cm_data = defaultdict(lambda: defaultdict(int))
    for i in range(len(y_true_rel)):
        actual = y_true_rel[i]
        predicted = y_pred_rel[i]
        cm_data[actual][predicted] += 1
    cm = pd.DataFrame(cm_data).T.fillna(0)
    cm = cm.sort_index()
    cm = cm.sort_index(axis=1)
    cm.to_csv("results/dependency/lg_confusion_matrix.csv")
   # print(cm_data["acomp"])
    plot_confusion_matrix(cm, "Large Confusion Matrix","plots/spacy_lg_confusion_matrix_adversarial.png")

    possible_rels = Counter(y_true_rel).keys()
    predicted_rels = Counter(y_pred_rel).keys()
    impossible_rels = predicted_rels - possible_rels
    magnitude = sum(Counter(y_pred_rel)[key] for key in list(impossible_rels))
    print("Guessed these impossible rels:", impossible_rels)
    print("Number of impossible rels:", len(impossible_rels))
    print("Guessed total:", magnitude)
    print("Total Number of Deprels:", len(y_true_rel))
    print("Mean UAS:", np.mean(sentence_level_uas))
    print("Mean LAS:", np.mean(sentence_level_las))
   # pd.DataFrame(rows).to_csv("results/dependency/lg_adversarial.csv",index=False)
