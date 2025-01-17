import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300
from sklearn.metrics import f1_score

def plot_bar_chart_of_tag_errors(csv_file):
    df = pd.read_csv(csv_file)
    cols = sorted(list(df.columns))
    x = list()
    y = list()
    for col in cols:
        if col.startswith("errors."):
            y.append(df[col].sum())
            x.append(col.split(".")[-1])
    plt.ylabel("Errors")
    plt.xlabel("Phrase/Sentence Tag")
    plt.bar(x, y)
    plt.show()

def total_stats(csv_file):
    df = pd.read_csv(csv_file)
    
    ret = { 
        "labeled_recall": df["correct_gold_subtrees"].sum()/df["total_gold_subtrees"].sum(),
        "unlabeled_recall":df["correct_gold_spans"].sum()/df["total_gold_spans"].sum(),
        "labeled_precision": df["correct_gold_subtrees"].sum()/df["total_test_subtrees"].sum(),
        "unlabeled_precision":df["correct_gold_spans"].sum()/df["total_test_spans"].sum(),
        "correct_unlabeled_spans":df["correct_gold_spans"].sum(),
        "correct_labeled_spans":df["correct_gold_subtrees"].sum(),
        "total_gold_spans":df["total_gold_spans"].sum(),
        "total_test_spans": df["total_test_spans"].sum(),
    }
    ret["labeled_f1"] = 2*(ret["labeled_recall"] * ret["labeled_precision"])/(ret["labeled_recall"] + ret["labeled_precision"])
    ret["unlabeled_f1"] = 2*(ret["unlabeled_recall"] * ret["unlabeled_precision"])/(ret["unlabeled_recall"] + ret["unlabeled_precision"])
    return ret

def get_table_of_tag_errors(csv_file):
    df = pd.read_csv(csv_file)
    cols = sorted(list(df.columns))
    data = dict()
    for col in cols:
        if col.startswith("errors."):
            data[col.split(".")[-1]] = df[col].sum()
    return data

def plot_confusion_matrix(csv_file, title, save_file):
    df = pd.read_csv(csv_file,index_col=0)
  
    s = sns.heatmap(df, annot=True, fmt='g',cmap="Blues")
    s.set_xlabel("Predicted")
    s.set_ylabel("Actual")
    s.set_title(title)
    plt.savefig(save_file)
    plt.close()

def get_f1_score(csv_file):
    df = pd.read_csv(csv_file,index_col=0)
    y_true = list()
    y_pred = list()
    for index in df.index:
        y_true += ([index] * int(df.loc[index].sum()))
        for column in df.columns:
            y_pred += ([column] * (int(df.loc[index, column])))
    print(f1_score(y_true, y_pred, average=None))
            

if __name__ == "__main__":
    
    benepar = get_table_of_tag_errors("results/constituency/berkeley_on_gold_sentences.csv")
    stanza = get_table_of_tag_errors("results/constituency/stanza_on_gold_sentences.csv")
    print(pd.DataFrame(
        {
            "benepar": benepar,
            "stanza": stanza
        }
    ).to_csv())
    print(total_stats("results/constituency/berkeley_on_gold_sentences.csv"))
    print(total_stats("results/constituency/stanza_on_gold_sentences.csv"))
    print()
    print(total_stats("results/constituency/berkeley_on_adversarial_errors.csv"))
    print(total_stats("results/constituency/stanza_on_adversarial_errors.csv"))
    bdf = pd.read_csv("results/constituency/berkeley_on_adversarial_errors.csv")
    sdf = pd.read_csv("results/constituency/stanza_on_adversarial_errors.csv")

    # columns = "sentence,correct_prop_gold_subtrees,correct_gold_subtrees,total_gold_subtrees,correct_prop_gold_spans,correct_gold_spans,total_gold_spans".split(",")
    # tdf = bdf[columns].merge(sdf[columns],how="left", on="sentence",suffixes=("_benepar","_stanza"))
    # k = tdf[(tdf["correct_prop_gold_spans_benepar"] - tdf["correct_prop_gold_spans_stanza"]) >= 0.3]
    
   # get_f1_score("results/constituency/berkeley_on_gold_sentences_confusion_matrix.csv")
   # get_f1_score("results/constituency/stanza_on_gold_sentences_confusion_matrix.csv")
    get_f1_score("results/constituency/berkeley_on_adversarial_confusion_matrix.csv")
    get_f1_score("results/constituency/stanza_on_adversarial_confusion_matrix.csv")
    print()
   # get_f1_score("results/constituency/berkeley_on_gum_fiction_corpus_confusion_matrix.csv")
   # get_f1_score("results/constituency/stanza_on_gum_fiction_corpus_confusion_matrix.csv")
   # get_f1_score("results/constituency/berkeley_on_gum_news_corpus_confusion_matrix.csv")
   # get_f1_score("results/constituency/stanza_on_gum_news_corpus_confusion_matrix.csv")
    plot_confusion_matrix("results/constituency/berkeley_on_adversarial_confusion_matrix.csv","benepar Confusion Matrix on Adversarial Sentences","plots/benepar_confusion_matrix_adversarial.png")
    plot_confusion_matrix("results/constituency/stanza_on_adversarial_confusion_matrix.csv","Stanza Confusion Matrix on Adversarial Sentences","plots/stanza_confusion_matrix_adversarial.png")
   # print(total_stats("results/constituency/berkeley_on_adversarial_errors.csv"))
   # print(total_stats("results/constituency/stanza_on_adversarial_errors.csv"))

    df = pd.read_csv("results/constituency/berkeley_on_adversarial_errors.csv")
    df["attacked_pos"] = pd.read_json("json/gold_standard_2024-2025_adversarial.json")["changed_pos"].to_list()
    error_cols = [col for col in df.columns if col.find("errors.") > -1]
   # error_cols = ["labeled_precision", "labeled_recall", "labeled_f1_score"]
    print(df.groupby("attacked_pos")[error_cols].mean().round(2))

