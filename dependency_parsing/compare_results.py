import pandas as pd




if __name__ == "__main__":
    sentences = pd.read_json("json/gold_standard_2024-2025_adversarial.json")
    df = pd.read_csv("results/dependency/lg_adversarial.csv")
    changed_pos = sentences["changed_pos"].values[df["sentence_id"].values]
    df["attacked_pos"] = list(changed_pos)
    print(df.groupby("attacked_pos")[["uas","las"]].mean())