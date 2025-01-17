import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("json/gold_standard_2024-2025_adversarial.json")
    sentence_splits = [[2, 27],
    [28, 53],
    [54,105],
    [106 , 135],
    [136 , 170],
    [171 , 209],
    [210 , 233],
    [234 , 243],
    [244 , 269],
    [270 , 300]]
    gold_df = pd.read_json("json/gold_standard_2024-2025.json")
    sentence_counter = 0
    new_frames = list()
    old_frames = list()
    for i in range(10):
        start, end = sentence_splits[i]
        start -= 2
        end -= 2
        subset = df.loc[start:end]
        print(len(subset))
        depframe = pd.DataFrame(gold_df["dependencies"].values[i])
        depframe["id"] = depframe["id"].astype(int) - 1
        depframe["head"] = depframe["head"].astype(int) - 1
        depframe["head_word"] = depframe["word"].values[depframe["head"].values]
        depframe.loc[depframe["head"] == -1, "head_word"] = "ROOT"
        depframe["sentence_id"] = i
        old_frames.append(depframe)
        old = subset["original_word"].values
        new = subset["new_word"].values
        for j in range(len(old)):
            depframe_copy = pd.DataFrame(depframe,copy=True)
            depframe_copy["sentence_id"] = sentence_counter
            sentence_counter += 1
            depframe_copy.loc[depframe_copy["word"] == old[j], "word"] = new[j]
            depframe_copy.loc[depframe_copy["head_word"] == old[j], "head_word"] = new[j]
            new_frames.append(depframe_copy)
    pd.concat(new_frames).to_csv("gold_standard/dependencies_adversarial.csv",index=False)
    pd.concat(old_frames).to_csv("gold_standard/dependencies_gold_standard_2024-2025.csv",index=False)
