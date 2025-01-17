import json
import pandas as pd
import nlpaug
import argparse
import nlpaug.augmenter.word as naw
from collections import Counter
import numpy as np
from tqdm import tqdm



if __name__ == "__main__":
    aug = naw.SpellingAug(aug_min=1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_json")
    parser.add_argument("--output_json")
    parser.add_argument("--gold_standard_output")
    args = parser.parse_args()

    with open(args.sentence_json) as f:
        sentences = json.load(f)
    new_json = list()
    
    for sentence in tqdm(sentences):
        tags = pd.DataFrame(sentence["pos_tags"])
        word_frequencies = Counter(tags[tags["upos"] != "PUNCT"]["word"].values)
        impacted_words = [w for w in word_frequencies if (word_frequencies[w] == 1) and (len(w.strip()) >= 3) and (w.strip() != "didn't")]
        for impacted_word in impacted_words:
            new_words = set(aug.augment(impacted_word, n=5))

            for new_word in new_words:
                new_sentence = sentence["sentence"].replace(impacted_word, new_word)
                new_json.append({
                    "sentence":  sentence["sentence"].replace(impacted_word, new_word),
                    "const_tree": sentence["const_tree"].replace(impacted_word, new_word),
                    "changed_pos": tags[tags["word"] == impacted_word]["upos"].values[0],
                    "original_word": impacted_word,
                    "new_word": new_word
                })
    new_df = pd.DataFrame(new_json)
    with open(args.output_json,'w+') as f:
        json.dump(new_json, f,indent=4)
    print(new_df.head())
    with open(args.gold_standard_output,'w+') as f:
        f.write("\n".join(new_df["const_tree"].values))
    