import spacy
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    df = pd.read_json("json/gold_standard_2024-2025_adversarial.json")
    sentences = df["sentence"].values
    deplens = pd.read_csv("gold_standard/dependencies_adversarial.csv")
    sent_deps = deplens.groupby("sentence_id")
    rows = list()
    for i, sentence in enumerate(tqdm(sentences)):
        doc = nlp(sentence)
        row = {"sentence_id": i}
        num_of_deps = len(sent_deps.get_group(i))
        sent = list(doc.sents)[0]
        if len(sent) != num_of_deps:
            continue
        for token in sent:
            row_copy = dict(row)
            row_copy["word"] = token.text
            row_copy["deprel"] = token.dep_
            row_copy["head_id"] = token.head.i
            if row_copy["deprel"] == "ROOT":
                row_copy["head"] = "ROOT"
                row_copy["head_id"] = -1
            else:
               # print(token.head.i, list(sent), token.text, len(sent))
                row_copy["head"] = sent[token.head.i].text
            rows.append(row_copy)
    pd.DataFrame(rows).to_csv("spacy_dependency_parses/spacy_lg_gold_standard_adversarial.csv",index=False)