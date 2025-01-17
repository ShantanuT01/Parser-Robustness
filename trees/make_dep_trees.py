import pandas as pd
import spacy
from spacy import displacy


if __name__ == "__main__":
    lg_df = pd.read_csv("results/dependency/lg_adversarial.csv")
    trf_df = pd.read_csv("results/dependency/transformers_adversarial.csv")
    k = trf_df.merge(lg_df,on="sentence_id",suffixes=("_trf","_lg"))
    k = k[k["sentence_id"] == 107]    
    print(k[["las_trf","las_lg"]])
    df = pd.read_json("json/gold_standard_2024-2025_adversarial.json")
    nlp_lg = spacy.load("en_core_web_lg")
    doc = nlp_lg(df["sentence"].values[k["sentence_id"].values[0]])
    options = {
        "compact": True,
        "arrow_spacing":20,
        "distance": 100
    }
    svg = displacy.render(doc, style="dep",options=options)
    with open("plots/lg_dd.svg",'w+') as f:
        f.write(svg)
    
    nlp_trf = spacy.load("en_core_web_trf")
    doc = nlp_trf(df["sentence"].values[k["sentence_id"].values[0]])
    svg = displacy.render(doc, style="dep",options=options)
    with open("plots/trf_dd.svg",'w+') as f:
        f.write(svg)
    