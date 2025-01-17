import benepar, spacy
from tqdm import tqdm 
import json
import pandas as pd
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_json")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    benepar.download('benepar_en3')
    nlp = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    #sentences = open("data/1000_cs.txt").readlines()
    gold_frame = pd.read_json(args.sentence_json)
    sentences = gold_frame["sentence"].values
  #  trees = gold_frame["const_tree"].values
    outputs = list()
    
   
    for _, sentence in enumerate(tqdm(sentences)):
        doc = nlp(sentence.strip())
        sents = list(doc.sents)
        strings = list()
        for sent in sents:
            strings.append(sent._.parse_string)
        outputs.append("(ROOT " + "".join(strings) + ")")
    with open(f"benepar_constituency_parses/{args.output_file}",'w+') as f:
        f.write("\n".join(outputs))

