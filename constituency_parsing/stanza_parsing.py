import stanza
from tqdm import tqdm
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_json")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'}, use_gpu=False)
    sentences = pd.read_json(args.sentence_json)["sentence"].values
    outputs = list()
    for _, sentence in enumerate(tqdm(sentences)):
        doc = nlp(sentence.strip())
        parts = list()

        for s in doc.sentences:
            parts.append(str(s.constituency.children[0]))
        outputs.append("(ROOT" + "".join(parts) + ")")
    
    with open(f"stanza_constituency_parses/{args.output_file}","w+") as f:
        f.write("\n".join(outputs))
