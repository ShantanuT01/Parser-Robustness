import pandas as pd
import requests
import pyconll
import json
import re
import argparse
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    args = parser.parse_args()
    domain = args.domain

    url = "https://api.github.com/repos/amir-zeldes/gum/contents/dep"
    response = requests.get(url).json()
    sentences = list()
    all_trees = list()
    dependencies = list()
    pos_tags = list()
    pathlib.Path(f"gum/{domain}").mkdir(parents=True, exist_ok=True)

    for file_obj in response:
        file_name = file_obj["name"]
        if file_name.find(f"GUM_{domain}") != -1:
            conllu_path = f"gum/{domain}/{file_name}"
            with open(conllu_path,'w+') as f:
                content = requests.get(file_obj["download_url"]).text
                f.write(content)
            conll = pyconll.load_from_file(conllu_path)
            
            for sentence in conll:
                sentence_dependency = list()
                sentence_tags = list()
                for token in sentence:
                    
                    if token.deprel:
                        row = dict()
                        row["id"] = token.id
                        row["word"] = str(token.form)
                        row["head"] = token.head
                        row["deprel"] = token.deprel
                        sentence_dependency.append(row)
                    pos_row = dict()
                    pos_row["word"] = str(token.form)
                    pos_row["lemma"] = str(token.lemma)
                    pos_row["upos"] = str(token.upos)
                    pos_row["pos"] = str(token.xpos)
                    sentence_tags.append(pos_row)

                sentences.append(sentence.text.strip())
                dependencies.append(sentence_dependency)
                pos_tags.append(sentence_tags)

            const_file = file_name.replace(".conllu",".ptb")
            trees = requests.get(f"https://raw.githubusercontent.com/amir-zeldes/gum/refs/heads/master/const/{const_file}").text
            trees = trees.split("\n\n")
            for tree in trees:
                const_tree = tree.replace("\n"," ").strip()
                const_tree = re.sub("\s+"," ", const_tree)
                all_trees.append(const_tree)
    data = {
        "sentence": sentences,
        "pos_tags": pos_tags,
        "const_tree": all_trees,
        "dependencies": dependencies,

    }
    with open(f"gold_standard/gum_{domain}_corpus.txt",'w+') as f:
        f.write("\n".join(all_trees))
    pd.DataFrame(data).to_json(f"json/gum_{domain}_corpus.json",orient="records",index=False, indent=4)
