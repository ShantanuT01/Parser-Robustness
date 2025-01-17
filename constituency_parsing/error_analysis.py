from nltk.tree import Tree
import pandas as pd
import re
from collections import Counter, defaultdict
import argparse
from tqdm import tqdm
import json

def split_tag_and_span(subtree_str):
    space_index = subtree_str.index(" ")
    tag = subtree_str[1:space_index].replace("\n"," ").strip()
    subtree_contents = subtree_str[space_index + 1:-1].replace("\n", " ").strip()
    trimmed_subtree = re.sub("\s+", " ", subtree_contents)
    return tag, trimmed_subtree

def get_tree_spans(subtrees):
    return [subtree[1] for subtree in subtrees]

def get_tags(subtrees):
    return [subtree[0] for subtree in subtrees]

def get_phrase_and_symbol_subtrees_as_strings(tree, symbols):
    ret = list()
    for subtree in tree.subtrees():
        subtree_str = str(subtree.flatten())
        tag, subtree_content = split_tag_and_span(subtree_str)
        if tag in symbols:
            ret.append((tag, subtree_content))
    return ret

def get_confusion_matrix(gold_subtrees, test_subtrees):
    mistaken_subtrees = test_subtrees - gold_subtrees
    missing_subtrees = gold_subtrees - test_subtrees
    missing_subtrees = list(missing_subtrees.elements())
    mistaken_subtrees = list(mistaken_subtrees.elements())

    correct_subtrees = gold_subtrees & test_subtrees
    cm = defaultdict(lambda: defaultdict(int))
    for missing in missing_subtrees:
        for mistaken in mistaken_subtrees:
            if (missing[1] == mistaken[1]) and (missing[0] != mistaken[0]):
                cm[missing[0]][mistaken[0]] += 1
    for correct in correct_subtrees:
        cm[correct[0]][correct[0]] += correct_subtrees[correct]
    return dict(cm)

def compute_f1_score_labeled(stats, prefix=""):
    try:
        f1_score = 2*(stats[prefix+"labeled_precision"] * stats[prefix+"labeled_recall"])/(stats[prefix+"labeled_precision"] + stats[prefix+"labeled_recall"])
    except:
        f1_score = 0
    return f1_score

def compare_gold_and_test_trees(gold_tree, test_tree, symbols):
    sentence = split_tag_and_span(str(gold_tree.flatten()))[1]
    
    gold_subtrees = get_phrase_and_symbol_subtrees_as_strings(gold_tree, symbols)
    test_subtrees = get_phrase_and_symbol_subtrees_as_strings(test_tree, symbols)
    gold_spans = get_tree_spans(gold_subtrees)
    test_spans = get_tree_spans(test_subtrees)

    gold_subtrees_counter = Counter(gold_subtrees)
    test_subtrees_counter = Counter(test_subtrees)

    gold_spans_counter = Counter(gold_spans)
    test_spans_counter = Counter(test_spans)

    stats = dict()
    num_gold_subtrees = len(gold_subtrees)
    num_test_subtrees = len(test_subtrees)
    if num_gold_subtrees == 0:
        return stats
    missing_subtrees = gold_subtrees_counter - test_subtrees_counter
    errors = Counter(get_tags(missing_subtrees.keys()))
    errors = dict(errors)
    missing_spans = gold_spans_counter - test_spans_counter
    confusion_matrix = get_confusion_matrix(gold_subtrees_counter, test_subtrees_counter)
    stats["sentence"] = sentence
    labeled_precision = (num_gold_subtrees - missing_subtrees.total()*1.0)
    if num_test_subtrees != 0:
        labeled_precision /= num_test_subtrees
    else:
        labeled_precision = 0
    
    unlabeled_precision = (num_gold_subtrees - missing_spans.total()*1.0)
    if num_test_subtrees != 0:
        unlabeled_precision /= num_test_subtrees
    else:
        unlabeled_precision = 0
    stats["labeled_precision"] = labeled_precision
    stats["labeled_recall"] = (num_gold_subtrees - missing_subtrees.total()*1.0)/num_gold_subtrees
    stats["correct_gold_subtrees"] = num_gold_subtrees - missing_subtrees.total()
    stats["labeled_f1_score"] = compute_f1_score_labeled(stats)
    stats["total_gold_subtrees"] = num_gold_subtrees
    stats["total_test_subtrees"] = num_test_subtrees

    stats["unlabeled_precision"] = unlabeled_precision
    stats["unlabeled_recall"] = (num_gold_subtrees - missing_spans.total()*1.0)/num_gold_subtrees
    stats["unlabeled_f1_score"] = compute_f1_score_labeled(stats, "un")

    stats["correct_gold_spans"] = num_gold_subtrees - missing_spans.total()
    stats["total_gold_spans"] = num_gold_subtrees
    stats["total_test_spans"] = num_test_subtrees
    stats["errors"] = errors
    stats["confusion"] = confusion_matrix
    return stats


def get_trees_from_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return [Tree.fromstring(line.strip()) for line in lines]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file")
    parser.add_argument("--test_file")
    parser.add_argument("--output_file")
    parser.add_argument("--confusion_matrix")
    args = parser.parse_args()

    gold_trees = get_trees_from_file(args.gold_file)
    test_trees = get_trees_from_file(args.test_file)

    with open("constituency_parsing/symbols.json") as f:
        symbols = json.load(f)
    symbols = symbols["symbols"].keys()

    stats = list()
    for i in tqdm(range(len(gold_trees))):
        stat = compare_gold_and_test_trees(gold_trees[i], test_trees[i], symbols)
        if len(stat) != 0:
            stats.append(stat)
    
    stat_frame = pd.json_normalize(stats).fillna(0)  
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for column in stat_frame.columns:
        if column.startswith("confusion."):
            actual, predicted = column[len("confusion."):].split(".")
            confusion_matrix[actual][predicted] = stat_frame[column].sum()
            
    confusion_matrix = pd.DataFrame(confusion_matrix)
    confusion_matrix = confusion_matrix.sort_index()
    confusion_matrix[sorted(confusion_matrix.columns)].T.fillna(0).to_csv(args.confusion_matrix)

    stat_frame.to_csv(args.output_file,index=False)


    