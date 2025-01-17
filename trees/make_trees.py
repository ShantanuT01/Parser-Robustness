from nltk.tree import Tree
from nltk.tree.prettyprinter import TreePrettyPrinter

if __name__ == "__main__":
    tree_string = "(ROOT(SINV (VP (VBZ Thinks)) (NP (NP (DT the) (NNS gods)) (SBAR (S (NP (PRP he)) (VP (VBD did) (RB n't) (VP (VB have) (S (VP (TO to) (VP (VB know) (PP (IN of) (NP (DT this))))))))))) (. .)))"
    tree = Tree.fromstring(tree_string)
    print(tree.leaves())
    pp = TreePrettyPrinter(tree)
    tree.pretty_print()
    with open("plots/stanza_sentence_10-typo.svg",'w+') as f:
        f.write(pp.svg())