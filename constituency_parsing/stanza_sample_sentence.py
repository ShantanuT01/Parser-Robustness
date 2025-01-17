import stanza
import spacy, benepar

if __name__ == "__main__":


    nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'}, use_gpu=False)
    sentence = "As she walked pasty it, the driver's glass started to open."
    doc = nlp(sentence)
    for s in doc.sentences:
        print(str(s.constituency.children[0]))
    print()
    benepar.download('benepar_en3')
    nlp = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    doc = nlp(sentence.strip())
    sents = list(doc.sents)
    for sent in sents:
        print(sent._.parse_string)
