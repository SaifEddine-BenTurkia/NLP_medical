import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load("en_core_sci_sm")

def clinical_tokenizer(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"IS_DIGIT": True}, {"TEXT": {"REGEX": "[-/"}}, {"IS_DIGIT": True}],
        [{"TEXT": {"REGEX": "^[A-Z]{2,4}$"}}],
        [{"TEXT": {"REGEX": "^[IVX]+$"}}]
    ]
    matcher.add("CLINICAL_PATTERNS", patterns, greedy="LONGEST")
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return [token.text for token in doc if not token.is_space]

def clinical_lemmatizer_from_tokens(tokens):
    if not isinstance(tokens, list):
        return ""
    
    doc = nlp(" ".join(tokens))  # Recreate a doc from tokens
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_.lower())
    return " ".join(lemmas)