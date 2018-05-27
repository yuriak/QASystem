# -*- coding:utf-8 -*-
import json
import nltk
import csv
import pickle
import spacy
from tqdm import tqdm
import sklearn
import collections
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

lmap = lambda func, it: list(map(func, it))
with open('documents.json', 'r+') as f:
    documents = json.loads(f.read())
with open('testing.json', 'r+') as f:
    test = json.loads(f.read())

# need to load spacy large language model
# python3 -m pip install spacy
# python3 -m spacy download en_core_web_lg


nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
n = lambda x: nlp(x)

# gather all the documents which appeared in the test set
test_docids = list(set(lmap(lambda x: x['docid'], test)))
test_docs = list(filter(lambda x: x['docid'] in test_docids, documents))
test_docs = dict(zip(test_docids, test_docs))


def tokenize(text):
    return lmap(lambda x: x.text, n(text))


def tokenize(text):
    return lmap(lambda x: x.text, n(text))


extract_features = lambda x: {
    'text': x.text,
    'pos': x.pos_,
    'tag': x.tag_,
    'dep': x.dep_,
    'digit': x.is_digit,
    'num': x.like_num,
    'ent': x.ent_type_,
}


def in_question(candidate, question):
    return np.sum(lmap(lambda x: x in question, candidate)) > 0


predict = []
retrived_doc = {}
retrived_tfidf = {}
stopwords = nltk.corpus.stopwords.words('english')

for ti, t in tqdm(enumerate(test)):
    question = t['question']
    d = test_docs[t['docid']]
    doc_text = ' '.join(d['text'])
    if t['docid'] not in retrived_doc:
        retrived_doc[t['docid']] = n(doc_text)
    doc = retrived_doc[t['docid']]
    if t['docid'] not in retrived_tfidf:
        sents = list(doc.sents)
        sents_text = lmap(lambda x: x.text, sents)
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords, lowercase=True, min_df=0, max_df=0.95)
        sent_score = tfidf.fit_transform(sents_text)
        retrived_tfidf[t['docid']] = (tfidf, sent_score.toarray(), sents)
    tfidf, sent_score, doc_sents = retrived_tfidf[t['docid']]
    q_score = tfidf.transform([question]).toarray()
    si = np.argsort((np.dot(sent_score, q_score.T).flatten()))[-1:][::-1]
    relative_sent = [doc_sents[i] for i in si]
    sent_tokens = itertools.chain.from_iterable([lmap(lambda x: extract_features(x), list(s)) for s in relative_sent])
    result = []
    q = n(question)
    if in_question(['when', 'what time', 'what day', 'day', 'year', 'month', 'time'], question.lower()):
        for t in sent_tokens:
            t_text = t['text']
            if t['ent'] in ['DATE', 'TIME']:
                if t_text not in result:
                    result.append(t_text)
        if len(result) == 0:
            for t in sent_tokens:
                t_text = t['text']
                if t['digit'] or t['number']:
                    if t_text not in result:
                        result.append(t_text)
        predict.append((ti, ' '.join(result)))
    elif in_question(['who', 'person'], question.lower()):
        for t in sent_tokens:
            t_text = t['text']
            if t['ent'] in ['PERSON', 'ORG', 'NORP']:
                if t_text not in result:
                    result.append(t_text)
        predict.append((ti, ' '.join(result)))
    elif in_question(['where', 'location', 'place', 'city', 'country'], question.lower()):
        for t in sent_tokens:
            t_text = t['text']
            if t['ent'] in ['GPE', 'FACILITY', 'LOC']:
                if t_text not in result:
                    result.append(t_text)
        predict.append((ti, ' '.join(result)))
    elif in_question(['how long', 'how much', 'how far'], question.lower()):
        for t in sent_tokens:
            t_text = t['text']
            if t['ent'] in ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                if t_text not in result:
                    result.append(t_text)
        if len(result) == 0:
            for t in sent_tokens:
                t_text = t['text']
                if t['digit'] or t['num']:
                    if t_text not in result:
                        result.append(t_text)
        predict.append((ti, ' '.join(result)))
    elif in_question(['what'], question.lower()):
        for t in sent_tokens:
            t_text = t['text']
            if len(t['ent']) > 0:
                if t_text not in result:
                    result.append(t_text)
        if len(result) == 0:
            result.extend(lmap(lambda x: x.text, list(relative_sent[0].noun_chunks)))
            result = list(set(result))
        predict.append((ti, ' '.join(result)))
    else:
        for t in sent_tokens:
            t_text = t['text']
            if len(t['ent']) > 0 and t['ent']:
                if t_text not in result:
                    result.append(t_text)
        if len(result) == 0:
            result.extend(lmap(lambda x: x.text, list(relative_sent[0].noun_chunks)))
            result = list(set(result))
        predict.append((ti, ' '.join(result)))

with open('result_rule_based.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerows(predict)
