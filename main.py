# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import nltk
import gensim
import itertools
import tflearn as tl
import csv

from RNNQANet import RNNQANet

lmap = lambda func, it: list(map(func, it))
with open('documents.json', 'r+') as f:
    documents = json.loads(f.read())
with open('training.json', 'r+') as f:
    query = json.loads(f.read())
with open('testing.json', 'r+') as f:
    test = json.loads(f.read())
    

tokenizer = nltk.tokenize.SpaceTokenizer()
sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()


# preprocess documents
paragraphs = {}
for d in documents:
    doc_id = d['docid']
    for pi, p in enumerate(d['text']):
        text = p.lower().replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
        text = list(filter(lambda x: len(x.strip()) > 0, tokenizer.tokenize(text)))
        paragraphs[(doc_id, pi)] = text

# preprocess questions
processed_query = []
for q in query:
    docid = q['docid']
    paraid = q['answer_paragraph']
    question_text = q['question'].lower().replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
    question_token = list(filter(lambda x: len(x.strip()) > 0, tokenizer.tokenize(question_text)))
    answer_text = q['text'].lower().replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
    answer_token = list(filter(lambda x: len(x.strip()) > 0, tokenizer.tokenize(answer_text)))
    processed_query.append({'dp_id': (docid, paraid), 'question': question_token, 'answer': answer_token})

# get tokens
doc_tokens = list(paragraphs.values())
query_tokens = lmap(lambda x: x['question'], processed_query)
answer_tokens = lmap(lambda x: x['answer'], processed_query)
all_tokens = doc_tokens + query_tokens + answer_tokens

# word embedding
w2c = gensim.models.Word2Vec(all_tokens, min_count=0, size=100)
w2c.train(all_tokens, epochs=5, total_examples=w2c.corpus_count)
for k, v in paragraphs.items():
    paragraphs[k] = (v, np.array(lmap(lambda x: w2c.wv[x.lower()], v)))


def lookup_answer_index(answer_tokens, doc_tokens):
    doc_length = len(doc_tokens)
    answer_length = len(answer_tokens)
    for i in range(doc_length):
        if doc_length - i < answer_length:
            return (0, 0)
        found = True
        for j in range(answer_length):
            found = answer_tokens[j] in doc_tokens[i + j]
            if not found: break
        if found:
            return (i, i + answer_length - 1)


def crop_pad_question(max_leng, embedding):
    if embedding.shape[0] > max_leng:
        return embedding[:max_leng]
    dim = embedding.shape[1]
    pad_leng = max_leng - embedding.shape[0]
    padded_embedding = np.concatenate((embedding, np.zeros((pad_leng, dim))))
    assert padded_embedding.shape[0] == max_leng
    return padded_embedding


# link embedding with tokens
max_question_length = max(lmap(lambda x: len(x['question']), processed_query))
for q in processed_query:
    #     {'dp_id':(doc_id,paraid),'question':question_token,'answer':answer_token}
    doc_t = paragraphs[q['dp_id']][0]
    answer_t = q['answer']
    query_t = q['question']
    start, end = lookup_answer_index(answer_t, doc_t)
    start_vec = np.zeros((len(doc_t), 1), dtype=np.float32)
    start_vec[start] = 1.0
    end_vec = np.zeros((len(doc_t), 1), dtype=np.float32)
    end_vec[end] = 1.0
    q['answer_span'] = (start_vec, end_vec)
    query_embedding = np.array(lmap(lambda x: w2c.wv[x], query_t))
    q['query_embedding'] = crop_pad_question(max_question_length, query_embedding)

qanet = RNNQANet()

# train
epoch = 10
batch_index = list(paragraphs.keys())
for e in range(epoch):
    np.random.shuffle(batch_index)
    batch_loss = []
    count = 0
    for bk in batch_index:
        context = np.expand_dims(paragraphs[bk][1], axis=0)
        para_questions = list(filter(lambda x: x['dp_id'] == bk, processed_query))
        if len(para_questions) == 0:
            continue
        question = np.array(lmap(lambda x: x['query_embedding'], para_questions))
        y_start = np.array(lmap(lambda x: x['answer_span'][0], para_questions))
        y_end = np.array(lmap(lambda x: x['answer_span'][1], para_questions))
        loss = qanet.train(context, question, y_start, y_end)
        batch_loss.append(loss)
        if count % 100 == 0:
            print(count, 'batch loss', np.mean(batch_loss))
            batch_loss = []
        count += 1

qanet.save_model()

# preprocess for test
concatenate_doc = {}
for docid in range(len(documents)):
    paraids = list(range(len(documents[docid]['text'])))
    doc_text = paragraphs[(docid, 0)][0]
    doc_embedding = paragraphs[(docid, 0)][1]
    for p in paraids[1:]:
        doc_text = doc_text + paragraphs[(docid, p)][0]
        doc_embedding = np.concatenate((doc_embedding, paragraphs[(docid, p)][1]))
    concatenate_doc[docid] = (doc_embedding, doc_text)

processed_test_query = []
for q in test:
    docid = q['docid']
    question_text = q['question'].lower().replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
    question_token = list(filter(lambda x: len(x.strip()) > 0, tokenizer.tokenize(question_text)))
    processed_test_query.append({'id': q['id'], 'docid': docid, 'question': question_token})

max_question_length = max(lmap(lambda x: len(x['question']), processed_query))
for q in processed_test_query:
    #     {'docid':docid,'question':question_token}
    query_t = q['question']
    query_embedding = np.array(lmap(lambda x: w2c.wv[x] if x in w2c.wv else np.random.normal(loc=0, scale=1, size=100), query_t))
    q['query_embedding'] = crop_pad_question(max_question_length, query_embedding)

# predict
test_doc_ids = list(set(lmap(lambda x: x['docid'], processed_test_query)))
final_result = {}
for bk in test_doc_ids:
    context = np.expand_dims(concatenate_doc[bk][0], axis=0)
    questions = list(filter(lambda x: x['docid'] == bk, processed_test_query))
    question_ids = lmap(lambda x: x['id'], questions)
    questions = np.array(lmap(lambda x: x['query_embedding'], questions))
    start, end = qanet.predict(context, questions)
    start = start.ravel()
    end = end.ravel()
    for qid, s, e in zip(question_ids, start, end):
        final_result[qid] = ' '.join(concatenate_doc[bk][1][s:e + 1]).strip()

result = list(final_result.items())
result = sorted(result, key=lambda x: x[0])

with open('result.csv', 'w+') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(result)