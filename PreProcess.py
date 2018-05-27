# -*- coding:utf-8 -*-
import numpy as np
import spacy
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.bm25 import BM25

lmap = lambda func, it: list(map(func, it))
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
n = lambda x: nlp(x, disable=['tagger', 'ner', 'textcat', 'parser'])


def lookup_answer_index(answer_tokens, doc_tokens):
    doc_length = len(doc_tokens)
    answer_length = len(answer_tokens)
    for i in range(doc_length):
        if doc_length - i < answer_length:
            return (0, 0)
        found = True
        for j in range(answer_length):
            found = answer_tokens[j].lower() == doc_tokens[i + j].lower()
            if not found: break
        if found:
            return (i, i + answer_length - 1)
    return (0, 0)


def crop_pad(max_leng, word_index):
    if len(word_index) > max_leng:
        return word_index[:max_leng]
    pad_leng = max_leng - len(word_index)
    word_index = word_index + [0] * pad_leng
    assert len(word_index) == max_leng
    return word_index


def tokenize(text):
    return lmap(lambda x: x.text, n(text))


def pre_process(documents, train, dev, test):
    training_corpus = []
    dev_corpus = []
    weight_matrix = []
    word_index = collections.defaultdict()
    test_corpus = []
    word_index.setdefault('', len(word_index))
    weight_matrix.append(np.zeros(300, dtype=np.float32))
    for d in documents:
        pure_paras = d['text']
        pure_paras_text = lmap(lambda x: x.replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', ''), pure_paras)
        docid = d['docid']
        paras_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, lowercase=True, min_df=0)
        paras_score = paras_tfidf.fit_transform(pure_paras_text).toarray()
        doc_para_bm25 = BM25(lmap(lambda x: lmap(lambda y: y.text, n(x)), pure_paras))
        doc_para_bm25_average_idf = sum(map(lambda k: float(doc_para_bm25.idf[k]), doc_para_bm25.idf.keys())) / len(doc_para_bm25.idf.keys())
        for pid, p in enumerate(pure_paras_text):
            para_train_questions = list(filter(lambda x: x['docid'] == docid and x['answer_paragraph'] == pid, train))
            para_dev_questions = list(filter(lambda x: x['docid'] == docid and x['answer_paragraph'] == pid, dev))
            para_text = p.replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
            para = n(para_text)
            p_sentences = list(para.sents)
            para_tokens = list(para)
            for t in para_tokens:
                t_text = t.text
                if t.is_oov:
                    t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                if t_text not in word_index:
                    word_index.setdefault(t_text, len(word_index))
                    weight_matrix.append(t.vector)
            for qid, q in enumerate(para_train_questions):
                question_text = q['question'].replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
                question = n(question_text)
                question_tokens = list(question)
                question_token_index = []
                for t in question_tokens:
                    t_text = t.text
                    if t.is_oov:
                        t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                    if t.text not in word_index:
                        word_index.setdefault(t_text, len(word_index))
                        weight_matrix.append(t.vector)
                    question_token_index.append(word_index[t_text])
                answer_text = q['text'].replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
                answer = n(answer_text)
                answer_tokens = list(answer)
                answer_token_index = []
                for t in answer_tokens:
                    t_text = t.text
                    if t.is_oov:
                        t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                    if t.text not in word_index:
                        word_index.setdefault(t_text, len(word_index))
                        weight_matrix.append(t.vector)
                    answer_token_index.append(word_index[t_text])
                answer_token_text = lmap(lambda x: x.text, answer_tokens)
                for s in p_sentences:
                    target_sent = s
                    sent_token_text = lmap(lambda x: x.text, target_sent)
                    y1, y2 = lookup_answer_index(answer_tokens=answer_token_text, doc_tokens=sent_token_text)
                    sent_token_index = []
                    for t in target_sent:
                        t_text = t.text
                        if t.is_oov:
                            t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                        if t.text not in word_index:
                            word_index.setdefault(t_text, len(word_index))
                            weight_matrix.append(t.vector)
                        sent_token_index.append(word_index[t_text])
                    if lmap(lambda x: x.lower(), answer_token_text) == lmap(lambda x: x.lower(), sent_token_text)[y1:y2 + 1]:
                        training_corpus.append((sent_token_index, question_token_index, y1, y2))
                        break
            for qid, q in enumerate(para_dev_questions):
                question_text = q['question'].replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
                question = n(question_text)
                question_tokens = list(question)
                question_token_index = []
                for t in question_tokens:
                    t_text = t.text
                    if t.is_oov:
                        t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                    if t.text not in word_index:
                        word_index.setdefault(t_text, len(word_index))
                        weight_matrix.append(t.vector)
                    question_token_index.append(word_index[t_text])
                answer_text = q['text'].replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
                answer = n(answer_text)
                answer_tokens = list(answer)
                answer_token_index = []
                for t in answer_tokens:
                    t_text = t.text
                    if t.is_oov:
                        t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                    if t.text not in word_index:
                        word_index.setdefault(t_text, len(word_index))
                        weight_matrix.append(t.vector)
                    answer_token_index.append(word_index[t_text])
                answer_token_text = lmap(lambda x: x.text, answer_tokens)
                for s in p_sentences:
                    target_sent = s
                    sent_token_text = lmap(lambda x: x.text, target_sent)
                    y1, y2 = lookup_answer_index(answer_tokens=answer_token_text, doc_tokens=sent_token_text)
                    sent_token_index = []
                    for t in target_sent:
                        t_text = t.text
                        if t.is_oov:
                            t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                        if t.text not in word_index:
                            word_index.setdefault(t_text, len(word_index))
                            weight_matrix.append(t.vector)
                        sent_token_index.append(word_index[t_text])
                    if lmap(lambda x: x.lower(), answer_token_text) == lmap(lambda x: x.lower(), sent_token_text)[y1:y2 + 1]:
                        dev_corpus.append((sent_token_index, question_token_index, y1, y2))
                        break
        test_questions = list(filter(lambda x: x['docid'] == docid, test))
        for t in test_questions:
            tid = t['id']
            question_text = t['question'].replace('"', '').replace('`', '').replace('``', '').replace("''", '').replace('``', '')
            question = n(question_text)
            question_tokens = list(question)
            question_token_index = []
            for t in question_tokens:
                t_text = t.text
                if t.is_oov:
                    t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                if t.text not in word_index:
                    word_index.setdefault(t_text, len(word_index))
                    weight_matrix.append(t.vector)
                question_token_index.append(word_index[t_text])
            q_para_tfidf_score = paras_tfidf.transform([question_text]).toarray()
            para_tfidf_sim = np.dot(paras_score, q_para_tfidf_score.T).flatten()
            para_tfidf_sim = (para_tfidf_sim - np.mean(para_tfidf_sim)) / (np.std(para_tfidf_sim) + 1e-10)

            para_bm25_sim = np.array(doc_para_bm25.get_scores(lmap(lambda x: x.text, n(question_text)), doc_para_bm25_average_idf))
            para_bm25_sim = (para_bm25_sim - np.mean(para_bm25_sim)) / (np.std(para_bm25_sim) + 1e-10)

            para_total_sim = para_bm25_sim * 0.5 + para_tfidf_sim * 0.5

            target_para_index = np.argmax(para_total_sim)
            target_para = pure_paras_text[target_para_index]
            target_para_sents = list(n(target_para).sents)
            target_para_sents_text = lmap(lambda x: x.text, target_para_sents)

            target_para_sents_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, lowercase=True, min_df=0)
            target_para_sents_tfidf_score = target_para_sents_tfidf.fit_transform(target_para_sents_text).toarray()
            q_sent_score = target_para_sents_tfidf.transform([question_text]).toarray()
            sent_tfidf_sim = np.dot(target_para_sents_tfidf_score, q_sent_score.T).flatten()

            embed_sim = np.array(lmap(lambda x: x.similarity(question), target_para_sents))

            sent_bm25 = BM25(lmap(lambda x: lmap(lambda y: y.text, list(x)), list(n(target_para).sents)))
            sent_bm25_average_idf = sum(map(lambda k: float(sent_bm25.idf[k]), sent_bm25.idf.keys())) / len(sent_bm25.idf.keys())
            sent_bm25_sim = np.array(sent_bm25.get_scores(lmap(lambda x: x.text, question_tokens), sent_bm25_average_idf))

            sent_tfidf_sim = (sent_tfidf_sim - np.mean(sent_tfidf_sim)) / (np.std(sent_tfidf_sim) + 1e-10)
            embed_sim = (embed_sim - np.mean(embed_sim)) / (np.std(embed_sim) + 1e-10)
            sent_bm25_sim = (sent_bm25_sim - np.mean(sent_bm25_sim)) / (np.std(sent_bm25_sim) + 1e-10)
            total_sim = (sent_tfidf_sim + embed_sim + sent_bm25_sim) / 3
            sent_index = np.argmax(total_sim)
            target_sent = target_para_sents[sent_index]
            sent_token_index = []
            for t in target_sent:
                t_text = t.text
                if t.is_oov:
                    t.vocab.set_vector(t_text, np.random.uniform(-1, 1, (300,)))
                if t.text not in word_index:
                    word_index.setdefault(t_text, len(word_index))
                    weight_matrix.append(t.vector)
                sent_token_index.append(word_index[t_text])
            test_corpus.append((tid, sent_token_index, question_token_index))

    weight_matrix = np.array(weight_matrix)
    invert_word_index = {v: k for k, v in word_index.items()}

    train_context = lmap(lambda x: x[0], training_corpus)
    train_questions = lmap(lambda x: x[1], training_corpus)
    train_y1 = lmap(lambda x: x[2], training_corpus)
    train_y2 = lmap(lambda x: x[3], training_corpus)
    dev_context = lmap(lambda x: x[0], dev_corpus)
    dev_questions = lmap(lambda x: x[1], dev_corpus)
    dev_y1 = lmap(lambda x: x[2], dev_corpus)
    dev_y2 = lmap(lambda x: x[3], dev_corpus)
    test_context = lmap(lambda x: x[1], test_corpus)
    test_questions = lmap(lambda x: x[2], test_corpus)

    max_train_context_length = max(lmap(lambda x: len(x), train_context))
    max_train_question_length = max(lmap(lambda x: len(x), train_questions))
    max_dev_context_length = max(lmap(lambda x: len(x), dev_context))
    max_dev_question_length = max(lmap(lambda x: len(x), dev_questions))
    max_test_context_length = max(lmap(lambda x: len(x), test_context))
    max_test_question_length = max(lmap(lambda x: len(x), test_questions))

    train_context = np.array(lmap(lambda x: crop_pad(max_train_context_length, x), train_context))
    train_questions = np.array(lmap(lambda x: crop_pad(max_train_question_length, x), train_questions))
    dev_context = np.array(lmap(lambda x: crop_pad(max_dev_context_length, x), dev_context))
    dev_questions = np.array(lmap(lambda x: crop_pad(max_dev_question_length, x), dev_questions))
    test_context = np.array(lmap(lambda x: crop_pad(max_test_context_length, x), test_context))
    test_questions = np.array(lmap(lambda x: crop_pad(max_test_question_length, x), test_questions))
    train_y = np.array([train_y1, train_y2]).T
    dev_y = np.array([dev_y1, dev_y2]).T

    concatenated_training_corpus = np.concatenate((train_context, train_questions, train_y), axis=-1)
    concatenated_dev_corpus = np.concatenate((dev_context, dev_questions, dev_y), axis=-1)
    concatenated_test_corpus = np.concatenate((test_context, test_questions), axis=-1)

    return {
        'train_corpus': {
            'max_length': (max_train_context_length, max_train_question_length),
            'data': concatenated_training_corpus
        },
        'dev_corpus': {
            'max_length': (max_dev_context_length, max_dev_question_length),
            'data': concatenated_dev_corpus
        },
        'test_corpus': {
            'max_length': (max_test_context_length, max_test_question_length),
            'data': concatenated_test_corpus
        },
        'embedding': {
            'word_index': word_index,
            'inverted_word_index': invert_word_index,
            'weight_matrix': weight_matrix,
        }
    }
