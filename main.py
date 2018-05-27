# -*- coding:utf-8 -*-
import json
from PreProcess import *
from RNNQANet import RNNQANet
from RCNNQANet import RCNNQANet

with open('documents.json', 'r+') as f:
    documents = json.loads(f.read())
with open('training.json', 'r+') as f:
    train = json.loads(f.read())
with open('testing.json', 'r+') as f:
    test = json.loads(f.read())
with open('devel.json', 'r+') as f:
    dev = json.loads(f.read())

processed_data = pre_process(documents, train, dev, test)

max_train_context_length, max_train_question_length = processed_data['train_corpus']['max_length']
training_corpus = processed_data['train_corpus']['data']
max_dev_context_length, max_dev_question_length = processed_data['dev_corpus']['max_length']
dev_corpus = processed_data['dev_corpus']['data']
max_test_context_length, max_test_question_length = processed_data['test_corpus']['max_length']
test_corpus = processed_data['test_corpus']['data']
weight_matrix = processed_data['embedding']['weight_matrix']
word_index = processed_data['embedding']['word_index']
inverted_word_index = processed_data['embedding']['inverted_word_index']

train_context_col, train_question_col = (max_train_context_length, max_train_context_length + max_train_question_length)
dev_context_col, dev_question_col = (max_dev_context_length, max_dev_context_length + max_dev_question_length)
test_context_col, test_question_col = (max_test_context_length, max_test_context_length + max_test_question_length)
print('training corpus context&question length', max_train_context_length, max_train_question_length)
print('dev corpus context&question length', max_dev_context_length, max_dev_question_length)
print('test corpus context&question length', max_test_context_length, max_test_question_length)

RNN_model_params = {
    'encoder_units_number': [300, 100],
    'attention_size': [100],
    'hidden_rnn_size': [100],
    'learning_rate': 0.001,
    'log_dir': './logs/RNNModel',
    'model_path': './RNNQANet'
}

RCNN_model_params = {
    'encoder_units_number': [300, 100],
    'attention_size': [100],
    'cnn_filters': [300, 200, 100],
    'context_kernal_size': 3,
    ' question_kernel_size': 2,
    'learning_rate': 0.001,
    'log_dir': './logs/RCNNModel',
    'model_path': './RCNNQANet'
}


def RNNTrain(model_params, max_epoch=4, max_batch_size=128, checkpoint_interval=10, evaluate_interval=10, evaluate_batch_size=128):
    rnn = RNNQANet(pretrained_embedding=weight_matrix,
                   encoder_units_number=model_params['encoder_units_number'],
                   attention_size=model_params['attention_size'],
                   hidden_rnn_size=model_params['hidden_rnn_size'],
                   learning_rate=model_params['learning_rate'],
                   log_dir=model_params['log_dir'],
                   model_path=model_params['model_path']
                   )
    global_step = 0
    previous_save_loss = np.inf
    for e in range(max_epoch):
        np.random.shuffle(training_corpus)
        epoch_loss = []
        it = 0
        while it < training_corpus.shape[0]:
            max_context_length = max(np.sum(training_corpus[it:it + max_batch_size, :train_context_col] > 0, axis=1))
            b_context = training_corpus[it:it + max_batch_size, :max_context_length]
            max_question_length = max(np.sum(training_corpus[it:it + max_batch_size, train_context_col:train_question_col] > 0, axis=1))
            b_question = training_corpus[it:it + max_batch_size, train_context_col:train_context_col + max_question_length]
            b_y1 = training_corpus[it:it + max_batch_size, -2]
            b_y2 = training_corpus[it:it + max_batch_size, -1]
            loss = rnn.train(context=b_context, question=b_question, y1=b_y1, y2=b_y2, record_interval=1)
            epoch_loss.append(loss)
            print('epoch', e, 'step', global_step, 'iteration', it, 'iteration loss', loss, 'epoch mean loss', np.mean(epoch_loss))
            it += max_batch_size
            global_step += 1
            if global_step % evaluate_interval == 0:
                np.random.shuffle(dev_corpus)
                batch_dev_context_length = max(np.sum(dev_corpus[:evaluate_batch_size, :dev_context_col] > 0, axis=1))
                d_context = dev_corpus[:evaluate_batch_size, :batch_dev_context_length]
                batch_dev_question_length = max(np.sum(dev_corpus[:evaluate_batch_size, dev_context_col:dev_question_col] > 0, axis=1))
                d_question = dev_corpus[:evaluate_batch_size, dev_context_col:dev_context_col + batch_dev_question_length]
                d_y1 = dev_corpus[:evaluate_batch_size, -2]
                d_y2 = dev_corpus[:evaluate_batch_size, -1]
                dev_loss = rnn.evaluate(d_context, d_question, d_y1, d_y2)
                if global_step % checkpoint_interval == 0 and previous_save_loss > dev_loss[0]:
                    rnn.save_model()
                    previous_save_loss = dev_loss[0]
                    print(global_step, 'save model @ val loss', dev_loss[0])
                print(global_step, 'evaluate loss', dev_loss[0])
    return rnn


def RCNNTrain(model_params, max_epoch=4, max_batch_size=128, checkpoint_interval=10, evaluate_interval=10, evaluate_batch_size=128):
    rcnn = RCNNQANet(pretrained_embedding=weight_matrix,
                     encoder_units_number=model_params['encoder_units_number'],
                     attention_size=model_params['attention_size'],
                     cnn_filters=model_params['cnn_filters'],
                     context_kernal_size=model_params['context_kernal_size'],
                     question_kernel_size=model_params['question_kernel_size'],
                     learning_rate=model_params['learning_rate'],
                     log_dir=model_params['log_dir'],
                     model_path=model_params['model_path'])
    global_step = 0
    previous_save_loss = np.inf
    for e in range(max_epoch):
        np.random.shuffle(training_corpus)
        epoch_loss = []
        it = 0
        while it < training_corpus.shape[0]:
            max_context_length = max(np.sum(training_corpus[it:it + max_batch_size, :train_context_col] > 0, axis=1))
            b_context = training_corpus[it:it + max_batch_size, :max_context_length]
            max_question_length = max(np.sum(training_corpus[it:it + max_batch_size, train_context_col:train_question_col] > 0, axis=1))
            b_question = training_corpus[it:it + max_batch_size, train_context_col:train_context_col + max_question_length]
            b_y1 = training_corpus[it:it + max_batch_size, -2]
            b_y2 = training_corpus[it:it + max_batch_size, -1]
            loss = rcnn.train(context=b_context, question=b_question, y1=b_y1, y2=b_y2, record_interval=1)
            epoch_loss.append(loss)
            print('epoch', e, 'step', global_step, 'iteration', it, 'iteration loss', loss, 'epoch mean loss', np.mean(epoch_loss))
            it += max_batch_size
            global_step += 1
            if global_step % evaluate_interval == 0:
                np.random.shuffle(dev_corpus)
                batch_dev_context_length = max(np.sum(dev_corpus[:evaluate_batch_size, :dev_context_col] > 0, axis=1))
                d_context = dev_corpus[:evaluate_batch_size, :batch_dev_context_length]
                batch_dev_question_length = max(np.sum(dev_corpus[:evaluate_batch_size, dev_context_col:dev_question_col] > 0, axis=1))
                d_question = dev_corpus[:evaluate_batch_size, dev_context_col:dev_context_col + batch_dev_question_length]
                d_y1 = dev_corpus[:evaluate_batch_size, -2]
                d_y2 = dev_corpus[:evaluate_batch_size, -1]
                dev_loss = rcnn.evaluate(d_context, d_question, d_y1, d_y2)
                if global_step % checkpoint_interval == 0 and previous_save_loss > dev_loss[0]:
                    rcnn.save_model()
                    previous_save_loss = dev_loss[0]
                    print(global_step, 'save model @ val loss', dev_loss[0])
                print(global_step, 'evaluate loss', dev_loss[0])
    return rcnn


def predict(model, max_batch_size=128):
    predict_context = test_corpus[:, :test_context_col]
    predict_question = test_corpus[:, test_context_col:test_question_col]
    predict_result = []
    it = 0
    qi = 0
    while it < len(predict_question):
        batch_max_context_length = max(np.sum(predict_context[it:it + max_batch_size, :max_test_context_length] > 0, axis=1))
        b_context = predict_context[it:it + max_batch_size, :batch_max_context_length]
        batch_max_question_length = max(np.sum(predict_question[it:it + max_batch_size, :max_test_question_length] > 0, axis=1))
        b_question = predict_question[it:it + max_batch_size, :batch_max_question_length]
        batch_result = np.array(model.predict(context=b_context, question=b_question)).T
        for a, c in zip(batch_result, b_context):
            y1 = min(a)
            y2 = max(a)
            result_token_index = c[y1:y2 + 1]
            result_tokens = ' '.join(lmap(lambda x: inverted_word_index[x], result_token_index))
            predict_result.append((qi, result_tokens))
            qi += 1
        print(it / len(predict_question))
        it += max_batch_size
    return predict_result
