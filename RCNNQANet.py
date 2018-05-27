# -*- coding:utf-8 -*-
import tensorflow as tf
import os


class CNNQANet():
    def __init__(self, pretrained_param, encoder_units_number=[300, 100], attention_size=[100], cnn_filters=[300, 200, 100], context_kernal_size=3, question_kernel_size=2, learning_rate=0.001, log_dir='./logs'):
        tf.reset_default_graph()
        self.question = tf.placeholder(shape=[None, None], dtype=tf.int32, name='question')
        self.context = tf.placeholder(shape=[None, None], dtype=tf.int32, name='context')
        self.y_start = tf.placeholder(shape=[None], dtype=tf.int32, name='y_start')
        self.y_end = tf.placeholder(shape=[None], dtype=tf.int32, name='y_end')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        self.global_step = 0
        with tf.variable_scope('embedding', initializer=tf.contrib.layers.xavier_initializer()):
            W = tf.Variable(pretrained_param, trainable=True, dtype=tf.float32)
            self.question_input = tf.nn.embedding_lookup(ids=self.question, params=W)
            self.context_input = tf.nn.embedding_lookup(ids=self.context, params=W)
        
        with tf.variable_scope('context_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            fcell, bcell = self._biGRUs(encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.context_output, self.c_state = tf.nn.bidirectional_dynamic_rnn(inputs=self.context_input, cell_fw=fcell, cell_bw=bcell, dtype=tf.float32, scope=scope)
            self.context_output = tf.concat(self.context_output, axis=-1)
            self.context_output = tf.contrib.layers.layer_norm(self.context_output)
            tf.summary.histogram('context_encoder', self.context_output)
        
        with tf.variable_scope('question_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            fcell, bcell = self._biGRUs(encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.question_output, self.q_state = tf.nn.bidirectional_dynamic_rnn(inputs=self.question_input, cell_fw=fcell, cell_bw=bcell, dtype=tf.float32, scope=scope)
            self.question_output = tf.concat(self.question_output, axis=-1)
            self.question_output = tf.contrib.layers.layer_norm(self.question_output)
            tf.summary.histogram('question_encoder', self.question_output)
        
        with tf.variable_scope('co_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            self.cq_att = self.gated_attention(self.context_output, self.question_output, hidden=attention_size, scope='cq_attention')
            self.cq_att = self._CNN_block(cnn_filters, context_kernal_size, self.cq_att, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('cq_att', self.cq_att)
            
            self.qc_att = self.gated_attention(self.question_output, self.context_output, hidden=attention_size, scope='qc_attention')
            self.qc_att = self._CNN_block(cnn_filters, question_kernel_size, self.qc_att, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('qc_att', self.qc_att)
        with tf.variable_scope('self_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            self.cc_att = self.gated_attention(self.cq_att, self.cq_att, hidden=attention_size, scope='cc_attention')
            self.cc_att = self._CNN_block(cnn_filters, context_kernal_size, self.cc_att, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('cc_att', self.cc_att)
            
            self.qq_att = self.gated_attention(self.qc_att, self.qc_att, hidden=attention_size, scope='qq_attention')
            self.qq_att = self._CNN_block(cnn_filters, question_kernel_size, self.qq_att, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('qq_att', self.qq_att)
        
        with tf.variable_scope('output_layer', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            output_att = self.gated_attention(self.cc_att, self.qq_att, hidden=attention_size, scope='output_attention')
            self.start_output = self._CNN_block(cnn_filters, context_kernal_size, output_att, keep_prob=self.dropout_keep_prob)
            self.end_output = self._CNN_block(cnn_filters, context_kernal_size, output_att, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('start_output', self.start_output)
            tf.summary.histogram('end_output', self.end_output)
        
        with tf.variable_scope('start_decoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            self.y_predict_start = self.start_output
            
            # p_y1=RNN(o_1)
            cell = [self._add_GRU(50, activation=tf.nn.relu), self._add_GRU(1, activation=tf.nn.relu)]
            cell = tf.contrib.rnn.MultiRNNCell(cells=cell, state_is_tuple=True)
            self.y_predict_start, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.y_predict_start, dtype=tf.float32)
            self.y_predict_start = tf.unstack(self.y_predict_start, axis=-1)[0]
            tf.summary.histogram('y_predict_start', self.y_predict_start)
            self.y_predict_start_softmax = tf.nn.softmax(self.y_predict_start)
            self.y_predict_start_index = tf.argmax(self.y_predict_start_softmax, axis=1)
            self.y_start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_predict_start, labels=self.y_start)
            tf.summary.scalar('start_loss', tf.reduce_mean(self.y_start_loss))
        
        with tf.variable_scope('end_decoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            self.y_predict_end = self.end_output
            
            # p_y2=RNN(o_2)
            cell = [self._add_GRU(50, activation=tf.nn.relu), self._add_GRU(1, activation=tf.nn.relu)]
            cell = tf.contrib.rnn.MultiRNNCell(cells=cell, state_is_tuple=True)
            self.y_predict_end, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.y_predict_end, dtype=tf.float32)
            self.y_predict_end = tf.unstack(self.y_predict_end, axis=-1)[0]
            tf.summary.histogram('y_predict_end', self.y_predict_end)
            self.y_predict_end_softmax = tf.nn.softmax(self.y_predict_end)
            self.y_predict_end_index = tf.argmax(self.y_predict_end_softmax, axis=1)
            self.y_end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_predict_end, labels=self.y_end)
            tf.summary.scalar('end_loss', tf.reduce_mean(self.y_end_loss))
        
        with tf.variable_scope('train'):
            self.optimizier = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.total_loss = tf.reduce_mean((self.y_start_loss + self.y_end_loss) / 2.0)
            tf.summary.scalar('total_loss', self.total_loss)
            self.train_op = self.optimizier.minimize(self.total_loss)
        self.init_op = tf.global_variables_initializer()
        self.merge_op = tf.summary.merge_all()
        self.session = tf.Session()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(log_dir,graph=self.session.graph)
    
    def _add_dense_layer(self, inputs, output_shape, drop_keep_prob, act=tf.nn.tanh, use_bias=True):
        output = inputs
        for n in output_shape:
            output = tf.layers.dense(output, n, activation=act, use_bias=use_bias)
            output = tf.nn.dropout(output, drop_keep_prob)
        return output
    
    def gated_attention(self, inputs, memory, hidden, keep_prob=1.0, is_train=None, scope="dot_attention", self_attention=False):
        with tf.variable_scope(scope):
            with tf.variable_scope("attention"):
                # u=W1*u_i
                inputs_ = self._add_dense_layer(inputs, hidden, keep_prob, act=tf.nn.relu, use_bias=False)
                
                # v=W2*v_i
                memory_ = self._add_dense_layer(memory, hidden, keep_prob, act=tf.nn.relu, use_bias=False)
                
                # s=softmax(u*v)
                outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1]))
                logits = tf.nn.softmax(outputs)
                
                # l=s*v_i
                outputs = tf.matmul(logits, memory)
                
                # r=[u_i,l]
                result = tf.concat([inputs, outputs], axis=-1)
            with tf.variable_scope("gate"):
                # g=\sigma(W_g*r)
                gate = self._add_dense_layer(result, [result.shape[-1]], keep_prob, act=tf.nn.sigmoid, use_bias=False)
                # o=g*r
                return result * gate
    
    def _CNN_block(self, hidden_units, kernel_size, inputs, activation=tf.nn.relu, normalize=True, keep_prob=1.0):
        output = inputs
        for n in hidden_units:
            conv1d = tf.layers.Conv1D(filters=n, kernel_size=kernel_size, strides=1, activation=activation, padding='same')
            output = conv1d(output)
        if normalize:
            output = tf.contrib.layers.layer_norm(output)
        output = tf.nn.dropout(output, keep_prob=keep_prob)
        return output
    
    def _biGRUs(self, units_number, activation=tf.nn.tanh, keep_prob=1.0):
        fcell = [self._add_GRU(units_number=n, keep_prob=keep_prob, activation=activation) for n in units_number]
        fcell = tf.contrib.rnn.MultiRNNCell(cells=fcell, state_is_tuple=True)
        bcell = [self._add_GRU(units_number=n, keep_prob=keep_prob, activation=activation) for n in units_number]
        bcell = tf.contrib.rnn.MultiRNNCell(cells=bcell, state_is_tuple=True)
        return fcell, bcell
    
    def _add_GRU(self, units_number, activation=tf.nn.tanh, keep_prob=1.0):
        cell = tf.contrib.rnn.GRUCell(units_number, activation=activation)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell
    
    def build_feed_dict(self, context, question, y_start, y_end, drop_keep_prob=0.7):
        feed_dict = {
            self.question: question,
            self.context: context,
            self.y_start: y_start,
            self.y_end: y_end,
            self.dropout_keep_prob: drop_keep_prob
        }
        return feed_dict
    
    def train(self, context, question, y1, y2, drop_keep_prob=0.85, record_interval=10):
        feed_dict = {
            self.question: question,
            self.context: context,
            self.y_start: y1,
            self.y_end: y2,
            self.dropout_keep_prob: drop_keep_prob
        }
        if self.global_step % record_interval == 0:
            _, loss, summaries = self.session.run([self.train_op, self.total_loss, self.merge_op], feed_dict=feed_dict)
            self.writer.add_summary(summaries, self.global_step)
        else:
            _, loss = self.session.run([self.train_op, self.total_loss], feed_dict=feed_dict)
        self.global_step += 1
        return loss
    
    def evaluate(self, context, question, y1, y2, drop_keep_prob=1.0):
        feed_dict = {
            self.question: question,
            self.context: context,
            self.y_start: y1,
            self.y_end: y2,
            self.dropout_keep_prob: drop_keep_prob
        }
        loss = self.session.run([self.total_loss], feed_dict=feed_dict)
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = loss[0]
        summary_value.tag = 'evaluate_loss'
        self.writer.add_summary(summary, self.global_step)
        return loss
    
    def predict(self, context, question):
        feed_dict = {
            self.question: question,
            self.context: context,
            self.dropout_keep_prob: 1.0
        }
        start, end = self.session.run([self.y_predict_start_index, self.y_predict_end_index], feed_dict=feed_dict)
        return start, end
    
    def load_model(self, model_path='./QAModel'):
        self.saver.restore(self.session, model_path + '/rcnnqanet')
    
    def save_model(self, model_path='./QAModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/rcnnqanet'
        self.saver.save(self.session, model_file)
