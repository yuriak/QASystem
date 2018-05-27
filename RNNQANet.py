# -*- coding:utf-8 -*-
import tensorflow as tf
import os


class RNNQANet():
    def __init__(self, pretrained_embedding, encoder_units_number=[300, 100], attention_size=[100], hidden_rnn_size=[100], learning_rate=0.001, log_dir='./logs', model_path='./RNNQANet'):
        tf.reset_default_graph()
        self.question = tf.placeholder(shape=[None, None], dtype=tf.int32, name='question')
        self.context = tf.placeholder(shape=[None, None], dtype=tf.int32, name='context')
        self.y_start = tf.placeholder(shape=[None], dtype=tf.int32, name='y_start')
        self.y_end = tf.placeholder(shape=[None], dtype=tf.int32, name='y_end')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        self.global_step = 0
        self.model_path=model_path
        with tf.variable_scope('embedding', initializer=tf.contrib.layers.xavier_initializer()):
            W = tf.Variable(pretrained_embedding, trainable=True, dtype=tf.float32, name='W_emb')
            self.question_input = tf.nn.embedding_lookup(ids=self.question, params=W)
            self.context_input = tf.nn.embedding_lookup(ids=self.context, params=W)
        
        with tf.variable_scope('context_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            # Using Bidirectional RNN to encode context
            # u_c=BiRNN(e_c)
            fcell, bcell = self._biGRUs(encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.context_output, self.c_state = tf.nn.bidirectional_dynamic_rnn(inputs=self.context_input, cell_fw=fcell, cell_bw=bcell, dtype=tf.float32, scope=scope)
            self.context_output = tf.concat(self.context_output, axis=-1)
            self.context_output = tf.contrib.layers.layer_norm(self.context_output)
            tf.summary.histogram('context_encoder', self.context_output)
        
        with tf.variable_scope('question_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            # Using Bidirectional RNN to encode question
            # u_q=BiRNN(e_q)
            fcell, bcell = self._biGRUs(encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.question_output, self.q_state = tf.nn.bidirectional_dynamic_rnn(inputs=self.question_input, cell_fw=fcell, cell_bw=bcell, dtype=tf.float32, scope=scope)
            self.question_output = tf.concat(self.question_output, axis=-1)
            self.question_output = tf.contrib.layers.layer_norm(self.question_output)
            tf.summary.histogram('question_encoder', self.question_output)
        
        with tf.variable_scope('co_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            # Co-attention: context -> question
            # a_cq=attn_biRNN(u_c,u_q)
            self.cq_att = self.gated_attention(self.context_output, self.question_output, hidden=attention_size, scope='cq_attention')
            cqfcell, cqbcell = self._biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.cq_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.cq_att, cell_fw=cqfcell, cell_bw=cqbcell, dtype=tf.float32, scope=tf.get_variable_scope().name + '/cq_attention_rnn')
            self.cq_att = tf.concat(self.cq_att, axis=-1)
            self.cq_att = tf.contrib.layers.layer_norm(self.cq_att)
            tf.summary.histogram('cq_att', self.cq_att)
            
            # Co-attention: question -> context
            # a_qc=attn_biRNN(u_q,u_c)
            self.qc_att = self.gated_attention(self.question_output, self.context_output, hidden=attention_size, scope='qc_attention')
            qcfcell, qcbcell = self._biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.qc_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.qc_att, cell_fw=qcfcell, cell_bw=qcbcell, dtype=tf.float32, scope=tf.get_variable_scope().name + '/qc_attention_rnn')
            self.qc_att = tf.concat(self.qc_att, axis=-1)
            self.qc_att = tf.contrib.layers.layer_norm(self.qc_att)
            tf.summary.histogram('qc_att', self.qc_att)
        
        with tf.variable_scope('self_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            # Self-attention: a_cq -> a_cq
            # a_cc=attn_biRNN(a_cq,a_cq)
            self.cc_att = self.gated_attention(self.cq_att, self.cq_att, hidden=attention_size, scope='cc_attention')
            ccfcell, ccbcell = self._biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.cc_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.cc_att, cell_fw=ccfcell, cell_bw=ccbcell, dtype=tf.float32, scope=tf.get_variable_scope().name + '/cc_attention_rnn')
            self.cc_att = tf.concat(self.cc_att, axis=-1)
            self.cc_att = tf.contrib.layers.layer_norm(self.cc_att)
            tf.summary.histogram('cc_att', self.cc_att)
            
            # Self-attention: a_qc -> a_qc
            # a_qq=attn_biRNN(a_qc,a_qc)
            self.qq_att = self.gated_attention(self.qc_att, self.qc_att, hidden=attention_size, scope='qq_attention')
            qqfcell, qqbcell = self._biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.qq_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.qq_att, cell_fw=qqfcell, cell_bw=qqbcell, dtype=tf.float32, scope=tf.get_variable_scope().name + '/qq_attention_rnn')
            self.qq_att = tf.concat(self.qq_att, axis=-1)
            self.qq_att = tf.contrib.layers.layer_norm(self.qq_att)
            tf.summary.histogram('qq_att', self.qq_att)
        
        with tf.variable_scope('output_layer', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            # Output-attention: a_cq -> a_qc
            # a_o1,a_o2=attn_biRNN(a_cc,a_qq)
            output_att = self.gated_attention(self.cc_att, self.qq_att, hidden=attention_size, scope='output_attention')
            output_att = tf.concat([self.context_output, output_att], axis=-1)
            fcell, bcell = self._biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            output, _ = tf.nn.bidirectional_dynamic_rnn(inputs=output_att, cell_fw=fcell, cell_bw=bcell, dtype=tf.float32, scope=scope)
            
            # use forward output to generate y1
            self.start_output = output[0]
            self.start_output = tf.contrib.layers.layer_norm(self.start_output)
            tf.summary.histogram('start_output', self.start_output)
            
            # use backward output to generate y2
            self.end_output = output[1]
            self.end_output = tf.contrib.layers.layer_norm(self.end_output)
            tf.summary.histogram('end_output', self.end_output)
        
        with tf.variable_scope('start_decoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            # p_y1=RNN(a_o1)
            cell = [self._add_GRU(50, activation=tf.nn.relu), self._add_GRU(25, activation=tf.nn.relu), self._add_GRU(1, activation=tf.nn.relu)]
            cell = tf.contrib.rnn.MultiRNNCell(cells=cell, state_is_tuple=True)
            self.y_predict_start, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.start_output, dtype=tf.float32)
            self.y_predict_start = tf.unstack(self.y_predict_start, axis=-1)[0]
            tf.summary.histogram('y_predict_start', self.y_predict_start)
            self.y_predict_start_softmax = tf.nn.softmax(self.y_predict_start)
            self.y_predict_start_index = tf.argmax(self.y_predict_start_softmax, axis=1)
            self.y_start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_predict_start, labels=self.y_start)
            tf.summary.scalar('start_loss', tf.reduce_mean(self.y_start_loss))
        
        with tf.variable_scope('end_decoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            # p_y2=RNN(a_o2)
            cell = [self._add_GRU(50, activation=tf.nn.relu), self._add_GRU(25, activation=tf.nn.relu), self._add_GRU(1, activation=tf.nn.relu)]
            cell = tf.contrib.rnn.MultiRNNCell(cells=cell, state_is_tuple=True)
            self.y_predict_end, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.end_output, dtype=tf.float32)
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
    
    def _biGRUs(self, units_number, activation=tf.nn.relu, keep_prob=1.0):
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
    
    def load_model(self):
        self.saver.restore(self.session, self.model_path + '/rnnqanet')
    
    def save_model(self, ):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        model_file = self.model_path + '/rnnqanet'
        self.saver.save(self.session, model_file)
