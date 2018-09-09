# -*- coding: utf-8 -*-

import tensorflow as tf

class SpellChecker(object):
    """
    base class of SpellChecker
    """
    def __init__(self,hps):
        """set the hyperparameters"""
        self.mode = hps.mode
        self.vocab_size = hps.vocab_size
        self.hidden_size = hps.hidden_size
        self.batch_size = hps.batch_size
        self.em_size = hps.em_size
        self.keep_ratio = hps.keep_ratio
        self.lr = hps.lr
        #self.pinyin_size = 2051
        self.pinyin_size = 2084 # full size
        self.max_gradient_norm = hps.max_gradient_norm
        self.head_num = hps.head_num

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            # build graph
            if self.mode == 'train':
                self.build_graph()
                self.build_loss()
                self.setup_train()
                self.setup_summary()
            elif self.mode == 'decode':
                self.build_graph()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep = hps.max_to_keep) 


    def init_embedding(self,pinyin_emb = True,word_emb = True):
        """Initialize the  embedding."""
        if pinyin_emb:
            self.pinyin_embedding = tf.Variable(tf.truncated_normal(
                    [self.pinyin_size, self.em_size]), name="pinyin_embedding")

        if word_emb:
            self.word_embedding = tf.Variable(tf.truncated_normal(
                    [self.vocab_size, self.em_size]), name="word_embedding")


    def build_graph(self):
        #[batch_size,input_length]
        self.input_pinyin_ids = tf.placeholder(tf.int32,[None,None],name = 'input_pinyin_ids')
        #[batch_size,input_length]
        self.input_word_ids = tf.placeholder(tf.int32,[None,None],name = 'input_word_ids')
        #[batch_size]
        self.input_lengths = tf.placeholder(tf.int32,[None],name = 'input_lengths')
        #[batch_size,target_length(=input_length)]
        self.target_word_ids = tf.placeholder(tf.int32,[None,None],name = 'target_word_ids')

        # create dynamic batch size
        self.vary_batch_size = tf.shape(self.input_pinyin_ids)[0]

        # initialize the embedding matrix
        self.init_embedding(pinyin_emb = True,word_emb = True)

        #[batch_size,input_length,pinyin_size]
        #feed_seq = tf.one_hot(indices = self.input_pinyin_ids, depth = self.pinyin_size)
        #[batch_size,input_length,em_size]
        feed_seq_pinyin = tf.nn.embedding_lookup(self.pinyin_embedding,self.input_pinyin_ids)
        #[batch_size,input_length,em_size]
        feed_seq_word = tf.nn.embedding_lookup(self.word_embedding,self.input_word_ids)
        #[batch_size,input_length,em_size * 2]
        feed_seq = tf.concat([feed_seq_pinyin, feed_seq_word], axis=2)

        # use bidirectional LSTM
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell = cell_fw, output_keep_prob=self.keep_ratio)

        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell = cell_bw, output_keep_prob=self.keep_ratio)

        # initial states
        initial_state_fw = cell_fw.zero_state(self.vary_batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.vary_batch_size, tf.float32)

        #BLSTM
        (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                                cell_bw, 
                                                                                inputs=feed_seq,
                                                                                initial_state_fw=initial_state_fw,
                                                                                initial_state_bw=initial_state_bw,
                                                                                sequence_length=self.input_lengths, 
                                                                                dtype=tf.float32,
                                                                                scope='BiLSTM')

        #[batch_size,input_length,hidden_size*2]
        outputs = tf.concat([forward_output, backward_output], axis=2)

        # add multihead self-attention
        result_list = []
        factor = tf.sqrt(tf.constant(self.hidden_size,dtype = tf.float32))
        for k in range(self.head_num):
            tmp_str = 'head_' + str(k+1)
            with tf.variable_scope(tmp_str):
                w_p = tf.Variable(tf.truncated_normal([self.hidden_size*2, self.hidden_size], stddev=0.1),name = 'w_p')
                b_p = tf.Variable(tf.zeros(self.hidden_size),name = 'b_p')

                # During training, we should calculate the attention for each sample in the batch
                ind = tf.constant(0)
                output_ta = tf.TensorArray(dtype=tf.float32, size=self.vary_batch_size)

                def cond(ind,output_ta):
                    return ind < self.vary_batch_size

                def body(ind,output_ta):
                    #[input_length,hidden_size*2]
                    single  = outputs[ind,:,:]
                    #[input_length,hidden_size]
                    single = tf.matmul(single,w_p) + b_p
                    #[input_length,input_length]
                    #soft_out = tf.nn.softmax( tf.matmul(a = single,b = single,transpose_b=True) / factor, axis = 1 )
                    soft_out = tf.nn.softmax( tf.matmul(a = single,b = single,transpose_b=True) / factor, dim = 1 ) # dim for tf 1.3.0
                    #[input_length,hidden_size]
                    att_out = tf.matmul(soft_out,single)
                    output_ta = output_ta.write(ind,att_out)

                    # increment
                    ind = ind + 1

                    return ind,output_ta

                _,final_output_ta = tf.while_loop(cond,body,[ind,output_ta])
                #[batch_size,input_length,hidden_size]
                single_output = final_output_ta.stack()
                print(type(single_output))
                print(single_output.get_shape())
                
            result_list.append(single_output)

        #[batch_size,input_length,hidden_size * head_num]
        new_outputs = tf.concat(result_list,axis = 2)
        print(type(new_outputs))
        print(new_outputs.get_shape())


        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.hidden_size * self.head_num, self.vocab_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.vocab_size))

        #[batch_size*input_length,hidden_size * head_num]
        x = tf.reshape(new_outputs, [-1,self.hidden_size * self.head_num])

        #[batch_size*input_length,vocab_size]
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            #[batch_size,input_length,vocab_size]
            y_one_hot = tf.one_hot(self.target_word_ids, self.vocab_size)
            y_reshaped = tf.reshape(y_one_hot, [-1,self.vocab_size])
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def setup_train(self):
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)


    def setup_summary(self):
        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def train_one_step(self, input_pinyin_ids,input_word_ids,input_lengths,target_word_ids,sess):
        """perform one step of training"""
        feed_dict = {}
        feed_dict[self.input_pinyin_ids] = input_pinyin_ids
        feed_dict[self.input_word_ids] = input_word_ids
        feed_dict[self.input_lengths] = input_lengths
        feed_dict[self.target_word_ids] = target_word_ids
        loss, _ = sess.run(
            [self.loss, self.updates], feed_dict=feed_dict)
        return loss

    def infer(self, input_pinyin_ids,input_word_ids,input_lengths,target_word_ids,sess):
        """perform inference"""
        feed_dict = {}
        feed_dict[self.input_pinyin_ids] = input_pinyin_ids
        feed_dict[self.input_word_ids] = input_word_ids
        feed_dict[self.input_lengths] = input_lengths
        proba = sess.run(
            self.proba_prediction, feed_dict=feed_dict)
        return proba


