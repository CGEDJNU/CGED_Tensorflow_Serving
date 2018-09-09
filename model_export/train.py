# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import pickle
from tqdm import tqdm

from model import SpellChecker
from utils import DataReader,norm,batch_generator_triple_with_length

flags = tf.app.flags
FLAGS = flags.FLAGS

# define hyperparameters
flags.DEFINE_string('mode', 'train', 'Must be one of train/decode')
flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
flags.DEFINE_string('data_file_path', '', 'Path of the data file')
flags.DEFINE_string('pinyin_dict_path', '/home/wangzheng/python_project/SpellChecker/data/pinyin_dict_full.pkl', 'Path of the pinyin dict')
flags.DEFINE_string('id_data_dir', '/home/zhanghaipeng/main/SpellChecker/data/lcsts_aug/', 'Dir of the saved id data')
flags.DEFINE_integer('n_epoch', 5, 'Number of epoch to train the model')
flags.DEFINE_integer('batch_size', 64, 'Minibatch size')
flags.DEFINE_integer('hidden_size', 128, 'Dimension of RNN hidden states')
flags.DEFINE_integer('head_num', 3, 'Number of heads used in self-attention')
flags.DEFINE_integer('seed_num', 42, 'Seed for random number generator')
flags.DEFINE_integer('num_hidden_layers', 1, 'Number of hidden layers')
flags.DEFINE_integer('max_timesteps', 40, 'Max length of the sequence')
flags.DEFINE_integer('max_gradient_norm', 5, 'Max norm of the gradient')
flags.DEFINE_integer('em_size', 100, 'Dimension of the embedding')
flags.DEFINE_integer('vocab_size', 5000, 'Size of vocabulary')
flags.DEFINE_integer('train_size', 2400000, 'Size of the training set')
flags.DEFINE_integer('gen_size', 500, 'Size of the generated samples')
flags.DEFINE_integer('max_to_keep', 3, 'Number of checkpoints to keep')
flags.DEFINE_integer('load_data_and_dr', 1, 'We load the dr and the id data if turned on')
flags.DEFINE_integer('use_local', 1, 'We load or save the dr and the id data from local dir if turned on')
flags.DEFINE_float('lr', 5e-4, 'Learning rate')
flags.DEFINE_float('keep_ratio', 0.75, 'keep ratio for dropout')

def main(unused_argv):
    # prints a message if you've entered flags incorrectly
    if len(unused_argv) != 1: 
        raise Exception("Problem with flags: %s" % unused_argv)
    
    # Get hyperparameters. We only get a subset of all the hyperparameters, others would be feed to Model directly.
    #logging.basicConfig(level=logging.INFO)
    print('Starting Basic model')
    log_root = FLAGS.log_root
    exp_name = FLAGS.exp_name
    data_file_path = FLAGS.data_file_path
    pinyin_dict_path = FLAGS.pinyin_dict_path
    id_data_dir = FLAGS.id_data_dir
    
    n_epoch = FLAGS.n_epoch
    batch_size = FLAGS.batch_size
    seed_num = FLAGS.seed_num
    max_timesteps= FLAGS.max_timesteps
    vocab_size = FLAGS.vocab_size
    train_size = FLAGS.train_size
    load_data_and_dr = FLAGS.load_data_and_dr
    use_local = FLAGS.use_local
        
    
    # make the directory for logs
    log_root = os.path.join(log_root, exp_name)
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    if use_local == 1:
        #load or save the DR class from local dir
        DR_path = os.path.join(log_root, 'DataReader.pkl')
        #load or save the id data from local dir
        id_data_path = os.path.join(log_root, 'id_data.pkl')
    else:
        #load or save the DR class from global dir
        DR_path = os.path.join(id_data_dir, 'DataReader.pkl')
        #load or save the id data from global dir
        id_data_path = os.path.join(id_data_dir, 'id_data.pkl')

    if load_data_and_dr == 1:
        with open(DR_path,'rb') as f:
            DR = pickle.load(f)
        with open(id_data_path,'rb') as f1:
            input_pinyin_data = pickle.load(f1)
            input_word_data = pickle.load(f1)
            target_data = pickle.load(f1)
    else:
        # load and make the data for training
        DR = DataReader(vocab_size = vocab_size, pinyin_dict_path = pinyin_dict_path)
        #input_data,target_data = DR.make_data_from_scratch(file_path = data_file_path,build_dictionary=True)
        input_pinyin_data,input_word_data,target_data = DR.make_data_from_dataframe(file_path = data_file_path,build_dictionary=True,max_rows = train_size)
        #save the DR class to local dir
        with open(DR_path,'wb') as f:
            pickle.dump(DR,f)

        #save the ids data to local dir
        with open(id_data_path,'wb') as f1:
            pickle.dump(input_pinyin_data,f1)
            pickle.dump(input_word_data,f1)
            pickle.dump(target_data,f1)
    
    # make the batch
    train_data_full= batch_generator_triple_with_length(input_pinyin_data,input_word_data,target_data,batch_size,max_timesteps,DR.word2id,DR.pinyin2id)

    # create the model
    model = SpellChecker(hps = FLAGS)

    
    # create the supervisor
    with model.graph.as_default():
        # print the variables of tensorflow
        print("Number of sets of parameters: {}".format(len(tf.trainable_variables())))
        print("Number of parameters: {}".format(
                np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))
        for v in tf.trainable_variables():
            print(v)

        sv = tf.train.Supervisor(logdir=log_root,
                                saver = model.saver,
                                summary_op=None,
                                save_model_secs=60,
                                global_step = model.global_step,
                                init_op=model.init_op) # Do not run the summary service


        # train the model 
        with sv.managed_session() as sess:
            n_iter_per_epoch = len(input_pinyin_data) // (batch_size * 2)
            epoch = 0.0
            print('number of iterations per epoch: {}'.format(n_iter_per_epoch))
            print('start training...')     
            for _ in range(n_epoch * 2):
                epoch += 0.5
                avg_loss = 0.0
                print("----- Epoch {}/{} -----".format(epoch, n_epoch))
                for t in tqdm(range(1, n_iter_per_epoch + 1)):
                    batch_full = next(train_data_full)
                    src_pinyin_list,src_word_list,src_length_list,tgt_list,tgt_length_list = batch_full
                    
                    #if epoch == 0.5:
                        #print(src_list[1])
                        #print(len(src_list[1]))
                        #print(src_length_list[1])
                        #print(tgt_list[1])
                        #print(len(tgt_list[1]))
                        #print(tgt_length_list[1])
                    
                    src_pinyin_list = np.asarray(src_pinyin_list,dtype = np.int32)
                    src_word_list = np.asarray(src_word_list,dtype = np.int32)
                    src_length_list = np.asarray(src_length_list,dtype = np.int32)
                    tgt_list = np.asarray(tgt_list,dtype = np.int32)
                    #tgt_length_list = np.asarray(tgt_length_list,dtype = np.int32)
                    loss = model.train_one_step(src_pinyin_list, src_word_list,src_length_list, tgt_list,sess)
                    avg_loss +=loss
                avg_loss /= n_iter_per_epoch
                print('the avg_loss is {}'.format(avg_loss))
if __name__ == "__main__":
    tf.app.run()




    








