import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from grpc.beta import implementations

from utils import DataReader,norm,batch_generator_triple_with_length


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'decode', 'Must be one of train/decode')
flags.DEFINE_string('test_file_path', '', 'Path of the test file')
flags.DEFINE_string('id_data_dir', '', 'Dir of the saved id data')
flags.DEFINE_integer('batch_size', 64, 'Minibatch size')
flags.DEFINE_integer('hidden_size', 128, 'Dimension of RNN hidden states')
flags.DEFINE_integer('head_num', 3, 'Number of heads used in self-attention')
flags.DEFINE_integer('seed_num', 42, 'Seed for random number generator')
flags.DEFINE_integer('max_timesteps', 40, 'Max length of the sequence')
flags.DEFINE_integer('max_gradient_norm', 5, 'Max norm of the gradient')
flags.DEFINE_integer('em_size', 100, 'Dimension of the embedding')
flags.DEFINE_integer('vocab_size', 5000, 'Size of vocabulary')
flags.DEFINE_integer('test_size', 240000, 'Size of the test set')
flags.DEFINE_integer('gen_size', 500, 'Size of the generated samples')
flags.DEFINE_integer('max_to_keep', 3, 'Number of checkpoints to keep')
flags.DEFINE_integer('use_local', 0, 'We load or save the dr and the id data from local dir if turned on')
flags.DEFINE_float('keep_ratio', 1.00, 'keep ratio for dropout')

def main(unused_argv):
    
    test_file_path = FLAGS.test_file_path
    id_data_dir = FLAGS.id_data_dir
    
    batch_size = FLAGS.batch_size
    seed_num = FLAGS.seed_num
    max_timesteps= FLAGS.max_timesteps
    vocab_size = FLAGS.vocab_size
    test_size = FLAGS.test_size
    use_local = FLAGS.use_local
    

    DR_path = os.path.join(id_data_dir, 'DataReader.pkl')
    with open(DR_path,'rb') as f:
        DR = pickle.load(f)

    input_pinyin_data,input_word_data,target_data = DR.make_data_from_dataframe(file_path = test_file_path,build_dictionary=False,max_rows = test_size)

    
    np.random.seed(seed_num)
    np.random.shuffle(input_pinyin_data)
    np.random.seed(seed_num)
    np.random.shuffle(input_word_data)
    np.random.seed(seed_num)
    np.random.shuffle(target_data)
    
    test_data_full= batch_generator_triple_with_length(input_pinyin_data,input_word_data,target_data,batch_size,max_timesteps,DR.word2id,DR.pinyin2id)
    
    n_iter_per_epoch = len(input_pinyin_data) // (batch_size)
    
    for t in range(1, n_iter_per_epoch + 1):
        batch_full = next(test_data_full)
        src_pinyin_list,src_word_list,src_length_list,tgt_list,tgt_length_list = batch_full
        src_pinyin_list = np.asarray(src_pinyin_list,dtype = np.int32)
        src_word_list = np.asarray(src_word_list,dtype = np.int32)
        src_length_list = np.asarray(src_length_list,dtype = np.int32)
        tgt_list = np.asarray(tgt_list,dtype = np.int32)
    
        
        hostport = '0.0.0.0:9012'
        host,port = hostport.split(':')
        
        channel = implementations.insecure_channel(host,int(port))
        #channel = insecure_channel(host,int(port))
        
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        request = predict_pb2.PredictRequest()

        request.model_spec.name = 'spell'
        request.model_spec.signature_name = "predict"

        request.inputs['pinyin_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(src_pinyin_list, shape=[src_pinyin_list.shape[0], src_pinyin_list.shape[1]]))
        request.inputs['word_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(src_word_list, shape=[src_word_list.shape[0], src_word_list.shape[1]]))
        request.inputs['input_lengths'].CopyFrom(tf.contrib.util.make_tensor_proto(src_length_list))
        request.inputs['keep_ratio'].CopyFrom(tf.contrib.util.make_tensor_proto(FLAGS.keep_ratio))
        
        print('Predict:')        
        proba = stub.Predict(request,50.0)
        results = {}
        for key in proba.outputs:
            tensor_proto = proba.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results[key] = nd_array
        
        #import pdb
        #pdb.set_trace()

        predict_ids = np.argmax(results['predict'],axis = 1)
        print(predict_ids)

if __name__ == '__main__':
    tf.app.run()
