# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

################################################################
# with pretrained embeddings and using sigmoid in f1 score
#Best Validation F1 Score is 0.6818 at threshold 0.41
################################################################
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import tensorflow as tf
from tensorflow.contrib import rnn
import re
from nltk.tokenize import word_tokenize
import collections
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Any results you write to the current directory are saved as output.
from tqdm import tnrange, tqdm_notebook, tqdm
import tensorflow.contrib.slim as slim
start = time.time()

print(os.listdir("../../input"))
print(os.listdir("."))

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def get_embeddings(word_dict, embed_size):
    # embdedding setup
    # Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    embeddings = {}
    f = open('../../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
    for i, line in enumerate(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings))    

    all_embs = np.stack(embeddings.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    nb_words = len(word_dict) # only want at most vocab_size words in our vocabulary 
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #first, make our embedding matric random7
    num_missed = 0
    for word, i in word_dict.items(): # insert embeddings we that exist into our matrix
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else: 
            num_missed += 1
    print(num_missed)
    
    return embedding_matrix

class AttentionRNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class, embedding):
        self.embedding_size = 100
        self.num_hidden = 64
        self.num_layers = 2
        self.learning_rate = 1e-3

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.7, 1.0)

        with tf.name_scope("embedding"):
            #print(embedding.shape) 
            self.embeddings = tf.get_variable("embeddings", shape=embedding.shape, initializer=tf.constant_initializer(embedding), trainable=False)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)                             
        with tf.name_scope("birnn"):
            fw_cells = [rnn.GRUCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [rnn.GRUCell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)

        with tf.name_scope("attention"):
            self.attention_score = tf.nn.softmax(tf.layers.dense(self.rnn_outputs, 1, activation=tf.nn.tanh), axis=1)
            self.attention_out = tf.squeeze(tf.matmul(tf.transpose(self.rnn_outputs, perm=[0, 2, 1]), self.attention_score),axis=-1)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(self.attention_out, num_class, activation=None)
            self.sigmoid = tf.nn.sigmoid(self.logits) #activation function
            self.sigmoid = tf.squeeze(self.sigmoid, [1]) # layers.dense returns a tensor, but we want to remove
            self.predictions = tf.cast(self.sigmoid >= 0.5, tf.int32)
            #self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(self.logits), labels=tf.cast(self.y, tf.float32)))
            tf.summary.scalar('cross_entropy', self.loss)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.f1_score = tf.contrib.metrics.f1_score(labels=self.y, predictions=self.sigmoid, name='f1_score')
            tf.summary.scalar('m_accuracy', self.accuracy)
            tf.summary.scalar('m_f1_score', self.f1_score[0])

        self.merged = tf.summary.merge_all()

TRAIN_PATH="../../input/train.csv"
TEST_PATH="../../input/test.csv"

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text

def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        train_df = pd.read_csv(TRAIN_PATH)
        contents = train_df["question_text"]

        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict


def build_word_dataset(step, word_dict, document_max_len):
    if step == "train":
        df = pd.read_csv(TRAIN_PATH)
    else:
        df = pd.read_csv(TEST_PATH)
     
    # returns random samples - we dont need it for test          
    #df = df.sample(frac=1)
    print("tokenising")
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["question_text"]))
    print("getting words to dict int")
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    print("adding eos")
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    print("cutting to document max len")
    x = list(map(lambda d: d[:document_max_len], x))
    print("padding shorter questions")
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    
    if step == "train":
        y = list(df["target"])
    else:
        y = [0]*df.shape[0]

    return x, y

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
            
WORD_MAX_LEN = 30

print("building word dict")
word_dict = build_word_dict()
vocabulary_size = len(word_dict)
print("Process time: ",(time.time() - start))

print("building train word dataset")
x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)
print("Process time: ",(time.time() - start))

print("building test word dataset")
test_x, test_y = build_word_dataset("test", word_dict, WORD_MAX_LEN)
print("Process time: ",(time.time() - start))


print("splitting the dataset")
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.07)
print(len(train_y), sum(train_y)/len(train_y))
print(len(valid_y), sum(valid_y)/len(valid_y))
with open("train_x_x.pickle", "wb") as f:
    pickle.dump(train_x, f)
with open("train_x_y.pickle", "wb") as f:
    pickle.dump(train_y, f)
with open("valid_x_x.pickle", "wb") as f:
    pickle.dump(valid_x, f)
with open("valid_x_y.pickle", "wb") as f:
    pickle.dump(valid_y, f)
print("Process time: ",(time.time() - start))
 
print("getting the embeddings")
embed = get_embeddings(word_dict, 300)
print("Process time: ",(time.time() - start))
  
NUM_CLASS = 1
BATCH_SIZE = 512
NUM_EPOCHS = 5
tf.reset_default_graph()
with tf.Session() as sess:
    model = AttentionRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, embed)
    
    train_writer = tf.summary.FileWriter('./log/train',sess.graph)
    test_writer = tf.summary.FileWriter('./log/test')

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    def model_summary():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
    model_summary()
    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss, summary = sess.run([model.optimizer, model.global_step, model.loss, model.merged], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))
            train_writer.add_summary(summary, step)

        if step % 500 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, sum_f1, cnt = 0, 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }

                accuracy, f1, summary = sess.run([model.accuracy, model.f1_score, model.merged], feed_dict=valid_feed_dict)
                test_writer.add_summary(summary, step)

                sum_accuracy += accuracy
                sum_f1 += f1[0]
                cnt += 1
            valid_accuracy = sum_accuracy / cnt
            valid_f1 = sum_f1 / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))
            print("\nValidation F1 = {1}\n".format(step // num_batches_per_epoch, sum_f1 / cnt))

            # Save model
            '''
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format('att_rnn', 'att_rnn'), global_step=step)
                print("Model is saved.\n")
                
            '''
            
    # select the best threshhold
    valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
    val_pred = np.zeros_like(valid_y, dtype=np.float)

    for i, (valid_x_batch, valid_y_batch) in enumerate(valid_batches):
        valid_feed_dict = {
                model.x: valid_x_batch,
                model.y: valid_y_batch,
                model.is_training: False
        }

        pred = sess.run([model.sigmoid], feed_dict=valid_feed_dict)
        val_pred[i*BATCH_SIZE: (i+1)*BATCH_SIZE] = pred[0]
    
    
    thresholds = [i/200 for i in range(10, 120, 1)] 
    scores = [metrics.f1_score(valid_y, np.int16(val_pred > t)) for t in thresholds]
    thresh = thresholds[np.argmax(scores)]
    print(f"Best Validation F1 Score is {max(scores):.4f} at threshold {thresh}")
    

    # predict the test results
    test_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
    test_pred = np.zeros_like(test_y, dtype=np.float)

    for i, (test_x_batch, test_y_batch) in enumerate(test_batches):
        test_feed_dict = {
                model.x: test_x_batch,
                model.is_training: False
        }

        pred = sess.run([model.sigmoid], feed_dict=test_feed_dict)
        test_pred[i*BATCH_SIZE: (i+1)*BATCH_SIZE] = pred[0]
    
    # generate the prediction file             
    test = pd.read_csv(TEST_PATH)
    sub = test[['qid']]
    sub['prediction'] = test_pred
    sub['prediction'] = (sub['prediction'] > thresh).astype(np.int16)
    sub.to_csv("submission.csv", index=False)
                
print("Process time: ",(time.time() - start))

#sub = pd.read_csv('submission.csv')
#ksub = pd.read_csv('ksubmission.csv')
#j = pd.merge(sub, ksub, on='qid')
