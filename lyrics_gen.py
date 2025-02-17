#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:45:03 2019

@author: adityapandey
"""

import tensorflow as tf
import numpy as np
from collections import Counter
import os

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pkg_resources
from symspellpy.symspellpy import SymSpell  # import the module

max_edit_distance_dictionary = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")

if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                     count_index=1):
        print("Dictionary file not found")

if not sym_spell.load_bigram_dictionary(dictionary_path, term_index=0,
                                            count_index=2):
        print("Bigram dictionary file not found")
    

flags = tf.app.flags

flags.DEFINE_string('train_file', 'Rock_clean.txt', 'text file to train LSTM')
flags.DEFINE_integer('seq_size', 32, 'sequence length')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('embedding_size', 128, 'Embedding hidden size')
flags.DEFINE_integer('lstm_size', 128, 'LSTM hidden size')
flags.DEFINE_float('dropout_keep_prob', 0.7, 'LSTM dropout keep probability')
flags.DEFINE_integer('gradients_norm', 5, 'norm to clip gradients')
flags.DEFINE_multi_string('initial_words', ['I', 'am'], 'Initial words to start prediction from')
flags.DEFINE_integer('predict_top_k', 5, 'top k results to sample word from')
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to train')
flags.DEFINE_string('checkpoint_path', 'checkpoint', 'directory to store trained weights')
FLAGS = flags.FLAGS

tf.reset_default_graph()

def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, encoding='utf-8') as f:
        text = f.read()

    text = text.split()
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k:w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w:k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    #print(in_text[:10, :10])
    #print(out_text[:10, :10])
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


def network(batch_size, seq_size, embedding_size, lstm_size, keep_prob, n_vocab, reuse=False):
    with tf.variable_scope('LSTM', reuse=reuse):
        in_op = tf.placeholder(tf.int32, [None, seq_size])
        out_op = tf.placeholder(tf.int32, [None, seq_size])
        embedding = tf.get_variable('embedding_weights', [n_vocab, embedding_size])
        embed = tf.nn.embedding_lookup(embedding, in_op)
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(lstm, embed, initial_state=initial_state, dtype=tf.float32)
        logits = tf.layers.dense(output, n_vocab, reuse=reuse)
        preds = tf.nn.softmax(logits)
        return in_op, out_op, lstm, initial_state, state, preds, logits


def get_loss_and_train_op(out_op, logits, gradients_norm):
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=out_op, logits=logits))
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss_op)
    # Don't do this
    # opt = tf.train.AdamOptimizer()
    # train_op = opt.minimize(loss_op)

    trainable_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, trainable_vars), gradients_norm)
    opt = tf.train.AdamOptimizer()
    train_op = opt.apply_gradients(zip(grads, trainable_vars))
    
    return loss_op, train_op


def main(unused_argv):
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        FLAGS.train_file, FLAGS.batch_size, FLAGS.seq_size)
    # For training
    in_op, out_op, lstm, initial_state, state, preds, logits = network(
        FLAGS.batch_size, FLAGS.seq_size, FLAGS.embedding_size,
        FLAGS.lstm_size, FLAGS.dropout_keep_prob, n_vocab)
    
    # For inteference
    val_in_op, _, _, val_initial_state, val_state, val_preds, _ = network(
        1, 1, FLAGS.embedding_size,
        FLAGS.lstm_size, FLAGS.dropout_keep_prob,
        n_vocab, reuse=True)
    loss_op, train_op = get_loss_and_train_op(out_op, logits, FLAGS.gradients_norm)
    sess = tf.Session()
    saver = tf.train.Saver()

    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)

    sess.run(tf.global_variables_initializer())

    iteration = 0
    for e in range(FLAGS.num_epochs):
        batches = get_batches(in_text, out_text, FLAGS.batch_size, FLAGS.seq_size)
        new_state = sess.run(initial_state)
        for x, y in batches:
            iteration += 1
            loss, new_state, _ = sess.run(
              [loss_op, state, train_op],
              feed_dict={in_op: x, out_op: y, initial_state: new_state})
            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e+1, FLAGS.num_epochs),
                    'Iteration: {}'.format(iteration),
                    'Loss: {:.4f}'.format(loss))
            if iteration % 100 == 0:
                predict(FLAGS.initial_words, FLAGS.predict_top_k,
                      sess, val_in_op, val_initial_state,
                      val_preds, val_state, n_vocab,
                      vocab_to_int, int_to_vocab)
                saver.save(
                  sess,
                  os.path.join(FLAGS.checkpoint_path, 'model-{}.ckpt'.format(iteration)))
    
i=0
def predict(initial_words, predict_top_k, sess, in_op,
            initial_state, preds, state, n_vocab, vocab_to_int, int_to_vocab):
    new_state = sess.run(initial_state)
    words = initial_words
    samples = [w for w in words]

    for word in words:
        x = np.zeros((1, 1))
        x[0, 0] = vocab_to_int[word]
        pred, new_state = sess.run([preds, state], feed_dict={in_op: x, initial_state: new_state})

    def get_word(pred):
        p = np.squeeze(pred)
        p[p.argsort()][:-predict_top_k] = 0
        p = p / np.sum(p)
        word = np.random.choice(n_vocab, 1, p=p)[0]

        return word
    word = get_word(pred)

    samples.append(int_to_vocab[word])

    n_samples = 200

    for _ in range(n_samples):
        x[0, 0] = word
        pred, new_state = sess.run([preds, state], feed_dict={in_op: x, initial_state: new_state})
        word = get_word(pred)
        samples.append(int_to_vocab[word])
    
    input_term = ' '.join(samples)

    global i
    i+=1
    file = "Rock-{}.txt".format(i)
    raw_file = "Raw_Rock-{}.txt".format(i)
    max_edit_distance_lookup = 2
    suggestions = sym_spell.lookup_compound(input_term,
                                        max_edit_distance_lookup)
    for suggestion in suggestions:
        print("{}".format(suggestion.term))
        
    with open("New Lyrics/"+file, "w") as text_file:
        text_file.write("{}".format(suggestion.term))

    with open("New Lyrics/"+raw_file, "w") as text_file:
        text_file.write("{}".format(input_term))

if __name__ == '__main__':
    tf.app.run()