#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:18:06 2019

@author: adityapandey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:14:49 2019

@author: adityapandey
"""

import tensorflow as tf
import numpy as np
from collections import Counter
import os


train_file = 'HipHop_clean.txt' #Give the name of the input file for that particular genre
seq_size = 32
batch_size = 16
embedding_size = 128
lstm_size = 64
dropout_keep_prob = 0.7
gradients_norm = 5
initial_words = ['I','am']
predict_top_k = 5
num_epochs = 5
checkpoint_path = 'checkpoint'

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
    print(in_text[:10, :10])
    print(out_text[:10, :10])
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
        output, state = tf.nn.bidirectional_dynamic_rnn(lstm, lstm, embed, initial_state_fw=initial_state, initial_state_bw=initial_state, dtype=tf.float32)
        logits = tf.layers.dense(output[0], n_vocab, reuse=reuse)
        preds = tf.nn.softmax(logits)
        return in_op, out_op, lstm, initial_state, state[0], preds, logits


def get_loss_and_train_op(out_op, logits, gradients_norm):
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=out_op, logits=logits))

    trainable_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, trainable_vars), gradients_norm)
    opt = tf.train.AdamOptimizer()
    train_op = opt.apply_gradients(zip(grads, trainable_vars))
    
    return loss_op, train_op


def main(unused_argv):
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        train_file, batch_size, seq_size)
    #Train
    in_op, out_op, lstm, initial_state, state, preds, logits = network(
        batch_size, seq_size, embedding_size,
        lstm_size, dropout_keep_prob, n_vocab)
    

    val_in_op, _, _, val_initial_state, val_state, val_preds, _ = network(
        1, 1, embedding_size,
        lstm_size, dropout_keep_prob,
        n_vocab, reuse=True)
    loss_op, train_op = get_loss_and_train_op(out_op, logits, gradients_norm)
    sess = tf.Session()
    saver = tf.train.Saver()

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)


    sess.run(tf.global_variables_initializer())

    iteration = 0
    for e in range(num_epochs):
        batches = get_batches(in_text, out_text, batch_size, seq_size)
        new_state = sess.run(initial_state)
        if e == num_epochs-1:
            break
        for x, y in batches:
            iteration += 1
            loss, new_state, _ = sess.run(
              [loss_op, state, train_op],
              feed_dict={in_op: x, out_op: y, initial_state: new_state})
            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, num_epochs),
                    'Iteration: {}'.format(iteration),
                    'Loss: {:.4f}'.format(loss))
            if iteration % 200 == 0:
                predict(initial_words, predict_top_k,
                      sess, val_in_op, val_initial_state,
                      val_preds, val_state, n_vocab,
                      vocab_to_int, int_to_vocab)
                saver.save(
                  sess,
                  os.path.join(checkpoint_path, 'model-{}.ckpt'.format(iteration)))
    
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
    file = "HipHop-{}.txt".format(i)
    raw_file = "Raw_HipHop-{}.txt".format(i)
    max_edit_distance_lookup = 2
    suggestions = sym_spell.lookup_compound(input_term,
                                        max_edit_distance_lookup)
    for suggestion in suggestions:
        print("{}".format(suggestion.term))
    
    '''    
    with open("New Lyrics/HipHop/"+file, "w") as text_file:
        text_file.write("{}".format(suggestion.term))

    with open("New Lyrics/HipHop/"+raw_file, "w") as text_file:
        text_file.write("{}".format(input_term))
    '''

if __name__ == '__main__':
    tf.app.run()