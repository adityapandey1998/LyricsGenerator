#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:13:48 2019

@author: adityapandey
"""

import lyrics_gen
import tensorflow as tf

seq_size = 32
embedding_size = 32
lstm_size = 128
dropout_keep_prob = 0.7
batch_size = 32 

print("Choose Type of Model: \n1. RNN with LSTM Cells\n2. RNN with GRU Cells\n3. Bi-Directional RNN")
#model = int(input())

print("Choose Genre of Lyrics: \n1. Rock\n2. Hip Hop\n3. Country\n4. Metal")
#genre = int(input())

genre_map = {
        1:"Rock",
        2:"Hip Hop",
        3:"Country",
        4:"Metal"}

#chosen_genre = genre_map[genre]
train_file = 'Rock_clean.txt'
'''
if model==1:
    print("Generating using LSTM Cells...")
if model==2:
    print("Generating using GRU Cells...")
if model==3:
    print("Generating using BiDirectional RNN with LSTM Cells...")
else:
    print("Wrong Input")
'''
 
tf.reset_default_graph()
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    # restore the saved vairable
    saver.restore(sess, './checkpoint')   
    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = lyrics_gen.get_data_from_file(
        train_file, batch_size, seq_size)

    val_in_op, _, _, val_initial_state, val_state, val_preds, _ = lyrics_gen.network(
        1, 1, embedding_size,
        lstm_size, dropout_keep_prob,
        n_vocab, reuse=True)
    
    lyrics_gen.predict(['I','am'], 5,
                      sess, val_in_op, val_initial_state,
                      val_preds, val_state, n_vocab,
                      vocab_to_int, int_to_vocab)