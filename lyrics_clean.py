#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:16:13 2019

@author: adityapandey
"""

import pkg_resources
from symspellpy.symspellpy import SymSpell  # import the module

max_edit_distance_dictionary = 2
prefix_length = 5
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



file = "tf2Rock"
        
with open("New Lyrics/kerasoutputs/"+file+".txt", "r") as text_file:
    lines = text_file.read().split('\n')
    max_edit_distance_lookup = 2
    
    op = ""
    for line in lines:
        suggestions = sym_spell.lookup_compound(line, max_edit_distance_lookup)
        for suggestion in suggestions:
            op += "{}".format(suggestion.term)
        op += "\n"
        
    with open("New Lyrics/kerasoutputs/"+file+"_clean.txt", "w") as out_file:
        out_file.write("{}".format(op))

#result = sym_spell.word_segmentation(input_term)