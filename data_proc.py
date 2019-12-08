#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:56:36 2019

@author: adityapandey
"""
import pandas as pd
import re

data = pd.read_csv('lyrics.csv')

lyrics = data.dropna(axis = 0, how ='any') 

lyrics = lyrics[lyrics['lyrics']!='INSTRUMENTAL']

HipHop = lyrics[lyrics['genre']=='Hip-Hop']
Rock = lyrics[lyrics['genre']=='Rock']
Pop = lyrics[lyrics['genre']=='Pop']
Metal = lyrics[lyrics['genre']=='Metal']
Country = lyrics[lyrics['genre']=='Country']

HipHop_sample = HipHop.sample(4000)
Rock_sample = Rock.sample(4000)
Pop_sample = Pop.sample(4000)
Metal_sample = Metal.sample(4000)
Country_sample = Country.sample(4000)

Rock_string = ""

for index, row in Rock_sample.iterrows():
    Rock_string += ' <start> '
    Rock_string += row['lyrics']
print("HipHop: ", len(Rock_string))
Rock_string = re.sub(r'\([^)]*\)', '', Rock_string)
print("HipHop: ", len(Rock_string))
#Rock_string = re.sub(r'\[[^)]*\]', '', Rock_string)
#print("HipHop: ", len(Rock_string))
Rock_string_clean = re.sub('[^A-Za-z \\.\n\'<>]+', '', Rock_string)

Pop_string = ""

for index, row in Pop_sample.iterrows():
    Pop_string += ' <start> '
    Pop_string += row['lyrics']
print("HipHop: ", len(Pop_string))
Rock_string = re.sub(r'\([^)]*\)', '', Rock_string)
print("HipHop: ", len(Pop_string))
#Rock_string = re.sub(r'\[[^)]*\]', '', Rock_string)
#print("HipHop: ", len(Pop_string))
Pop_string_clean = re.sub('[^A-Za-z \\.\n\'<>]+', '', Pop_string)

HipHop_string = ""

for index, row in HipHop_sample.iterrows():
    HipHop_string += ' <start> '
    HipHop_string += row['lyrics']
print("HipHop: ", len(HipHop_string))
HipHop_string = re.sub(r'\([^)]*\)', '', HipHop_string)
print("HipHop: ", len(HipHop_string))
#HipHop_string = re.sub(r'\[[^)]*\]', '', HipHop_string)
#print("HipHop: ", len(HipHop_string))
HipHop_string_clean = re.sub('[^A-Za-z \\.\n\'<>]+', '', HipHop_string)

Metal_string = ""

for index, row in Metal_sample.iterrows():
    Metal_string += ' <start> '
    Metal_string += row['lyrics']
print("HipHop: ", len(Metal_string))
Metal_string = re.sub(r'\([^)]*\)', '', Metal_string)
print("HipHop: ", len(Metal_string))
#Metal_string = re.sub(r'\[[^)]*\]', '', Metal_string)
#print("HipHop: ", len(Metal_string))
Metal_string_clean = re.sub('[^A-Za-z \\.\n\'<>]+', '', Metal_string)

Country_string = ""

for index, row in Country_sample.iterrows():
    Country_string += ' <start> '
    Country_string += row['lyrics']    
print("HipHop: ", len(Country_string))
Country_string = re.sub(r'\([^)]*\)', '', Country_string)
print("HipHop: ", len(Country_string))
#Country_string = re.sub(r'\[[^)]*\]', '', Country_string)
#print("HipHop: ", len(Country_string))
Country_string_clean = re.sub('[^A-Za-z \\.\n\'<>]+', '', Country_string)

with open("Rock_clean.txt", "w") as text_file:
    text_file.write(Rock_string_clean)
    
with open("Pop_clean.txt", "w") as text_file:
    text_file.write(Pop_string_clean)
    
with open("HipHop_clean.txt", "w") as text_file:
    text_file.write(HipHop_string_clean)

with open("Metal_clean.txt", "w") as text_file:
    text_file.write(Metal_string_clean)

with open("Country_clean.txt", "w") as text_file:
    text_file.write(Country_string_clean)