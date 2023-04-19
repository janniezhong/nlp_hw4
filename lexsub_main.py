#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = []
    synsets = wn.synsets('lemma', pos=pos)
    for s in synsets:
        for lem in s:
            lem_str = str(lem.name())
            lem_str = lem_str.replace("_", " ")
            if lem_str not in candidates and lem_str != lemma:
                candidates.append(lem_str)
    return candidates 

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Write the function wn_frequency_predictor(context) that takes a context object as input and predicts 
    # the possible synonym with the highest total occurence frequency (according to WordNet). Note that 
    # you have to sum up the occurence counts for all senses of the word if the word and the target appear
    # together in multiple synsets. You can use the get_candidates method or just duplicate the code for 
    # finding candidate synonyms (this is possibly more convenient). Using this simple baseline should give
    # you about 10% precision and recall. Take a look at the output to see what kinds of mistakes the system makes.

    # Each Context object corresponds to one target token in context. The instance variables of Context are as follows:

# cid - running ID of this instance in the input file (needed to produce the correct output for the scoring script).
# word_form - the form of the target word in the sentence (for example 'tighter').
# lemma - the lemma of the target word (for example 'tight').
# pos - this can be either 'n' for noun, 'v' for verb, 'a', for adjective, or 'r' for adverb.
# left_context - a list of tokens that appear to the left of the target word. For example ['Anyway', ',', 'my', 'pants', 'are', 'getting']
# right_context - a list of tokens that appear to the right of the target word. For example ['every','day','.']
    # takes lemma, pos, and counts number of candidates

    count = {}
    synsets = wn.synsets('lemma', pos=pos)
    for s in synsets:
        for lem in s:
            lem_str = str(lem.name())
            lem_str = lem_str.replace("_", " ")
            if lem_str != lemma:
                count[lem_str] += lem.count()
    max_key = max(count, key = count.get)
    return max_key 
    

    return None # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
    return None #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        return None # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        return None # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 
        prediction = wn_frequency_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
