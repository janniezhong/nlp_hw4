#!/usr/bin/env python
import sys
import string

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
    synsets = wn.synsets(lemma, pos)
    for s in synsets:
        for lem in s.lemmas():
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
    synsets = wn.synsets(context.lemma, context.pos)
    for s in synsets:
        for lem in s.lemmas():
            lem_str = str(lem.name())
            lem_str = lem_str.replace("_", " ")
            if lem_str != context.lemma:
                if lem_str not in count:
                    count[lem_str] = lem.count()
                else:
                    count[lem_str] += lem.count()
    max_key = max(count, key = count.get)
    return max_key 
    

def wn_simple_lesk_predictor(context : Context) -> str:

    # Look at all possible synsets that the target word apperas in. 
    # Compute the overlap between the definition of the synset and the context of the target word. 
    # You may want to remove stopwords (function words that don't tell you anything about a word's 
    # semantics). You can load the list of English stopwords in NLTK like this:
    # stop_words = stopwords.words('english')

    # You should therefore add the following to the definition:

    # All examples for the synset.
    # The definition and all examples for all hypernyms of the synset.


    # Even with these extensions, the Lesk algorithm will often not produce any overlap. If this is 
    # the case (or if there is a tie), you should select the most frequent synset (i.e. the Synset 
    # with which the target word forms the most frequent lexeme, according to WordNet). Then select 
    # the most frequent lexeme from that synset as the result. One sub-task that you need to solve is 
    # to tokenize and normalize the definitions and examples in WordNet. You could either look up various
    # tokenization methods in NLTK or use the tokenize(s) method provided with the code.

    # WSD to select synset
    lemma = context.lemma
    pos = context.pos

    stop_words = stopwords.words('english')
    sentence = context.left_context+context.right_context
    filtered_sentence = []
    for word in sentence:
        if word not in stop_words:
            filtered_sentence.append(word)

    overlap_dict = {}
    for syn in wn.synsets(lemma, pos):
        definition = tokenize(syn.definition())
        for example in syn.examples():
            definition.append(tokenize(example))
        for hypernym_syn in syn.hypernyms():
            definition.append(tokenize(hypernym_syn.definition()))
            for example in hypernym_syn.examples():
                definition.append(tokenize(example))

        print(filtered_sentence)
        print(definition)
        intersection = set(filtered_sentence).intersection(definition)
        num_intersect = len(intersection)
        if num_intersect in overlap_dict:
            overlap_dict[num_intersect].append(syn)
        else:
            overlap_dict[num_intersect] = [syn]
    
    max_intersect = max(overlap_dict.keys())
    best_synset = None
    if len(overlap_dict[max_intersect]) == 1:
        best_synset = overlap_dict[max_intersect][0]
    else:
        best_synset_list = overlap_dict[max_intersect]
        max_count = -1
        count = None
        # most frequent synset
        for syn in best_synset_list:
            for lexeme in syn.lemmas():
                count += lexeme.count()

            if count > max_count:
                best_synset = syn
                max_count = count

    # most frequent lexeme from sysnet
    freq_lex_name = None
    max_count = 0
    count = {}
    # Get the lemma from the synset
    for lem in best_synset.lemmas():
        lem_str = str(lem.name())
        if lem_str != lemma:
            if lem_str not in count:
                count[lem_str] = lem.count()
            else:
                count[lem_str] += lem.count()
            if count[lem_str] > max_count:
                max_count = count[lem_str]
                freq_lex_name = lem_str
    
    return lem_str.replace("_", " ")
   

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
        #prediction = wn_frequency_predictor(context)
        prediction = wn_simple_lesk_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
