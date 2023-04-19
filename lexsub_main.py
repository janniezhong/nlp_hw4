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

    # context of the target word
    stop_words = stopwords.words('english')
    sentence = context.left_context+context.right_context
    filtered_sentence = []
    for word in sentence:
        if word not in stop_words:
            filtered_sentence.append(word)

    # iterate through the synsets to find the amount of overlap
    overlap_dict = {}
    for syn in wn.synsets(lemma, pos):
        definition = tokenize(syn.definition())
        for example in syn.examples():
            definition= definition+tokenize(example)
        for hypernym_syn in syn.hypernyms():
            definition= definition+tokenize(hypernym_syn.definition())
            for example in hypernym_syn.examples():
                definition= definition+tokenize(example)

        lexemes = syn.lemmas()
        if len(lexemes) == 1:
            if lexemes[0].name() == lemma:
                continue
        intersection = list(set(filtered_sentence)&set(definition))
        num_intersect = len(intersection)
        if num_intersect in overlap_dict.keys():
            overlap_dict[num_intersect].append(syn)
        else:
            overlap_dict[num_intersect] = [syn]
    
    #compute the max overlap
    max_intersect = max(overlap_dict.keys())
    best_synset = None
    weighted_intersect = {}
    # lemma.count()
    if max_intersect == 0:
        synset_overlap_all_list = set().union(*overlap_dict.values())
        max_count = -1
        # most frequent synset
        for syn in synset_overlap_all_list:
            lexemes = syn.lemmas()
            count = 0
            if len(lexemes) == 1:
                if lexemes[0].name() == lemma:
                    continue
            for lexeme in syn.lemmas():
                count += lexeme.count()
                if count > max_count:
                    best_synset = syn
                    max_count = count
    elif len(overlap_dict[max_intersect]) == 1:
        best_synset = overlap_dict[max_intersect][0]
    else:
        best_synset_list = overlap_dict[max_intersect]

        max_count = -1
        # most frequent synset
        for syn in best_synset_list:
            lexemes = syn.lemmas()
            count = 0
            if len(lexemes) == 1:
                if lexemes[0].name() == lemma:
                    continue
            for lexeme in syn.lemmas():
                count += lexeme.count()
                if count > max_count:
                    best_synset = syn
                    max_count = count

    # most frequent lexeme from synset
    most_freq_lex = None
    max_count = 0
    # Get the lemma from the synset
    for lem in best_synset.lemmas():
        if lem.name() != lemma:
            if lem.count() > max_count:
                max_count = lem.count()
                most_freq_lex = lem
    if max_count == 0:
        most_freq_lex = best_synset.lemmas()[0]
    
    return most_freq_lex.name().replace("_", " ")
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # Write the method predict_nearest(context) that should first obtain a set of possible synonyms from 
        # WordNet (either using the method from part 1 or you can rewrite this code as you see fit), and then 
        # return the synonym that is most similar to the target word, according to the Word2Vec embeddings. In my 
        # experiments, this approach worked slightly better than the WordNet Frequency baseline and resulted in a 
        # precision and recall of about 11%.
        lemma = context.lemma
        pos = context.pos

        synonyms = get_candidates(lemma, pos)
        max_sim = -1
        nearest_synonym = None
        for synonym in synonyms:
            if synonym in self.model.key_to_index:
                sim = self.model.similarity(lemma, synonym)
                if sim > max_sim:
                    max_sim = sim
                    nearest_synonym = synonym

        return nearest_synonym


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        lemma = context.lemma
        pos = context.pos

        synonyms = get_candidates(lemma, pos)
        sentence = ""
        for tok in context.left_context:
            if tok.isalpha():
                sentence = sentence + ' ' + tok
            else:
                sentence += tok

        sentence = sentence + ' ' + '[MASK]'
        for tok in context.right_context:
            if tok.isalpha():
                sentence = sentence + ' ' + tok
            else:
                sentence += tok

        input_toks = self.tokenizer.encode(sentence)
        sent_tokenized = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_id = sent_tokenized.index('[MASK]')
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = this.model.predict(input_ids)
        predictions = outputs[0]

        print(predictions[0])

        best_words = np.argsort(predictions[0][mask_id])[::-1]
        tokenizer.convert_ids_to_tokens(best_words)

        for word in best_words:
            word_clean = word.replace("_", ' ')
            if word_clean in synonyms:
                return word_clean

        return ""

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor = BertPredictor()

    
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context) 
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
