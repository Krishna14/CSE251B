################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# See this for input references - https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu
# A Caption should be a list of strings.
# Reference Captions are list of actual captions - list(list(str))
# Predicted Caption is the string caption based on your model's output - list(str)
# Make sure to process your captions before evaluating bleu scores -
# Converting to lower case, Removing tokens like <start>, <end>, padding etc.

def bleu1(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)


def bleu4(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)


def generate_text_caption(caption,vocab,max_count=20):
    batch_caption = []
    words_sequence = []
    #print('length of caption is {} and type is {}'.format(len(caption),type(caption)))
    for idx in range(0,len(caption)):
        img_caption = caption[idx]
        #print('length of image caption is', len(img_caption))
        #print('image caption is',img_caption)
        for word_id in img_caption:
            word = vocab.idx2word[word_id]
            if word=="<start>":
                words_sequence = [] 
                continue
            #print('The index is {} and the word is {} '.format(word_id,word))
            if word=="<end>":
                sentence = ' '.join(words_sequence)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                words_sequence = [] 
                break
            words_sequence.append(word)
            if (len(words_sequence)==max_count):
                sentence = ' '.join(words_sequence)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                words_sequence = [] 
    #print('length of batch caption is', len(batch_caption))
    return batch_caption