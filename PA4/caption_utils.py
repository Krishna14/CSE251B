################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocotools.coco import COCO


# See this for input references - https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu
# A Caption should be a list of strings.
# Reference Captions are list of actual captions - list(list(str))
# Predicted Caption is the string caption based on your model's output - list(str)
# Make sure to process your captions before evaluating bleu scores -
# Converting to lower case, Removing tokens like <start>, <end>, padding etc.

def bleu1(all_reference_captions, all_predicted_captions):
   # print(type(all_reference_captions),type(all_predicted_captions))
    bleu1_score = 0
    total = len(all_reference_captions)
    for idx in range(0,total):
        reference_captions = all_reference_captions[idx] #list(str)
        predicted_caption = all_predicted_captions[idx] #string
        if idx==0:
            print("Reference captions = {}, predicted caption = {}".format(reference_captions,predicted_caption))
        bleu1_score += sentence_bleu(reference_captions, predicted_caption,
                    weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    return 100 * (bleu1_score/total)
    #return bleu1_score
    
# def bleu1(all_reference_captions, all_predicted_captions):
#     bleu1_score = sentence_bleu(all_reference_captions, all_predicted_captions,
#                      weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
#     return bleu1_score

def bleu4(all_reference_captions, all_predicted_captions):
    bleu4_score = 0
    total = len(all_reference_captions)
    for idx in range(0,total):
        reference_captions = all_reference_captions[idx]
        predicted_caption = all_predicted_captions[idx]
        bleu4_score += sentence_bleu(reference_captions, predicted_caption,
                              weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)
    return 100 * (bleu4_score/total)
    #return bleu4_score

    
def get_true_captions(img_ids,coco):
    batch_captions = []
    for img_id in img_ids:
        image_metadata = coco.imgToAnns[img_id]
        captions = []
        for metadata in image_metadata:
            captions.append(metadata['caption'].lower().split()) 
        batch_captions.append(captions)
    return batch_captions

def generate_text_caption(caption,vocab,max_count=20):
    batch_caption = []
    words_sequence = []
    #print('length of caption is {} and type is {}'.format(len(caption),type(caption)))
    for idx in range(0,len(caption)):
        img_caption = caption[idx]
        is_processed = False
        #print('length of image caption is', len(img_caption))
        #print('image caption is',img_caption)
        for word_id in img_caption:
            word = vocab.idx2word[word_id].lower()
            if word=="<start>":
                words_sequence = [] 
                continue
            #print('The index is {} and the word is {} '.format(word_id,word))
            if word=="<end>":
               # sentence = ' '.join(words_sequence)
                batch_caption.append(words_sequence)
                is_processed = True
                words_sequence = [] 
                break
            words_sequence.append(word)
            if (len(words_sequence)==max_count):
               # sentence = ' '.join(words_sequence)
                #sentence = sentence.lower().split()
                batch_caption.append(words_sequence)
                is_processed = True
                words_sequence = []
        if is_processed == False:
            #print("No caption is added for this image")
            #sentence = ' '.join(words_sequence)
            #sentence = sentence.lower()
            batch_caption.append(words_sequence)
            #print("Image {} caption curr_sentence: {} result: {}".format(idx,sentence,batch_caption[idx]))
#     if len(batch_caption)!=64:
#         print("CAPTION_UTILS ERROR: Generated caption length {} is lesser than 64".format(len(batch_caption)))
    return batch_caption