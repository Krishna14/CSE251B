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

# BLEU1 score is computed here
def bleu1(all_reference_captions, all_predicted_captions):
    """
        Here, we compute the BLEU score for the reference and predicted captions
    """
    bleu1_score = 0
    total = len(all_reference_captions)
    for i in range(total):
        reference_caption = all_reference_captions[i]
        predicted_caption = all_predicted_captions[i]
        bleu1_score += sentence_bleu(reference_caption, predicted_caption,
                                     weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        
    return 100 * (bleu1_score / total)

# BLEU4 score is computed here
def bleu4(all_reference_captions, all_predicted_captions):
    """
        bleu4 is used to compute the modified BLEU 4 score
    """
    bleu4_score = 0
    total = len(all_reference_captions)
    for i in range(total):
        reference_caption = all_reference_captions[i]
        predicted_caption = all_predicted_captions[i]
        bleu4_score += sentence_bleu(reference_caption, predicted_caption,
                                     weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)
    
    return 100 * (bleu4_score / total)

def get_true_captions(img_ids, coco):
    """
        Compute the captions for each batch of images
    """
    batch_captions = []
    for img_id in img_ids:
        image_metadata = coco.imgToAnns[img_id]
        # Here, we compute the image metadata
        #print("Image metadata is {}".format(image_metadata))
        captions = []
        for metadata in image_metadata:
            captions.append(metadata['caption'].lower())
        batch_captions.append(captions)
    return batch_captions

def generate_text_caption(caption, vocab, max_count=20):
    """
        Here, we generate the text caption for the given batch of images
    """
    batch_caption = []
    words_sequence = []
    # We select a particular caption from the list of captions that we have
    for idx in range(0, len(caption)):
        img_caption = caption[idx]
        is_processed = False
        # assert
        assert len(img_caption) <= max_count, "Length of image caption = {}".format(len(img_caption))
        for word_id in img_caption:
            word = vocab.idx2word[word_id]
            if word == "<start>":
                words_sequence = []
                continue
            if word == "<end>":
                sentence = ' '.join(words_sequence)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                is_processed = True
                words_sequence = []
                break
            words_sequence.append(word)
            if(len(words_sequence) == max_count):
                sentence = ' '.join(words_sequence).lower()
                batch_caption.append(sentence)
                is_processed = True
                words_sequence = []
        if is_processed == False:
            print("No caption is generated for image at index = {}".format(idx))
            sentence = ' '.join(words_sequence).lower()
            batch_caption.append(sentence)
        #else:
            #print("Caption generated for image at index = {} is {}".format(idx, sentence))
    
    return batch_caption
