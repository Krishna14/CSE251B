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