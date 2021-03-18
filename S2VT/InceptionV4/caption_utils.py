#Reference: PA4 starter code of CSE251B Winter 2021 & PA4 submission code from our team.
from collections import defaultdict
from nltk.translate.meteor_score import meteor_score, single_meteor_score

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
            if word=="<bos>":
                words_sequence = []
                continue
            #print('The index is {} and the word is {} '.format(word_id,word))
            if word=="<eos>":
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
            words_sequence = []
            #print("Image {} caption curr_sentence: {} result: {}".format(idx,sentence,batch_caption[idx]))
#     if len(batch_caption)!=64:
#         print("CAPTION_UTILS ERROR: Generated caption length {} is lesser than 64".format(len(batch_caption)))
    return batch_caption

def generate_text_sentence(caption,vocab,max_count=20):
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
            if word=="<bos>":
                words_sequence = []
                continue
            #print('The index is {} and the word is {} '.format(word_id,word))
            if word=="<eos>":
                sentence = ' '.join(words_sequence)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                is_processed = True
                words_sequence = []
                break
            words_sequence.append(word)
            if (len(words_sequence)==max_count):
                sentence = ' '.join(words_sequence)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                is_processed = True
                words_sequence = []
        if is_processed == False:
            #print("No caption is added for this image")
            sentence = ' '.join(words_sequence)
            sentence = sentence.lower()
            batch_caption.append(sentence)
            words_sequence = []
            #print("Image {} caption curr_sentence: {} result: {}".format(idx,sentence,batch_caption[idx]))
#     if len(batch_caption)!=64:
#         print("CAPTION_UTILS ERROR: Generated caption length {} is lesser than 64".format(len(batch_caption)))
    return batch_caption

def meteor(all_reference_captions, all_predicted_captions):
    score = 0
    total = len(all_reference_captions)
    for idx in range(0,total):
        reference_captions = all_reference_captions[idx]
        predicted_caption = all_predicted_captions[idx]
        if idx==0:
            print("METEOR SCORE: Reference caption = {}, predicted caption = {}".format(reference_captions,predicted_caption))
        score += single_meteor_score(reference_captions, predicted_caption)                        
    return 100 * (score/total)