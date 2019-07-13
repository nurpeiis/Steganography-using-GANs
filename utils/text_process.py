# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : text_process.py
# @Time         : Created at 2019-05-14
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import nltk
from nltk.tokenize import TweetTokenizer
import os
import torch
import codecs
import random # for unk words

import config as cfg


def get_tokenlized(file):
    """tokenlize the file"""
    tokenlized = list()
    tknzr = TweetTokenizer(reduce_len=True)
    with codecs.open(file,'r',encoding='utf8',errors='ignore') as raw:
        for text in raw:
            text = tknzr.tokenize((text.lower()))
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    """get word set"""
    word_set = list()
    dictc = dict()
    for sentence in tokens:
        for word in sentence:
            if word in dictc:
                dictc[word] += 1
            else:
                dictc[word] = 1
    for key, value in sorted(dictc.items(), key=lambda item: item[1], reverse=True):
        if value > 2:
            word_set.append(key)
    #sort and get only most popular 15000 words
    """
    word_counter = {}
    for word in word_set:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
    popular_words = sorted(word_counter, key = word_counter.get, reverse=True)
    print(popular_words[0])
    word_set = popular_words[:cfg.vocab_size]
    """
    return list(set(word_set))


def get_dict(word_set):
    """get word_index_dict and index_word_dict"""
    word_index_dict = dict()
    index_word_dict = dict()

    index = 2
    word_index_dict[cfg.padding_token] = str(cfg.padding_idx)
    index_word_dict[str(cfg.padding_idx)] = cfg.padding_token
    word_index_dict[cfg.start_token] = str(cfg.start_letter)
    index_word_dict[str(cfg.start_letter)] = cfg.start_token

    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_precess(train_text_loc, test_text_loc=None):
    """get sequence length and dict size"""
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    # with open(oracle_file, 'w') as outfile:
    #     outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return sequence_len, len(word_index_dict) + 1


# ========================================================================
def init_dict():
    """
    Initialize dictionaries of dataset, please note that '0': padding_idx, '1': start_letter.
    Finally save dictionary files locally.
    """
    # image_coco
    tokens = get_tokenlized('dataset/image_coco.txt')
    tokens.extend(get_tokenlized('dataset/testdata/image_coco_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with open('dataset/image_coco_wi_dict.txt', 'w') as dictout:
        dictout.write(str(word_index_dict))
    with open('dataset/image_coco_iw_dict.txt', 'w') as dictout:
        dictout.write(str(index_word_dict))

    #twitter
    tokens = get_tokenlized('dataset/tweets.txt')
    tokens.extend(get_tokenlized('dataset/testdata/tweets_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with codecs.open('dataset/tweets_wi_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(word_index_dict))
    with codecs.open('dataset/tweets_iw_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(index_word_dict))

    #twitter-15000
    tokens = get_tokenlized('dataset/tweets_15000.txt')
    tokens.extend(get_tokenlized('dataset/testdata/tweets_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with codecs.open('dataset/tweets_15000_wi_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(word_index_dict))
    with codecs.open('dataset/tweets_15000_iw_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(index_word_dict))
    #twitter-20000
    tokens = get_tokenlized('dataset/tweets_20000.txt')
    tokens.extend(get_tokenlized('dataset/testdata/tweets_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with codecs.open('dataset/tweets_20000_wi_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(word_index_dict))
    with codecs.open('dataset/tweets_20000_iw_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(index_word_dict))
    #twitter-25000
    tokens = get_tokenlized('dataset/tweets_25000.txt')
    tokens.extend(get_tokenlized('dataset/testdata/tweets_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with codecs.open('dataset/tweets_25000_wi_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(word_index_dict))
    with codecs.open('dataset/tweets_25000_iw_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(index_word_dict))
    
    # emnlp
    tokens = get_tokenlized('dataset/emnlp_news.txt')
    tokens.extend(get_tokenlized('dataset/testdata/emnlp_news_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with codecs.open('dataset/emnlp_news_wi_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(word_index_dict))
    with codecs.open('dataset/emnlp_news_iw_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(index_word_dict))
    

def load_dict(dataset):
    """Load dictionary from local files"""
    iw_path = 'dataset/{}_iw_dict.txt'.format(dataset)
    wi_path = 'dataset/{}_wi_dict.txt'.format(dataset)

    if not os.path.exists(iw_path) or not os.path.exists(iw_path):  # initialize dictionaries
        init_dict()

    with codecs.open(iw_path, 'r', encoding='utf8',errors='ignore') as dictin:
        index_word_dict = eval(dictin.read().strip())
    with codecs.open(wi_path, 'r', encoding='utf8',errors='ignore') as dictin:
        word_index_dict = eval(dictin.read().strip())

    return word_index_dict, index_word_dict


def tensor_to_tokens(tensor, dictionary):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for i, word in enumerate(sent.tolist()):
            if i != 0:
                if word == cfg.padding_idx:
                    break
            if (str(word) in dictionary.keys()):
                sent_token.append(dictionary[str(word)])
            else:
                sent_token.append(dictionary[str(cfg.padding_idx)])
            #print("Sent token: {}".format(sent_token))
        tokens.append(sent_token)
    return tokens


def tokens_to_tensor(tokens, dictionary):
    """transform word tokens to Tensor"""
    tensor = []
    for sent in tokens:
        sent_ten = []
        for i, word in enumerate(sent):
            if word == cfg.padding_token:
                break
            if word in dictionary:
                sent_ten.append(int(dictionary[str(word)]))
            else: 
                sent_ten.append(int(dictionary[str(random.choice(list(dictionary.keys())))]))
        while i < cfg.max_seq_len - 1:
            sent_ten.append(cfg.padding_idx)
            i += 1
        #Strange bug that sometimes only 19 was given, whilst 20 should be the size
        while (len(sent_ten[:cfg.max_seq_len]) < cfg.max_seq_len):
            sent_ten.append(cfg.padding_idx)
        #print("Sent_ten: {}".format(len(sent_ten[:cfg.max_seq_len])))
        tensor.append(sent_ten[:cfg.max_seq_len])
    return torch.LongTensor(tensor)


def padding_token(tokens):
    """pad sentences with padding_token"""
    pad_tokens = []
    for sent in tokens:
        sent_token = []
        for i, word in enumerate(sent):
            if word == cfg.padding_token:
                break
            sent_token.append(word)
        while i < cfg.max_seq_len - 1:
            sent_token.append(cfg.padding_token)
            i += 1
        pad_tokens.append(sent_token)
    return pad_tokens


def write_tokens(filename, tokens):
    """Write word tokens to a local file (For Real data)"""
    with codecs.open(filename, 'w', encoding='utf8',errors='ignore') as fout:
        for sent in tokens:
            fout.write(' '.join(sent))
            fout.write('\n')


def write_tensor(filename, tensor):
    """Write Tensor to a local file (For Oracle data)"""
    with codecs.open(filename, 'w', encoding='utf8',errors='ignore') as fout:
        for sent in tensor:
            fout.write(' '.join([str(i) for i in sent.tolist()]))
            fout.write('\n')
