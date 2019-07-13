import codecs 
from nltk.tokenize import TweetTokenizer

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
    return list(set(word_set))


def get_dict(word_set):
    """get word_index_dict and index_word_dict"""
    word_index_dict = dict()
    index_word_dict = dict()

    index = 0

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
    dictc = dict()
    new_list = test_tokens + train_tokens
    for words in new_list:
        for word in words:
            if word in dictc:
                dictc[word] += 1
            else:
                dictc[word] = 1
    #print(sorted(dictc.items(), key = lambda kv:(kv[1], kv[0])))
    i = 0
    for key, value in sorted(dictc.items(), key=lambda item: item[1], reverse=True):
        if value > 2:
            print("%s: %s" % (key, value))
            i += 1
    print(i)
    word_set = get_word_list(train_tokens + test_tokens)
    word_index_dict, index_word_dict = get_dict(word_set)
    with codecs.open(train_text_loc+'_wi_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(word_index_dict))
    with codecs.open(train_text_loc + '_iw_dict.txt', 'w', encoding='utf8',errors='ignore') as dictout:
        dictout.write(str(index_word_dict))

text_precess("../tweets_25000.txt", "tweets_test.txt")