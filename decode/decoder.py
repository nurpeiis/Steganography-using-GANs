import numpy as np
import pickle
import binascii

lstm_key2_file = 'lstm_key2.txt'
word2idx_2 = 'word2idx_2.txt'
with open(lstm_key2_file, 'rb') as f:
    key2 = pickle.load(f)
with open(word2idx_2, 'rb') as f:
    word2idx2 = pickle.load(f)
with open('final_1.txt', 'r') as f:
    words = f.read().split()
ids = list()
#convert into ids
for i in range(0, len(words)):
    idx = word2idx2.get(words[i])
    if idx != None:
        ids.append(idx)


#use key2 to get to binary representation
bins = list()
for id1 in ids:
    for i in range(0, len(key2)):
        for j in range(0, len(key2[i])):
            if (id1 == key2[i][j]):
                bins.append(i)
                break
#convert into binary
def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'
def from_final_to_intermediate(bits, bins):
    text = [int('0b'+ bits[i : i + bins], 2) for i in range(0, len(bits), bins)]
    return text
binary = ''
for bin1 in bins:
    b = '{0:02b}'.format(bin1)
    binary += b
#divide into 13 bits 
text = from_final_to_intermediate(binary, 13)
with open("idx2word_1.txt", 'rb') as f:
    idx2word1 = pickle.load(f)
for i in range(0, len(text)):
    text[i] = idx2word1[text[i]]
print("Text: ", text)

def decode(file, key_file, word2idx_file):
    lstm_key2_file = key_file
    word2idx_2 = word2idx_file
    with open(lstm_key2_file, 'rb') as f:
        key2 = pickle.load(f)
    with open(word2idx_2, 'rb') as f:
        word2idx2 = pickle.load(f)
    with open(file, 'r') as f:
        words = f.read().split()
    ids = list()
    #convert into ids
    for i in range(0, len(words)):
        idx = word2idx2.get(words[i])
        if idx != None:
            ids.append(idx)


    #use key2 to get to binary representation
    bins = list()
    for id1 in ids:
        for i in range(0, len(key2)):
            for j in range(0, len(key2[i])):
                if (id1 == key2[i][j]):
                    bins.append(i)
                    break
    print(bins)
    #convert into binary
    def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
        n = int(bits, 2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'
    binary = ''
    for bin1 in bins:
        b = '{0:{fill}12b}'.format(bin1, fill='0')
        binary += b
    return binary

if decode('intermediate.txt', 'lstm_key1.txt', 'word2idx_1.txt') == '111011100101000111000011110111101111110111000110011010110110':
    print("Hello")