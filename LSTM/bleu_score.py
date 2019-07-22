from metrics.bleu import BLEU
import argparse
import codecs
import re

parser = argparse.ArgumentParser(description='Evaluate BLEU Score of Generated text')
parser.add_argument('--test_data', type=str, default='./outputs/stegoemnlp_news.txt',
                    help='location of test data')
parser.add_argument('--real_data', type=str, default='./data/emnlp_news/test.txt',
                    help='location of real data')
parser.add_argument('--gram', type=int, default= 5, 
                    help='Order of BLEU score')
args = parser.parse_args()
#Step 1: Read data from files and put them into list
test_sentences = list()
with open(args.test_data, 'r') as f:
    for line in f:
       test_sentences.append(line.replace("<eos>", ""))
temp = list()
for i in range(0, len(test_sentences)):
    if test_sentences[i] != '\n':
        temp.append(test_sentences[i])
test_sentences = temp
print(test_sentences)


real_sentences = list()
with codecs.open(args.real_data,'r',encoding='utf8',errors='ignore') as f:
    for line in f:
        real_sentences.append(line)

#Step 2: BLEU Score
bleu = BLEU(test_sentences, real_sentences, args.gram)
bleu3_score = bleu.get_score(ignore=False)
print("BLEU{} score:{}".format(args.gram,bleu3_score))
for i in range(1, args.gram):
    bleu = BLEU(test_sentences, real_sentences, i)
    bleu_score = bleu.get_score(ignore=False)
    print("BLEU{} score:{}".format(i,bleu_score))