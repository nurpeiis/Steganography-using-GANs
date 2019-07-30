from metrics.bleu import BLEU
import argparse
import codecs

parser = argparse.ArgumentParser(description='Evaluate BLEU Score of Generated text')
parser.add_argument('--test_data', type=str, default='./decode/final_stego.txt',
                    help='location of test data')
parser.add_argument('--real_data', type=str, default='./LSTM/data/emnlp_news/test.txt',
                    help='location of real data')
parser.add_argument('--gram', type=int, default= 5, 
                    help='Order of BLEU score')
parser.add_argument('--leakGAN', action='store_true', 
                    help='If you want to calculate BLEU score for LSTM-LeakGAN double-layer, else LSTM-LSTM double layer')
args = parser.parse_args()
#Step 1: Read data from files and put them into list
def leakGAN():
    test_sentences = list()
    with codecs.open(args.test_data, 'r',encoding='utf8',errors='ignore') as f:
        for line in f:
            line = line.split(' ', 1)[1] #remove initial EOS
            test_sentences.append(line)
    real_sentences = list()
    with codecs.open(args.real_data,'r',encoding='utf8',errors='ignore') as f:
        for line in f:
            real_sentences.append(line)

    #Step 2: BLEU Score
    print("LSTM - LeakGAN double layer encoding")
    for i in range(1, args.gram + 1):
        bleu = BLEU(test_sentences, real_sentences, i)
        bleu_score = bleu.get_score(ignore=False)
        print("BLEU{} score:{}".format(i,bleu_score))

def LSTM():
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
    print("LSTM - LSTM double layer encoding")
    for i in range(1, args.gram + 1):
        bleu = BLEU(test_sentences, real_sentences, i)
        bleu_score = bleu.get_score(ignore=False)
        print("BLEU{} score:{}".format(i,bleu_score)) 


def main():
    if args.leakGAN:
        print("Helllo")
        leakGAN()
        
    else:
        LSTM()

if __name__ == "__main__":
    main()