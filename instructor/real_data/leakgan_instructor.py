# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : leakgan_instructor.py
# @Time         : Created at 2019-06-05
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.optim as optim

import time
import math
import random
import numpy as np

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from metrics.bleu import BLEU
from models.LeakGAN_D import LeakGAN_D
from models.LeakGAN_G import LeakGAN_G
from utils import rollout
from utils.data_loader import GenDataIter, DisDataIter
from utils.text_process import tensor_to_tokens, write_tokens


class LeakGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(LeakGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = LeakGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, cfg.goal_size, cfg.step_size, cfg.CUDA)
        self.dis = LeakGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        if (cfg.CUDA):
            self.dis.cuda()
            self.gen.cuda()
        self.init_model()

        # optimizer
        mana_params, work_params = self.gen.split_params()
        mana_opt = optim.Adam(mana_params, lr=cfg.gen_lr)
        work_opt = optim.Adam(work_params, lr=cfg.gen_lr)

        self.gen_opt = [mana_opt, work_opt]
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis))
        self.dis_data = DisDataIter(self.gen_data.random_batch()['target'], self.oracle_data.random_batch()['target'])

        # Metrics
        self.bleu3 = BLEU(test_text=tensor_to_tokens(self.gen_data.target, self.index_word_dict),
                          real_text=tensor_to_tokens(self.test_data.target, self.index_word_dict),
                          gram=3)

    def _run(self):
        for inter_num in range(cfg.inter_epoch):
            self.log.info('>>> Interleaved Round %d...' % inter_num)
            self.sig.update()  # update signal
            if self.sig.pre_sig:
                # =====DISCRIMINATOR PRE-TRAINING=====
                if not cfg.dis_pretrain:
                    self.log.info('Starting Discriminator Training...')
                    self.train_discriminator(cfg.d_step, cfg.d_epoch)
                    if cfg.if_save and not cfg.if_test:
                        torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                        print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

                # =====GENERATOR MLE TRAINING=====
                if not cfg.gen_pretrain:
                    self.log.info('Starting Generator MLE Training...')
                    self.pretrain_generator(cfg.MLE_train_epoch)
                    if cfg.if_save and not cfg.if_test:
                        torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                        print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))
            else:
                self.log.info('>>> Stop by pre_signal! Skip to adversarial training...')
                break

        # =====ADVERSARIAL TRAINING=====
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (str(self.cal_metrics(fmt_str=True))))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step)  # Generator
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator

                if adv_epoch % cfg.adv_log_step == 0:
                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break

    def _test(self):
        print('>>> Begin test...')
        torch.nn.Module.dump_patches = True
        epoch_start_time = time.time() 
        # Set the random seed manually for reproducibility.
        seed = 1111
        torch.manual_seed(seed)
        #Step 1: load the latest model
        with open("instructor/real_data/gen_ADV_00028.pt", 'rb') as f:
            self.gen.load_state_dict(torch.load(f))
        print("Finish Loading")
        self.gen.eval()
        corpus = self.index_word_dict
        #print("Corpus: {}".format(corpus))

        def string2bins(bit_string, n_bins):
            n_bits = int(math.log(n_bins, 2))
            return [bit_string[i:i+n_bits] for i in range(0, len(bit_string), n_bits)]
        #Step 1: Get secret data 
        secret_file =  "instructor/real_data/secret_file.txt"
        secret_file = open(secret_file, 'r')
        secret_data = secret_file.read()
        #print("Secret Data: {}".format(secret_data))
        #Step 2: Compress string into binary string
        bit_string = ''.join(bin(ord(letter))[2:].zfill(8) for letter in secret_data)
        #print(bit_string)
        #bit_string = '1110111001010001110000111101111011111101110001100110101101100111'
        #In the first step we will use 8 bins to convert so that we can convert 64 bits into 8 word 
        bins_num = 4   
        secret_text = [int(i,2) for i in string2bins(bit_string, bins_num)] #convert to bins 
        def get_common_tokens(n):
            dictionary = self.word_index_dict
            d = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            common_tokens = [item[0] for item in d]
            common_tokens = common_tokens[0:n]
            return common_tokens

        if bins_num >= 2:
            common_bin_indices = np.random.choice(range(bins_num), size=0, replace=False) 

            ntokens = len(corpus) 
            tokens = list(range(ntokens)) # * args.replication_factor
            #print(ntokens)
            random.shuffle(tokens)
            words_in_bin = int(len(tokens) / bins_num) 
            #There are left over words
            leftover = int(len(tokens) % bins_num) 
            #print(leftover)
            # common words
            num_tokens = 1
            common_tokens = get_common_tokens(num_tokens)
            #remove_words = ['<user>','rt']
            #print(common_tokens)
            #common_tokens = list(set(common_tokens) - set(remove_words))
            common_tokens.append('EOS')
            # common_tokens = [':',",",'.','"','to','a','the','in','of','and','is']
            common_tokens_idx = [self.word_index_dict[word] for word in common_tokens]
            #Random sampling
            bins = list()
            tokens_temp = tokens
            for i in range(0, bins_num):
                bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)]
                #tokens = list(set(tokens) - set(bins[i]))
                #print("Tokens size: {}".format(len(tokens)))
            for i in range(0, leftover):
                bins[i].append(tokens[i])
            #print(len(bins))
            #print(len(bins[0]))
            bins = [list(set(bin_) | set(common_tokens_idx)) if bins.index(bin_) in common_bin_indices else bin_ for bin_ in bins]
            #print("Bins:{}".format(bins))
            #print(len(bins))
            #print(len(bins[0]))
            zero = [list(set(tokens_temp) - set(bin_)) for bin_ in bins]
            #print("Zero: {}".format(zero[1]))
            print('Finished Initializing')
            print('time: {:5.2f}s'.format(time.time() - epoch_start_time))
            print('-' * 89)
            out_file = 'test5.txt'
            w = 0 
            i = 1
            bin_sequence_length = len(secret_text[:]) # 85
            print("bin sequence length", bin_sequence_length)
            batch_size = cfg.batch_size
            seq_len = cfg.max_seq_len
            sentences = torch.zeros((batch_size, seq_len))
            feature_array = torch.zeros((batch_size, seq_len + 1, self.gen.goal_out_size))
            goal_array = torch.zeros((batch_size, seq_len + 1, self.gen.goal_out_size))
            leak_out_array = torch.zeros((batch_size, seq_len + 1, cfg.vocab_size))
            samples = torch.zeros(batch_size, seq_len + 1).long()
            work_hidden = self.gen.init_hidden(batch_size)
            mana_hidden = self.gen.init_hidden(batch_size)
            leak_inp = torch.LongTensor([cfg.start_letter] * batch_size)
            # dis_inp = torch.LongTensor([start_letter] * batch_size)
            real_goal = self.gen.goal_init[:batch_size, :]
            if torch.cuda.is_available():
                feature_array = feature_array.cuda()
                goal_array = goal_array.cuda()
                leak_out_array = leak_out_array.cuda()
            goal_array[:, 0, :] = real_goal  # g0 = goal_init
            if_sample = True
            no_log = False
            j = i
            print("secret_text:{}".format(secret_text))
            print("secret_text []:{}".format(secret_text[:]))
            print("secret_text:{}".format(secret_text[:][1]))
            print("zero:{}".format(zero[secret_text[:][1]]))
            #samples = torch.zeros(batch_size * batch_size, self.gen.max_seq_len).long()  # larger than num_samples
            index = cfg.start_letter
            while i <= seq_len:
                
                # print("bin: ", bin_)
                if if_sample:
                    dis_inp = samples[:, :bin_sequence_length]
                else:  # to get feature and goal
                    dis_inp = torch.zeros(batch_size, bin_sequence_length).long()
                    if i > 1:
                        dis_inp[:, :i - 1] = samples[:, :i - 1]  # cut sentences
                        leak_inp = samples[:, i - 2]
                
                if torch.cuda.is_available():
                    dis_inp = dis_inp.cuda()
                    leak_inp = leak_inp.cuda()
                feature = self.dis.get_feature(dis_inp).unsqueeze(0)  
                #print(feature)
                feature_array[:, i - 1, :] = feature.squeeze(0)
                out, cur_goal, work_hidden, mana_hidden = self.gen(index, leak_inp, work_hidden, mana_hidden, feature,
                                                            real_goal, no_log=no_log, train=False)
                #print("Out size: {}".format(out.size()))
                leak_out_array[:, i - 1, :] = out
                #print("New Method")
                goal_array[:, i, :] = cur_goal.squeeze(1)
                # =====My implement according to paper=====
                # Update real_goal and save goal
                # if 0 < i < 4:  # not update when i=0
                #     real_goal = torch.sum(goal_array, dim=1)  # num_samples * goal_out_size
                # elif i >= 4:
                #     real_goal = torch.sum(goal_array[:, i - 4:i, :], dim=1)
                # if i > 0:
                #     goal_array[:, i, :] = cur_goal.squeeze(1)  # !!!note: save goal after update last_goal
                # =====LeakGAN origin=====
                # Save goal and update real_goal
                if i > 0 and i % self.gen.step_size == 0:
                    real_goal = torch.sum(goal_array[:, i - 3: i + 1, :], dim=1)
                    if i / self.gen.step_size == 1:
                        real_goal += self.gen.goal_init[:batch_size, :]
                # Sample one token
                if not no_log:
                    out = torch.exp(out)
                #print("Out before: {}".format(out))
                #out = torch.multinomial(out, 1).view(-1)  # [batch_size] (sampling from each row)
                #print("Out after: {}".format(out))
                #print("Out one size: {}".format(out.size()))
                #print("Out sample: {}".format(out[0]))
                zero_index = zero[secret_text[:][i-1]] #indecies that has to be zeroed, as they are not in the current bin
                #zero_index.append(0)
                zero_index = torch.LongTensor(zero_index)
                if torch.cuda.is_available():
                    zero_index = zero_index.cuda()
                temperature = 1.5
                word_weights = out
                #print("Word weights: {}".format(word_weights[0]))
                #print("Word weights size: {}".format(word_weights.size()))
                #word_weights = word_weights.cpu()
                word_weights = word_weights.index_fill_(1, zero_index, 0) #make all the indecies zero if they are not in the bin
                #print("Word weights: {}".format(word_weights[0]))
                #print("Word weights size: {}".format(word_weights.size()))
                word_weights = torch.multinomial(word_weights, 1).view(-1) #choose one word with highest probability for each sample
                #print("Out after: {}".format(word_weights))
                samples[:, i] = word_weights
                leak_inp = word_weights
                word_idx = word_weights[0]
                #print("Word idx: {}".format(int(word_idx)))
                word = corpus[str(int(word_idx))]
                #print("Word: {}".format(word))
                #if word not in common_tokens:
                i += 1
                w += 1
                word = word.encode('ascii', 'ignore').decode('ascii')
                #outf.write(word + ('\n' if i % 20 == 19 else ' '))
            #samples = samples[:, :seq_len]
            leak_out_array = leak_out_array[:, :seq_len, :]
            tokens = []
            write_tokens(out_file, tensor_to_tokens(samples, self.index_word_dict))
            print("Samples: {}".format(samples))

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pretraining for the gen

        - gen_opt: [mana_opt, work_opt]
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                pre_mana_loss = 0
                pre_work_loss = 0

                # =====Train=====
                for i, data in enumerate(self.oracle_data.loader):
                    inp, target = data['input'], data['target']
                    if cfg.CUDA:
                        inp, target = inp.cuda(), target.cuda()

                    mana_loss, work_loss = self.gen.pretrain_loss(target, self.dis)
                    self.optimize_multi(self.gen_opt, [mana_loss, work_loss])
                    pre_mana_loss += mana_loss.data.item()
                    pre_work_loss += work_loss.data.item()
                pre_mana_loss = pre_mana_loss / len(self.oracle_data.loader)
                pre_work_loss = pre_work_loss / len(self.oracle_data.loader)

                # =====Test=====
                if epoch % cfg.pre_log_step == 0:
                    self.log.info('[MLE-GEN] epoch %d : pre_mana_loss = %.4f, pre_work_loss = %.4f, %s' % (
                        epoch, pre_mana_loss, pre_work_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step, current_k=0):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        adv_mana_loss = 0
        adv_work_loss = 0
        for step in range(g_step):
            with torch.no_grad():
                gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis,
                                              train=True)  # !!! train=True, the only place
                inp, target = self.gen_data.prepare(gen_samples, gpu=cfg.CUDA)

            # =====Train=====
            rewards = rollout_func.get_reward_leakgan(target, cfg.rollout_num, self.dis,
                                                      current_k).cpu()  # reward with MC search
            mana_loss, work_loss = self.gen.adversarial_loss(target, rewards, self.dis)

            # update parameters
            self.optimize_multi(self.gen_opt, [mana_loss, work_loss])
            adv_mana_loss += mana_loss.data.item()
            adv_work_loss += work_loss.data.item()
        # =====Test=====
        self.log.info('[ADV-GEN] adv_mana_loss = %.4f, adv_work_loss = %.4f, %s' % (
            adv_mana_loss / g_step, adv_work_loss / g_step, self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phrase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.oracle_data.target
            neg_samples = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
            self.dis_data.reset(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # =====Train=====
                d_loss, train_acc = self.train_dis_epoch(self.dis, self.dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # =====Test=====
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phrase, step, d_loss, train_acc))

    def cal_metrics(self, fmt_str=False):
        self.gen_data.reset(self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis))
        self.bleu3.test_text = tensor_to_tokens(self.gen_data.target, self.index_word_dict)
        bleu3_score = self.bleu3.get_score(ignore=False)

        with torch.no_grad():
            gen_nll = 0
            for data in self.oracle_data.loader:
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()
                loss = self.gen.batchNLLLoss(target, self.dis)
                gen_nll += loss.item()
            gen_nll /= len(self.oracle_data.loader)

        if fmt_str:
            return 'BLEU-3 = %.4f, gen_NLL = %.4f,' % (bleu3_score, gen_nll)
        return bleu3_score, gen_nll

    def _save(self, phrase, epoch):
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phrase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phrase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis)
        write_tokens(save_sample_path, tensor_to_tokens(samples, self.index_word_dict))
