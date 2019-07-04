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

import numpy as np
import random
import time 

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        #I have to change this function
        #Step 1: load the latest model
        with open("gen_ADV_00199.pt", 'rb') as f:
            model = torch.load(f)
        if cfg.CUDA:
            model.cuda()
        else:
            model.cpu()
        corpus = self.index_word_dict
        ntokens = len(corpus.dictionary) #cfg.vocab_size #number of tokens is equal to vocab size
        hidden = model.init_hidden(1)
        input = Variable(torch.rand(1, 1).mul(ntokens).long()) # look into this later
        if args.cuda:
            input.data = input.data.cuda()
        ###############################################################################
        # Secret Text Modification

        def string2bins(bit_string, n_bins):
            n_bits = int(math.log(n_bins, 2))
            return [bit_string[i:i+n_bits] for i in range(0, len(bit_string), n_bits)]
        secret_file =  "secret_file.txt"
        secret_file = open(secret_file, 'r')
        secret_data = secret_file.read()
        bit_string = ''.join(bin(ord(letter))[2:].zfill(8) for letter in secret_data)
         # secret_text = np.random.choice(range(args.bins), args.words)
        bins = 4   #4 bins
        secret_text = [int(i,2) for i in string2bins(bit_string, bins)] #convert to binary
        ###############################################################################

        def get_common_tokens(n):
            dictionary = corpus.dictionary.word_count
            d = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            common_tokens = [item[0] for item in d]
            common_tokens = common_tokens[0:n]
            return common_tokens

        if bins >= 2:
            common_bin_indices = np.random.choice(range(bins), size=args.common_bin_factor, replace=False) 

            ntokens = len(corpus) 
            tokens = list(range(ntokens)) # * args.replication_factor

            random.shuffle(tokens)
            words_in_bin = int(len(tokens) / args.bins) 

            # common words
            common_tokens = get_common_tokens(args.num_tokens)
            remove_words = ['<user>','rt']
            common_tokens = list(set(common_tokens) - set(remove_words))
            # common_tokens = [':',",",'.','"','to','a','the','in','of','and','is']
            common_tokens_idx = [corpus.dictionary.word2idx[word] for word in common_tokens]

            bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)] # words to keep in each bin...
            bins = [list(set(bin_) | set(common_tokens_idx)) if bins.index(bin_) in common_bin_indices else bin_ for bin_ in bins]

            zero = [list(set(tokens) - set(bin_)) for bin_ in bins]

            print('Finished Initializing')
            print('time: {:5.2f}s'.format(time.time() - epoch_start_time))
            print('-' * 89)
            out_file = 'test1.txt'
            with open(out_file, 'w') as outf:
        """
                def forward(self, idx, inp, work_hidden, mana_hidden, feature, real_goal, no_log=False, train=False):
                    
                Embeds input and sample on token at a time (seq_len = 1)

                :param idx: index of current token in sentence
                :param inp: [batch_size]
                :param work_hidden: 1 * batch_size * hidden_dim
                :param mana_hidden: 1 * batch_size * hidden_dim
                :param feature: 1 * batch_size * total_num_filters, feature of current sentence
                :param real_goal: batch_size * goal_out_size, real_goal in LeakGAN source code
                :param no_log: no log operation
                :param train: if train

                :return: out, cur_goal, work_hidden, mana_hidden
                    - out: batch_size * vocab_size
                    - cur_goal: batch_size * 1 * goal_out_size
        """
                w = 0 
                i = 1
                bin_sequence_length = len(secret_text[:]) # 85
                print("bin sequence length", bin_sequence_length)
                while i <= bin_sequence_length:
                    """
                    # Get feature
                    if if_sample:
                        dis_inp = samples[:, :seq_len]
                    else:  # to get feature and goal
                        dis_inp = torch.zeros(batch_size, seq_len).long()
                        if i > 0:
                            dis_inp[:, :i] = sentences[:, :i]  # cut sentences
                            leak_inp = sentences[:, i - 1]

                    if self.gpu:
                        dis_inp = dis_inp.cuda()
                        leak_inp = leak_inp.cuda()
                    feature = dis.get_feature(dis_inp).unsqueeze(0)  # !!!note: 1 * batch_size * total_num_filters

                    feature_array[:, i, :] = feature.squeeze(0)

                    # Get output of one token
                    # cur_goal: batch_size * 1 * goal_out_size
                    out, cur_goal, work_hidden, mana_hidden = self.forward(i, leak_inp, work_hidden, mana_hidden, feature,
                                                                        real_goal, no_log=no_log, train=train)
                    leak_out_array[:, i, :] = out

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
                    goal_array[:, i, :] = cur_goal.squeeze(1)
                    if i > 0 and i % self.step_size == 0:
                        real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                        if i / self.step_size == 1:
                            real_goal += self.goal_init[:batch_size, :]

                    # Sample one token
                    if not no_log:
                        out = torch.exp(out)
                    out = torch.multinomial(out, 1).view(-1)  # [batch_size] (sampling from each row)
                    samples[:, i] = out.data
                    leak_inp = out
            """
                    batch_size, seq_len = sentences.size()
                    epoch_start_time = time.time()
                    inp = leak_inp = torch.LongTensor([cfg.start_letter] * batch_size)
                    work_hidden = self.init_hidden(cfg.batch_size)
                    mana_hidden = self.init_hidden(cfg.batch_size)
                    feature_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))
                    goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))
                    leak_out_array = torch.zeros((batch_size, seq_len + 1, self.vocab_size))
                    output, hidden = model()

                    if self.gpu:
                        feature_array = feature_array.cuda()
                        goal_array = goal_array.cuda()
                        leak_out_array = leak_out_array.cuda()
                    # print("bin: ", bin_)
                    zero_index = zero[secret_text[:][i-1]]
                    zero_index = torch.LongTensor(zero_index) 

                    word_weights = output.squeeze().data.div(args.temperature).exp().cpu() 

                    word_weights.index_fill_(0, zero_index, 0)
                    word_idx = torch.multinomial(word_weights, 1)[0]
                
                    input.data.fill_(word_idx)
                    word = corpus[word_idx]

                    if word not in common_tokens:
                        i += 1
                    w += 1
                    word = word.encode('ascii', 'ignore').decode('ascii')
                    outf.write(word + ('\n' if i % 20 == 19 else ' '))
                    log_interval = 100
                    if i % log_interval == 0:
                        print("total number of words", w)
                        print("total length of secret", i)
                        print('| Generated {}/{} words'.format(i, len(secret_text)))
                        print('-' * 89)

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
