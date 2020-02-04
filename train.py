#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:49:13 2019

@author: s1583620
"""

from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
from models.model import *
from torch.nn import NLLLoss
import matplotlib.pyplot as plt
import os

speaker = SpeakingAgent().to(DEVICE)
listener = ListeningAgent().to(DEVICE)
spk_optimizer = OPTIMISER(speaker.parameters(), lr=LEARNING_RATE)
lis_optimizer = OPTIMISER(listener.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)

def cal_correct_preds(data_batch, data_candidate, pred_idx):
    '''
        Use to calculate the reward or the valid accuracy. As it is possible that
        there are multiple same elements in one row of data_candidate, we will
        check the predicting object rather than index to decide whether it is correct
    '''
    batch_size = data_batch.shape[0]
    cnt_correct = 0
    idx_correct = torch.zeros((batch_size,)).to(DEVICE)
    for i in range(batch_size):
        if data_candidate[i][pred_idx[i]]==data_batch[i]:
            cnt_correct += 1
            idx_correct[i] = 1
    return cnt_correct, idx_correct



def train_phaseA(speaker, spk_optimizer, data_for_spk, clip=CLIP):
    '''
        After re-initialization of the speaker, we should use the D[t-1] to pre-train
        it to make sure it have the knowledge from its predecesors.
        Input:
            data_for_spk is a dictionary, data_for_spk['data'] is the x,
            data_for_spk['msg'] is the y, both of which has size BATCH_SIZE.
            msg is on-hot vector.

            this value should be designed based on the learning curve of speaker.
    '''
    speaker.train()
    spk_optimizer.zero_grad()
    spk_loss_fun = nn.CrossEntropyLoss()

    X = data_for_spk['data']
    Y = data_for_spk['msg']
    Y_hat = Y.transpose(0,1).argmax(dim=2)
    msg, _, _, Y_hiddens = speaker(X)
    spk_loss = spk_loss_fun(Y_hiddens.transpose(0,1).transpose(1,2), Y_hat)
    spk_loss.backward()
    nn.utils.clip_grad_norm_(speaker.parameters(), clip)
    spk_optimizer.step()

    acc_cnt = 0
    for i in range(X.shape[0]):
        Y_pred = msg.transpose(0,1).argmax(dim=2)
        if (Y_pred[i]==Y_hat[i]).sum()==ATTRI_SIZE:
            acc_cnt += 1

    return acc_cnt/X.shape[0]




def train_phaseB(speaker, listener, spk_optimizer, lis_optimizer, train_batch, train_candidates,
                sel_idx_train, exp_ratio = 1, rwd_comp = False, update='BOTH', clip=CLIP):
    '''
        Phase B: playing the game and update parameters in speaker or/and listener
        At the beginning of Phase B, we should re-initialize the listener.
    '''
    speaker.train()
    listener.train()
    lis_loss_fun = nn.CrossEntropyLoss()

    spk_optimizer.zero_grad()
    lis_optimizer.zero_grad()

    true_idx = torch.tensor(sel_idx_train).to(DEVICE)

            # =========== Forward process =======
    msg, spk_log_prob, spk_entropy, _ = speaker(train_batch)
    pred_vector = listener(train_candidates, msg)
    lis_entropy = -(F.softmax(pred_vector)*F.log_softmax(pred_vector)).sum(dim=1)
    lis_log_prob = F.log_softmax(pred_vector.max(dim=1)[0])
    pred_idx = F.softmax(pred_vector).argmax(dim=1)
    reward, reward_vector = cal_correct_preds(train_batch, train_candidates, pred_idx)


    if rwd_comp == True:
        comp_p, comp_s = compos_cal_inner(msg, train_batch)
        reward_vector *= comp_p

            # ========== Perform backpropatation ======
    #lis_loss = lis_loss_fun(pred_vector, true_idx.long().detach())
    lis_loss = -((reward_vector.detach()*lis_log_prob).mean() + 0.05*exp_ratio*lis_entropy.mean())
    lis_loss.backward()

    if MSG_MODE == 'REINFORCE':
        spk_loss = -((reward_vector.detach()*spk_log_prob).mean() + 0.1*exp_ratio*spk_entropy.mean())
        spk_loss.backward()
    elif MSG_MODE == 'SCST':
        speaker.eval()
        listener.eval()

        msg_, spk_log_prob_, _, _ = speaker(train_batch)
        pred_vector_ = listener(train_candidates, msg_)
        pred_idx_ = F.softmax(pred_vector_).argmax(dim=1)
        _, reward_vector_ = cal_correct_preds(train_batch, train_candidates, pred_idx_)

        speaker.train()
        listener.train()

        spk_loss = -(((reward_vector.detach()-reward_vector_.detach())*spk_log_prob).mean() + 0.1*spk_entropy.mean())
        spk_loss.backward()
    elif MSG_MODE == 'GUMBEL':
        spk_loss = lis_loss

            # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(speaker.parameters(), clip)
    nn.utils.clip_grad_norm_(listener.parameters(), clip)

    if update == 'BOTH':
        spk_optimizer.step()
        lis_optimizer.step()
    elif update == 'SPEAKER':
        spk_optimizer.step()
        lis_optimizer.zero_grad()
    elif update == 'LISTENER':
        spk_optimizer.zero_grad()
        lis_optimizer.step()
    else:
        print('Please input "BOTH", "SPEAKER" or "LISTENER" for the train_epoch function')

    # =========== Result Statistics ==============

    return reward, spk_loss.item(), lis_loss.mean().item()


def train_phaseC(speaker, listener, train_batch, train_candidates, sel_idx_train, rwd_filter = False):
    '''
        Phase C of the training procedure. Here we assume both the speaker and listener are well trained.
        We only let two agents play the game to generate the data for next generation.
        Input:
            rwd_filter is used to control whether we use reward to impose bias to those 'precise' languages.
        Output:
            A dictionary of data to train the speaker.
    '''
    with torch.no_grad():
        speaker.train(True)  # Here we use train model because we want to sample from posterior, not argmax
        listener.train(True) # Here we use train model because we want to sample
        data_for_spk = {}
        msg, _, _, _ = speaker(train_batch)
        # ============== Use rewards to change those non_accurate pairs =================
        if rwd_filter == True:
            pred_vector = listener(train_candidates, msg)
            pred_idx = F.softmax(pred_vector).argmax(dim=1)
            _, rewards = cal_correct_preds(train_batch, train_candidates, pred_idx)

            msg = msg.transpose(0,1)    # Change the size to [N_B, ATTRI_SIZE, MSG_VOCSIZE]
            new_msg = []
            for i in range(rewards.shape[0]):
                if rewards[i] == 1:         # If this obj-msg pair can correctly play the game, use this pair
                    new_msg.append(msg[i])
                else:                       # If this obj-msg pair cannot play the game, randomly gen. the pair
                    rnd_msg = torch.zeros(msg[i].shape).to(DEVICE)
                    for j in range(msg[i].shape[0]):
                        rnd_idx = np.random.randint(0,msg[i].shape[1])
                        rnd_msg[j, rnd_idx] = 1
                    new_msg.append(rnd_msg)
            msg = torch.stack(new_msg).transpose(0,1)
        # ============== End of msg changing part =================
        data_for_spk['data'] = train_batch
        data_for_spk['msg'] = msg

        return data_for_spk

def valid_cal(speaker, listener, valid_full, valid_candidates):
    '''
        Use valid data batch to see the accuracy for validation.
    '''
    with torch.no_grad():
        speaker.eval()
        listener.eval()
        msg, spk_log_prob, spk_entropy, _ = speaker(valid_full)
        pred_vector = listener(valid_candidates, msg)

        pred_idx = F.softmax(pred_vector).argmax(dim=1)
        val_acc, _ = cal_correct_preds(valid_full, valid_candidates, pred_idx)

        return val_acc/valid_full.shape[0]
    
# ============= Iterated method 1: just re-initialize listener =======
rewards = []
comp_ps = []
comp_ss = []
msg_types = []
valid_accs = []
comp_generations = []
comp_generations_before = []
comp_generations_after = []
max_comp = 0

for i in range(args.max_gen):       
    # ====================== Phase B ===================================
    listener = ListeningAgent().to(DEVICE)
    lis_optimizer = OPTIMISER(listener.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)

    rwd_avg20 = 0
    Ig_cnt = 0
    decay_explore_ratio = 1
    while(Ig_cnt<args.Ig):
        Ig_cnt += 1
        if Ig_cnt%4==1:
            batch_list, _ = batch_data_gen_valid(train_list, valid_list)
            train_batch, train_candidates, sel_idx_train = batch_list[0]['data'], batch_list[0]['candidates'], batch_list[0]['sel_idx']

        if Ig_cnt<=args.Ib:
            reward, spk_loss, lis_loss = train_phaseB(speaker, listener, spk_optimizer, lis_optimizer,
                                                  train_batch, train_candidates, sel_idx_train,
                                                  exp_ratio = decay_explore_ratio,
                                                  update='LISTENER')
        else:
            reward, spk_loss, lis_loss = train_phaseB(speaker, listener, spk_optimizer, lis_optimizer,
                                                  train_batch, train_candidates, sel_idx_train,
                                                  exp_ratio = decay_explore_ratio,
                                                  update='BOTH')

        rewards.append(reward)
        rwd_avg20 = (1-0.01)*rwd_avg20 + 0.01*reward

       

        if Ig_cnt%500==1:
            print('Gen.%d ==PhaseB==Round %d, rwd (%d, %d), spk_loss %.4f, lis_loss %.4f'%(i,Ig_cnt,reward, rwd_avg20, spk_loss, lis_loss))
            all_msgs = msg_generator(speaker, all_list, vocab_table_full, padding=True)
            msg_types.append(len(set(all_msgs.values())))
            comp_p, comp_s = compos_cal(all_msgs)
            comp_ps.append(comp_p)
            comp_ss.append(comp_s)
            if comp_p>=max_comp:
                max_comp = comp_p
                max_msg_all = all_msgs

    # ====================== Calculate Validation Score ===========================
    for val in range(200):
        _, batch_list_valid = batch_data_gen_valid(train_list, valid_list)
        valid_full, valid_candidates = batch_list_valid[0]['data'], batch_list_valid[0]['candidates']
        valid_acc = valid_cal(speaker, listener, valid_full, valid_candidates)
        valid_accs.append(valid_acc)
    # ====================== Record of language ===================================
    data_list = []
    #comp_list = []
    for c in range(200):
        batch_list = batch_data_gen()
        train_batch, train_candidates, sel_idx_train = batch_list[0]['data'], batch_list[0]['candidates'], batch_list[0]['sel_idx']
        data_for_spk = train_phaseC(speaker, listener, train_batch, train_candidates, sel_idx_train, rwd_filter = True)
        data_list.append(data_for_spk)
        #comp_list.append(compos_cal_inner(data_for_spk['msg'],data_for_spk['data'])[0])
    print('Gen.%d @@PhaseC@@, round %d'%(i,c))
    #comp_generations.append(comp_list)


    # ====================== Phase C ===================================
    shuf_pairs = pair_gen(data_list, phA_rnds = args.pairs_teach, sub_batch_size = 1)
    # ====================== Phase A ===================================
    speaker = SpeakingAgent().to(DEVICE)
    spk_optimizer = OPTIMISER(speaker.parameters(), lr=LEARNING_RATE)
    acc_avg20 = 0
    Ia_cnt = 0
    while (Ia_cnt<args.Ia):
        Ia_cnt += 1
        data_for_spk = random.choice(shuf_pairs)
        acc = train_phaseA(speaker, spk_optimizer, data_for_spk)
        acc_avg20 = (1-0.05)*acc_avg20 + 0.05*acc
    print('Gen.%d @@PhaseA@@, round is %d, acc is %.4f, acc_avg20 is %.4f'%(i,Ia_cnt, acc,acc_avg20))
    print(comp_ps[-1])



if not os.path.exists('exp_results'):
    os.mkdir('exp_results')

save_path = 'exp_results/' + args.path + '/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
np.save(save_path+'comp_ps.npy', comp_ps)
np.save(save_path+'rewards.npy',rewards)
np.save(save_path+'msg_types.npy',np.asarray(msg_types))
#np.save(save_path+'comp_generations', np.asarray(comp_generations))
np.save(save_path+'valid_accs', np.asarray(valid_accs))
msg_print_to_file(max_msg_all, save_path)
