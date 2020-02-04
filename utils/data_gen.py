#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:03:06 2019

@author: xiayezi
"""
import sys
sys.path.append("..")
import numpy as np
from utils.conf import *





def valid_list_gen(low, high, num):
    '''
        Randomly generate distinct numbers, range in (low, high), with size.
    '''
    s = []
    while(len(s)<num):
        x = np.random.randint(low, high)
        if x not in s:
            s.append(x)    
    return s


def gen_distinct_candidates(tgt_list, sel_list, candi_size = SEL_CANDID):
    '''
        tgt_list may contain part of elements in sel_list
        output the (data_candidates, sel_idx)
    '''
    batch_size = len(tgt_list)
    data_candidates = np.zeros((batch_size, candi_size))
    sel_idx = []
    for i in range(batch_size):
        tmp_idx = np.random.randint(0, candi_size)
        sel_idx.append(tmp_idx)
        for j in range(candi_size):
            if j == 0:
                data_candidates[i,j]=tgt_list[i]
                continue
            rand_candi = random.choice(sel_list)
            while (rand_candi in data_candidates[i,:]):
                rand_candi = random.choice(sel_list)
            data_candidates[i, j] = rand_candi
        data_candidates[i, 0] = data_candidates[i, tmp_idx]
        data_candidates[i, tmp_idx] = tgt_list[i]
    
    return data_candidates, np.asarray(sel_idx)


def gen_candidates(low, high, valid_list, batch = BATCH_SIZE, candi = SEL_CANDID, train=True):
    if train == True:
        s = []
        num = batch*candi
        while (len(s)<num):
            x = np.random.randint(low, high)
            while (x in valid_list):
                x = np.random.randint(low, high)
            s.append(x)
        return np.asarray(s).reshape((batch, candi))
    elif train == False:
        s = []
        valid_num = len(valid_list)
        while (len(s)<valid_num*candi):
            x = np.random.randint(0,valid_num)
            s.append(valid_list[x])
        return np.asarray(s).reshape((valid_num, candi))



def valid_data_gen():
    sel_idx_val = np.random.randint(0,SEL_CANDID, (len(valid_list),))
    valid_candidates = gen_candidates(0, NUM_SYSTEM**ATTRI_SIZE, valid_list, train=False)
    valid_full = np.zeros((valid_num,))
    
    for i in range(valid_num):
        valid_full[i] = valid_candidates[i, sel_idx_val[i]]    
    
    return valid_full, valid_candidates, sel_idx_val


def batch_data_gen():
    num_batches = int(len(all_list)/BATCH_SIZE) # Here we assume batch size=x*100 first
    random.shuffle(all_list)
    
    batch_list = []
    
    for i in range(num_batches):
        one_batch = {}
        tmp_list = all_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        train_candidates, sel_idx_train = gen_distinct_candidates(tmp_list, all_list)

        for i in range(BATCH_SIZE):
            train_candidates[i,sel_idx_train[i]] = tmp_list[i]
            
        one_batch['sel_idx'] = sel_idx_train
        one_batch['candidates'] = train_candidates
        one_batch['data'] = np.asarray(tmp_list)
        batch_list.append(one_batch)
    return batch_list




def batch_data_gen_valid(train_list, valid_list):
    '''
        Only one batch in batch_list
    '''   
    
    train_batch_list = []
    valid_batch_list = []
    train_batch = {}
    valid_batch = {}
    random.shuffle(train_list)
    random.shuffle(valid_list)
    train_candidates, sel_idx_train = gen_distinct_candidates(train_list, train_list)
    valid_candidates, sel_idx_valid = gen_distinct_candidates(valid_list, all_list)

    for i in range(len(train_list)):
        train_candidates[i,sel_idx_train[i]] = train_list[i]
    for j in range(len(valid_list)):
        valid_candidates[j,sel_idx_valid[j]] = valid_list[j]
            
    train_batch['sel_idx'] = sel_idx_train
    train_batch['candidates'] = train_candidates
    train_batch['data'] = np.asarray(train_list)
    
    valid_batch['sel_idx'] = sel_idx_valid
    valid_batch['candidates'] = valid_candidates
    valid_batch['data'] = np.asarray(valid_list)
    
    train_batch_list.append(train_batch)
    valid_batch_list.append(valid_batch)
    return train_batch_list, valid_batch_list
'''
tl,vl = batch_data_gen_valid(train_list, valid_list)
for i in range(56):
    for j in range(15):
        if tl[0]['candidates'][i,j] in valid_list:
            print('@@@@')
'''    




def shuffle_batch(batch_list):
    '''
        Shuffle the order of data in the same batch.
    '''
    shuf_batch_list = []
    for j in range(len(batch_list)):
        tmp_batch = {}
        train_batch, train_candidates, sel_idx_train = batch_list[j]['data'], batch_list[j]['candidates'], batch_list[j]['sel_idx']
        train_batch
        tmp = np.concatenate((train_batch.reshape((-1,1)),
                              train_candidates,
                              sel_idx_train.reshape((-1,1))),axis=1)
        np.random.shuffle(tmp)
        tmp_batch['data'] = tmp[:,0]
        tmp_batch['candidates'] = tmp[:,1:-1]
        tmp_batch['sel_idx'] = tmp[:,-1]
        shuf_batch_list.append(tmp_batch)
    return shuf_batch_list
        

def pair_gen(data_list, phA_rnds = 100, degnerate='none', sub_batch_size = 1):
    '''
        Given the list of x-y pairs generated by speaker(t), we shuffle the mappings
        and yield a pair set of number
        degnerate could be 'none', 'mix', 'full'
    '''
    all_data = []
    all_msgs = []
    cnt_samples = 0
    for i in range(len(data_list)):
        for j in range(data_list[i]['data'].shape[0]):
            cnt_samples += 1
            all_data.append(data_list[i]['data'][j])
            if degnerate=='full':
                all_msgs.append(data_list[0]['msg'].transpose(0,1)[0])
            elif degnerate=='mix':
                all_msgs.append(data_list[i]['msg'].transpose(0,1)[j])
                all_data.append(data_list[i]['data'][j])
                all_msgs.append(data_list[0]['msg'].transpose(0,1)[0])
            else:
                all_msgs.append(data_list[i]['msg'].transpose(0,1)[j])
            
    phA_data_list = []
    for i in range(phA_rnds):
        phA_data_for_spk = {}
        phA_data = []
        phA_msgs = []
        for j in range(sub_batch_size):
            ridx = np.random.randint(0, cnt_samples)
            phA_data.append(all_data[ridx])
            phA_msgs.append(all_msgs[ridx])
        phA_data_for_spk['data'] = np.asarray(phA_data)
        phA_data_for_spk['msg'] = torch.stack(phA_msgs).transpose(0,1)    
        phA_data_list.append(phA_data_for_spk)  
    
    return phA_data_list     

valid_num = VALID_NUM
train_num = NUM_SYSTEM**ATTRI_SIZE - VALID_NUM
all_list = [i for i in range(NUM_SYSTEM**ATTRI_SIZE)]
valid_list = valid_list_gen(0, NUM_SYSTEM**ATTRI_SIZE, valid_num)
train_list = list(set([i for i in range(NUM_SYSTEM**ATTRI_SIZE)]) ^ set(valid_list))

'''
batch_list = batch_data_gen()
shuf_batch_list = shuffle_batch(batch_list)
batch_list = batch_data_gen()
'''

