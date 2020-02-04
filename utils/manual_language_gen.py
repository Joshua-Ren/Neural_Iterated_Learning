#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:01:32 2019

@author: xiayezi
"""
import sys 
sys.path.append("..") 
import numpy as np
from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
import matplotlib.pyplot as plt

vocab_table_full = [chr(97+int(v)) for v in range(26)]

char_mapping = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                '1','2','3','4','5','6','7','8','9','0','~','!','@','#','$','%','^','&','*','(',')','_','+','<','>','?']
#random.shuffle(char_mapping)

def value_to_onehot(value, char_mapping):
    '''
        Map value to one-hot tensor. Shape is [ATTRI_SIZE, MSG_VOCSIZE]
    '''
    msg_onehot = torch.zeros((ATTRI_SIZE, MSG_VOCSIZE))
    tmp_idx = 0
    for i in range(len(value)):
        tmp_idx = char_mapping.index(value[i])
        msg_onehot[i,tmp_idx] = 1
    
    return msg_onehot

def key_to_value(key, char_mapping,comp = True):
    '''
        Generate value based on key. Now only for NUM_SYSTEM=10, ATTRI_SIZE=2
    '''
    key[0]
    tmp = ''.join([s for s in key])
    int_key = int(tmp)
    dig_0 = int(key[0])
    dig_1 = int(key[1])
    #dig_2 = np.mod(int(int_key/NUM_SYSTEM**2), NUM_SYSTEM)
    value = []
    if comp == True:
        #value.append(char_mapping[dig_2])
        value.append(char_mapping[dig_1])
        value.append(char_mapping[dig_0])
    else:
        #value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        
    return ''.join(value)



# ========== Degenerate language ===================
deg_all = {}
deg_train = {}
deg_valid = {}

deg_spk_train = {}  # Data for spk training, 'data' should be dicimal, 'msg' one hot
data_list = []
msg_list = []
for i in range(NUM_SYSTEM**ATTRI_SIZE):
    # ===== For dictionary version
    key = num_to_tup(i)
    value = 'aa'
    deg_all[key] = value
    if i in valid_list:
        deg_valid[key] = value
    elif i in all_list:
        deg_train[key] = value
    # ==== For spk training version
    msg_list.append(value_to_onehot(value, char_mapping))
    data_list.append(i)
    
deg_spk_train['data'] = np.asarray(data_list)
deg_spk_train['msg'] = torch.stack(msg_list).transpose(0,1)
    
    
#compos_cal(deg_all)    # Should be approximate 0
        
# ========== Compositional language ===================
comp_all = {}

comp_spk_train = {}  # Data for spk training, 'data' should be dicimal, 'msg' one hot
data_list = []
msg_list = []
for i in range(NUM_SYSTEM**ATTRI_SIZE):
    # ===== For dictionary version
    key = num_to_tup(i)
    value = key_to_value(key, char_mapping, True)
    comp_all[key] = value
    # ==== For spk training version
    msg_list.append(value_to_onehot(value, char_mapping))
    data_list.append(i)

comp_spk_train['data'] = np.asarray(data_list)
comp_spk_train['msg'] = torch.stack(msg_list).transpose(0,1)
print('Comp comp is: '+ str(compos_cal(comp_all)))
#compos_cal(comp_all)   # Should approximate 1.

# ========== Holistic language ===================
holi_spk_train = {}
new_idx = torch.randperm(64)
holi_spk_train['data'] = comp_spk_train['data']
holi_spk_train['msg'] = comp_spk_train['msg'][:,new_idx,:]

comp, _, _ = compos_cal_inner(holi_spk_train['msg'],holi_spk_train['data'])
print('Holi comp is: '+ str(comp))

# ========== Holistic language2 ===================
PERM2 = 20#50
holi_spk_train2 = {}
new_idx2 = comp_spk_train['data']
perm = torch.randperm(PERM2)
new_idx2 = torch.cat((perm, torch.tensor(new_idx2[PERM2:])),0)

holi_spk_train2['data'] = comp_spk_train['data']
holi_spk_train2['msg'] = comp_spk_train['msg'][:,new_idx2,:]

comp, _, _ = compos_cal_inner(holi_spk_train2['msg'],holi_spk_train2['data'])
print('Holi2 comp is: '+ str(comp))

# ========== Holistic language3 ===================
PERM3 = 10#35
holi_spk_train3 = {}
new_idx3 = comp_spk_train['data']
perm = torch.randperm(PERM3)
new_idx3 = torch.cat((perm, torch.tensor(new_idx3[PERM3:])),0)

holi_spk_train3['data'] = comp_spk_train['data']
holi_spk_train3['msg'] = comp_spk_train['msg'][:,new_idx3,:]

comp, _, _ = compos_cal_inner(holi_spk_train3['msg'],holi_spk_train3['data'])
print('Holi3 comp is: '+ str(comp))

'''
# ========== Read language from txt ===================
path = 'exp_results/test_both_spk_and_lis/msg_all.txt'
read_spk_train = {}
cnt = 0
all_messages = []
msg_list = []
with open(path,'r') as f:
    for lines in f:
        for i in range(8):
            cnt += 1
            if cnt > 8:
                #all_messages.append(lines.split()[i+1])
                msg_list.append(value_to_onehot(lines.split()[i+1], char_mapping))
                
read_spk_train['data'] = comp_spk_train['data']
read_spk_train['msg'] = torch.stack(msg_list).transpose(0,1)
comp, _, _ = compos_cal_inner(read_spk_train['msg'],read_spk_train['data'])
print('Txt comp is: '+ str(comp))
'''


# =================== Manual Language For the listener ========================

def get_lis_curve_msg(lis_curve_batch_ls, language_train):
    '''
        Input is lis_curve_batch [N_B,1]. language should use the *_train version
        Output has the same structure with *_train
        The function only add lis_train['msg'] part
    '''
    lis_train = lis_curve_batch_ls[0]
    tmp_data = lis_train['data']
    msg_table = language_train['msg'].transpose(0,1)
    msg_list = []
    for i in range(tmp_data.shape[0]):   
        tmp_msg = msg_table[tmp_data[i]]
        msg_list.append(tmp_msg)
    lis_train['msg'] = torch.stack(msg_list).transpose(0,1)
    return lis_train 



#comp_p,_, all_msg = compos_cal_inner(comp_spk_train['msg'],comp_spk_train['data'])





'''
test_msg = {}
for i in range(100):
    tmp = []
    key = num_to_tup(i,2)
    dig_0 = np.mod(i, 10)
    dig_1 = np.mod(int(i*0.1),10)
    tmp = [char_mapping[dig_0], char_mapping[dig_1]]
    value = ''.join(tmp)
    test_msg[key] = value
    
compos_cal(test_msg)

simple_msg = {}
simple_msg['0','0'] = 'aa'
simple_msg['0','1'] = 'ab'
simple_msg['1','0'] = 'ba'
simple_msg['1','1'] = 'bb'
compos_cal(simple_msg)

msg = {}
msg['green','box'] = 'aa'     
msg['blue','box'] = 'ba'
msg['green','circle'] = 'ab'      
msg['blue','circle'] = 'bb'
compos_cal(msg) 
'''
