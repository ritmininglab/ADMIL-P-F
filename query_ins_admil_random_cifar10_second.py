#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:32:56 2021

@author: hxs1943
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)


def filter_ins(cluster_pts, queried_instances):
    idxs = []
    for pt in cluster_pts:
        if pt in queried_instances:
            continue
        idxs.append(pt)
    idxs = np.array(idxs)
    return idxs


def getinstances(pos, neg):
    bag_infos = {}
    for data in pos:
        bag_name = data[2]
        if "added" in bag_name:
            [_, _, _, bag_no, _, _, instance_no] = bag_name.split("_")
            if int(bag_no) not in bag_infos:
                bag_infos[int(bag_no)] = []
            bag_infos[int(bag_no)].append(int(instance_no))
    for data in neg:
        bag_name = data[2]
        if "added" in bag_name:
            [_, _, _, bag_no, _, _, instance_no] = bag_name.split("_")
            if int(bag_no) not in bag_infos:
                bag_infos[int(bag_no)] = []
            bag_infos[int(bag_no)].append(int(instance_no))
    return bag_infos


if __name__=="__main__":
    run = 1
  
    X_train_pos, X_train_neg = np.load("Dataset/Cifar10/X_train_pos.npy"), np.load("Dataset/Cifar10/X_train_neg.npy")
    y_train_pos, y_train_neg = np.load("Dataset/Cifar10/y_train_pos.npy"), np.load("Dataset/Cifar10/y_train_neg.npy")
    input_dim = X_train_pos.shape[2]
    
    random.seed(run)
    curr_query = 300
    prev_query = 150
    best_test_map = 0
    prev_train_pos = np.load("Dataset/Cifar10/added_pos_data_"+str(run)+"_"+str(prev_query)+"_random.npy", allow_pickle = True)
    prev_train_neg = np.load("Dataset/Cifar10/added_neg_data_"+str(run)+"_"+str(prev_query)+"_random.npy", allow_pickle = True)

    queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)
    indices = list(range(len(y_train_pos.flatten())))
    random.shuffle(indices)
          
    queried_pos_data = []
    queried_neg_data = []
    total_queried = 0
    for index in indices:
        if total_queried>=curr_query-prev_query:
            break
                
        for i in range(len(X_train_pos)):
            lower = i*32
            upper = (i+1)*32
            if index>=lower and index<upper:
                instance_no = index-lower
                if i in queried_instance_ids:
                    if instance_no in queried_instance_ids[i]:
                        continue
                feat = X_train_pos[i][instance_no]
                label = y_train_pos[i][instance_no]
                bag_feat = feat.reshape(1, input_dim)
                bag_label = np.array([label]).reshape(1, 1)
                if label==1:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_"+str(instance_no)]
                    queried_pos_data.append(np.array(temp))
                    total_queried+=1
                elif label==0:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_"+str(instance_no)]
                    queried_neg_data.append(np.array(temp))
                    total_queried+=1
                else:
                    print("Something abnormal is going on here")
                
        
    queried_pos_data = np.array(queried_pos_data)
    queried_neg_data = np.array(queried_neg_data)
    print("Added: For query", curr_query-prev_query, "fetched positive instance", len(queried_pos_data), "negative instance", len(queried_neg_data))
        
    if len(queried_pos_data)>0 and len(prev_train_pos)>0:
        queried_pos_data = np.concatenate([queried_pos_data, prev_train_pos])
    elif len(queried_pos_data)==0 and len(prev_train_pos)>0:
        queried_pos_data = prev_train_pos
    elif len(queried_pos_data)>0 and len(prev_train_pos)==0:
        queried_pos_data = queried_pos_data
    else:
        queried_pos_data = []
        
    if len(queried_neg_data)>0 and len(prev_train_neg)>0:
        queried_neg_data = np.concatenate([queried_neg_data, prev_train_neg])
    elif len(queried_neg_data)==0 and len(prev_train_neg)>0:
        queried_neg_data = prev_train_neg
    elif len(queried_neg_data)>0 and len(prev_train_neg)==0:
        queried_neg_data = queried_neg_data
    else:
        ried_neg_data = []
        
        
        
            
        
    np.save("Dataset/Cifar10/added_pos_data_"+str(run)+"_"+str(curr_query)+"_admil_random.npy",queried_pos_data)
    np.save("Dataset/Cifar10/added_neg_data_"+str(run)+"_"+str(curr_query)+"_admil_random.npy", queried_neg_data)
            
        
        
        
        
        
