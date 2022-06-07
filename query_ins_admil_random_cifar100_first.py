
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)



if __name__=="__main__":
    run = 1
    X_train_pos, X_train_neg = np.load("Dataset/Cifar100/X_train_pos.npy"), np.load("Dataset/Cifar100/X_train_neg.npy")
    y_train_pos, y_train_neg = np.load("Dataset/Cifar100/y_train_pos.npy"), np.load("Dataset/Cifar100/y_train_neg.npy")
    gammas = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])*1e-06
    random.seed(run)
    input_dim = X_train_pos.shape[2]
    no_query = 150
    
    indices = list(range(len(y_train_pos.flatten())))
    random.shuffle(indices)
    indices = indices[:no_query]
    
    queried_pos_data = []
    queried_neg_data = []
    added_ins = 0
    for i in range(len(X_train_pos)):
        lower = i*32
        upper = (i+1)*32
        for index in indices:
            
            if added_ins>=no_query:
               break
            if index>=lower and index<upper:
                instance_no = index-lower
                feat = X_train_pos[i][instance_no]
                label = y_train_pos[i][instance_no]
               
                bag_feat = feat.reshape(1,input_dim)
                bag_label = np.array([label]).reshape(1, 1)
                if label==1:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_"+str(instance_no)]
                    queried_pos_data.append(np.array(temp))
                    added_ins+=1
                elif label==0:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_"+str(instance_no)]
                    queried_neg_data.append(np.array(temp))
                    added_ins+=1
                else:
                    print("Something abnormal is going on here")

    queried_pos_data = np.array(queried_pos_data)
    queried_neg_data = np.array(queried_neg_data)
    np.save("Dataset/Cifar100/added_pos_data_"+str(run)+"_"+str(no_query)+"_admil_random.npy",queried_pos_data)
    np.save("Dataset/Cifar100/added_neg_data_"+str(run)+"_"+str(no_query)+"_admil_random.npy", queried_neg_data)
   
    
    
