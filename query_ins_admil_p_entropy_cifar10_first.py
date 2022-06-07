
#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)


class network(torch.nn.Module):
    def __init__(self, input_dim):
        super(network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.6)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        feat = self.fc2(x)
        x = self.dropout(feat)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return [x, feat]



if __name__=="__main__":
    run = 1
    X_train_pos, X_train_neg = np.load("Dataset/Cifar10/X_train_pos.npy"), np.load("Dataset/Cifar10/X_train_neg.npy")
    y_train_pos, y_train_neg = np.load("Dataset/Cifar10/y_train_pos.npy"), np.load("Dataset/Cifar10/y_train_neg.npy")
    gammas = np.array([1, 100, 10000,  1000000,  100000000])*1e-06
    input_dim = X_train_pos.shape[2]
    no_query = 150

    for gamma in gammas:
        model_path = "dro_passive/Cifar10/models/model_"+str(run)+"_"+str(gamma)+"_best.pth.tar"
        checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
        print("Loading Model from:", model_path)
        model = network(input_dim = input_dim)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval() 
        pos_features = torch.from_numpy(X_train_pos)
        pos_features = pos_features.reshape(-1, input_dim)
        pos_feat = Variable(pos_features).to(device)
        [pos_output, _] = model(pos_feat.float())
        pos_output = pos_output.data.cpu().numpy()
        pos_output = pos_output.flatten()
        closeness = np.abs(pos_output-0.5)
        indices = np.argsort(closeness)[:no_query]
        queried_pos_data = []
        queried_neg_data = []
        out_th = 0.2
        for i in range(len(X_train_pos)):
            lower = i*32
            upper = (i+1)*32
            for index in indices:
                if closeness[index]>out_th:
                    continue
                if index>=lower and index<upper:
                    instance_no = index-lower
                    feat = X_train_pos[i][instance_no]
                    label = y_train_pos[i][instance_no]
                    prediction = pos_output[index]
                    bag_feat = feat.reshape(1,input_dim)
                    bag_label = np.array([label]).reshape(1, 1)
                    if label==1:
                        temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_"+str(instance_no)]
                        queried_pos_data.append(np.array(temp))
                    elif label==0:
                        temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_"+str(instance_no)]
                        queried_neg_data.append(np.array(temp))
                    else:
                        print("Something abnormal is going on here")
                        
        queried_pos_data = np.array(queried_pos_data)
        queried_neg_data = np.array(queried_neg_data)
        np.save("Dataset/Cifar10/added_pos_data_"+str(run)+"_"+str(no_query)+"_"+str(gamma)+"_admil_f_entropy.npy",queried_pos_data)    
        np.save("Dataset/Cifar10/added_neg_data_"+str(run)+"_"+str(no_query)+"_"+str(gamma)+"_admil_f_entropy.npy", queried_neg_data)
                    
                    
                    
                    
                    
            
            
