
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import os
import sys
from sklearn.metrics import average_precision_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)

class CustomLossMinMax(torch.nn.Module):
    def __init__(self):
        super(CustomLossMinMax, self).__init__()
    def forward(self, pos_outputs, neg_outputs, p):
        
        weighted_pos_outputs = torch.sum(pos_outputs*p.detach(), axis = 1)
        [neg_max_value, _] = torch.max(neg_outputs, axis=1)
        hinge_loss = torch.zeros_like(pos_outputs)[0][0]
        
        for neg in neg_max_value:
            dro_loss = 1-weighted_pos_outputs+neg
            dro_loss[dro_loss<0]=0
            dro_loss = torch.sum(dro_loss)
            hinge_loss += dro_loss
        
        
        return hinge_loss/(neg_outputs.shape[0]*len(pos_outputs))


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
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def evaluate(model, X_pos, X_neg, y_pos, y_neg, input_dim = 4096):
     pos_features = torch.from_numpy(X_pos)
     neg_features = torch.from_numpy(X_neg)
     pos_features = pos_features.reshape(-1, input_dim)
     neg_features = neg_features.reshape(-1, input_dim)
     pos_feat = Variable(pos_features).to(device)
     neg_feat = Variable(neg_features).to(device)
     pos_output = model(pos_feat.float())
     neg_output = model(neg_feat.float())
     pos_output = pos_output.data.cpu().numpy().flatten()
     neg_output = neg_output.data.cpu().numpy().flatten()
     preds = np.concatenate([pos_output, neg_output])
     gts = np.concatenate([y_pos.flatten(), y_neg.flatten()])
     auc = roc_auc_score(gts, preds)
     ap = average_precision_score(gts, preds)
     return [ap,auc]
    
    
if __name__=="__main__":
    [_, run, gamma] = sys.argv
    run = run
    lr = 0.01
    gamma = float(gamma)*1e-14
    
    X_train_pos, X_train_neg = np.load("Dataset/Cifar10/X_train_pos.npy", allow_pickle = True), np.load("Dataset/Cifar10/X_train_neg.npy", allow_pickle = True)
    y_train_pos, y_train_neg = np.load("Dataset/Cifar10/y_train_pos.npy", allow_pickle = True), np.load("Dataset/Cifar10/y_train_neg.npy", allow_pickle = True)
    X_test_pos, X_test_neg = np.load("Dataset/Cifar10/X_test_pos.npy", allow_pickle = True), np.load("Dataset/Cifar10/X_test_neg.npy", allow_pickle = True)
    y_test_pos, y_test_neg = np.load("Dataset/Cifar10/y_test_pos.npy", allow_pickle = True), np.load("Dataset/Cifar10/y_test_neg.npy", allow_pickle = True)
    batch_size = 120
    input_dim = X_train_pos.shape[2]
    no_segments = X_train_pos.shape[1]
    max_iterations = 20000

    pos_idx = list(range(len(X_train_pos)))
    neg_idx = list(range(len(X_train_neg)))
    model = network(input_dim = input_dim)
    customobjective = CustomLossMinMax()
    model.to(device)
    customobjective.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 0.001)
    
    losses = []
    model.train()
    p_values = []
    test_aucs = []
    test_maps = []
    best_test_map = 0
    for i in range(max_iterations):
        
        model.train()
        
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        train_pos_feat = X_train_pos[pos_idx[:int(batch_size/2)]]
        train_pos_feat = train_pos_feat.reshape(-1, input_dim)

        train_neg_feat = X_train_neg[neg_idx[:int(batch_size/2)]]
        train_neg_feat = train_neg_feat.reshape(-1,  input_dim)

        train_feat = np.concatenate([train_pos_feat, train_neg_feat])
        train_feat = torch.from_numpy(train_feat).float()
        
        train_feat = Variable(train_feat, requires_grad = True).to(device)
        optimizer.zero_grad()
        outputs = model(train_feat.float())
        outputs = outputs.reshape(int(outputs.shape[0]/no_segments), no_segments)
        pos_outputs, neg_outputs = outputs[:int(batch_size/2)], outputs[int(batch_size/2):]
        max_pos_outputs, _ = torch.max(pos_outputs, axis = 1)
        subtracted_pos_outputs = pos_outputs-max_pos_outputs[:, None]
        p = torch.exp(subtracted_pos_outputs/gamma)/(torch.sum(torch.exp(subtracted_pos_outputs/gamma), axis = 1))[:, None]
        loss = customobjective(pos_outputs, neg_outputs, p=p)
        loss.backward()
        optimizer.step()
        a = loss.data.cpu()
        losses.append(a)
        
        
        if i%10==0:
            model.eval()
            [test_map, test_auc] = evaluate(model, X_test_pos, X_test_neg, y_test_pos, y_test_neg)
            test_aucs.append(test_auc)
            test_maps.append(test_map)
            
            if test_map>best_test_map:
                best_test_map = test_map
                
                torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("dro_passive/Cifar10/models/model_"+str(run)+"_"+str(gamma)+"_best.pth.tar"))
              
            
           
            
    
    losses = np.array(losses)
    test_aucs = np.array(test_aucs)
    test_maps = np.array(test_maps)
    np.save("dro_passive/Cifar10/logs/aucs_"+str(run)+"_"+str(gamma)+".npy", test_aucs)
    np.save("dro_passive/Cifar10/logs/losses_"+str(run)+"_"+str(gamma)+".npy", losses)
    np.save("dro_passive/Cifar10/logs/maps_"+str(run)+"_"+str(gamma)+".npy", test_maps)
    
        
        
    
    
    

