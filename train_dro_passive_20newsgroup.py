
import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)




class DROLoss(torch.nn.Module):
    def __init__(self, gamma):
        super(DROLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, abnormal_outputs, normal_outputs, p_values):

        weighted_abnormal_outputs = Variable(torch.zeros(len(abnormal_outputs))).to(device)
        max_normal_outputs = Variable(torch.zeros(len(normal_outputs))).to(device)
        
        for i, abnormal_output in enumerate(abnormal_outputs):
            p_value = p_values[i]
            weighted_abnormal_outputs[i] = torch.sum(p_value*abnormal_output)
        
        for i, normal_output in enumerate(normal_outputs):
           
            max_normal_outputs[i] = torch.max(normal_output)
         
        
        hinge_loss = torch.zeros_like(abnormal_outputs[0][0]).to(device)
        for normal in max_normal_outputs:
            dro_loss = 1-weighted_abnormal_outputs+normal
            dro_loss[dro_loss<0] = 0
            dro_loss = torch.sum(dro_loss)
            hinge_loss +=dro_loss
        return hinge_loss/(len(max_normal_outputs)*len(abnormal_outputs))
    


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
    
    
def get_mapping(mapping, X):
    for i in range(len(X)):
        mapping[X[i][2]] = X[i][0].shape[0]
    return mapping


def evaluate(model, data_pos, data_neg):
    pos_feats, _, pos_instance_labels = data_pos[:, 0],  data_pos[:, 2], data_pos[:, 1]
    
    neg_feats, _, neg_instance_labels = data_neg[:, 0],  data_neg[:, 2], data_neg[:, 1]
    
    squeezed_pos_feats = []
    for x in pos_feats:
        squeezed_pos_feats.extend(x)
    squeezed_pos_feats = np.array(squeezed_pos_feats)
    
    squeezed_neg_feats = []
    for x in neg_feats:
        squeezed_neg_feats.extend(x)
    squeezed_neg_feats = np.array(squeezed_neg_feats)
    
    train_pos_feat = torch.from_numpy(squeezed_pos_feats)
        
    train_pos_feat = Variable(train_pos_feat, requires_grad = False).to(device)
    pos_outputs = model(train_pos_feat.float())
    
    pos_outputs = pos_outputs.data.cpu().numpy().flatten()
    
    train_neg_feat = torch.from_numpy(squeezed_neg_feats)
        
    train_neg_feat = Variable(train_neg_feat, requires_grad = False).to(device)
    neg_outputs = model(train_neg_feat.float())
    neg_outputs = neg_outputs.data.cpu().numpy().flatten()
    
    preds = np.concatenate([pos_outputs, neg_outputs])
    gts = []
    for label in pos_instance_labels:
        gts.extend(list(label.flatten()))
    for label in neg_instance_labels:
        gts.extend(list(label.flatten()))
    gts = np.array(gts)
    preds = preds.flatten()
    auc = roc_auc_score(gts, preds)
    prec = average_precision_score(gts, preds)
    return [auc, prec]
    
       

if __name__=="__main__":
    
    [_, run, topic_no, gamma] = sys.argv
    topic_mapping = np.load("topic_map.npy", allow_pickle = True).item()
    
    topic_no = int(topic_no)
    run = int(run)
    gamma = float(gamma)*1e-06
    topic = topic_mapping[topic_no]
    lr = 0.1
    
    train_pos = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_train_pos.npy", allow_pickle = True)
    train_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_train_neg.npy", allow_pickle = True)
    test_pos = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_test_pos.npy", allow_pickle = True)
    test_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_test_neg.npy", allow_pickle = True)
        
   
    bag_instance_mapping = {}
    bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)
    
    bag_instance_mapping = get_mapping(bag_instance_mapping, test_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, test_neg)
    
    max_iterations = 20000
    batch_size = 30
    
    model = network(input_dim = 200)
    customobjective = DROLoss(gamma = gamma)
    model.to(device)
    customobjective.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 0.001)
    
    losses = []
    p_values = []
    test_maps = []
    test_aucs = []
    best_test_map = 0
    
    pos_idx = list(range(len(train_pos)))
    neg_idx = list(range(len(train_neg)))
        
    for i in range(max_iterations):
        
        model.train()
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        
        pos_data_batch = train_pos[pos_idx[:int(batch_size/2)]]
        train_pos_feat, batch_pos_bag_names = pos_data_batch[:, 0],  pos_data_batch[:, 2]
      
        
        squeezed_pos_feats = []
        for x in train_pos_feat:
            squeezed_pos_feats.extend(x)
        squeezed_pos_feats = np.array(squeezed_pos_feats)
        
        
        neg_data_batch = train_neg[neg_idx[:int(batch_size/2)]]
        
        train_neg_feat, batch_neg_bag_names = neg_data_batch[:, 0], neg_data_batch[:, 2]
        
        
        squeezed_neg_feats = []
        for x in train_neg_feat:
            squeezed_neg_feats.extend(x)
        squeezed_neg_feats = np.array(squeezed_neg_feats)
        
        
        train_feat = np.concatenate([squeezed_pos_feats, squeezed_neg_feats])
        train_feat = np.array(train_feat, dtype = np.float)
        train_feat = torch.from_numpy(train_feat)
        
        train_feat = Variable(train_feat, requires_grad = True).to(device)
        optimizer.zero_grad()
        outputs = model(train_feat.float())
        
        outputs_positive, outputs_negative = [], []
        
        count = 0
        p = []
        for bag_name in batch_pos_bag_names:
            no_instances = bag_instance_mapping[bag_name]
            pred = outputs[count: count+no_instances]
            outputs_positive.append(pred)
            sub_pred = pred-torch.max(pred)
            p.append(torch.exp(sub_pred/gamma)/(torch.sum(torch.exp(sub_pred/gamma))))
            count+= no_instances
        
        for bag_name in batch_neg_bag_names:
            no_instances = bag_instance_mapping[bag_name]
            
            outputs_negative.append(outputs[count: count+no_instances])
            
            count+=no_instances
       
        loss = customobjective(outputs_positive, outputs_negative, p_values=p)
        loss.backward()
        optimizer.step()
        a = loss.data.cpu()
        losses.append(a)
      
       
        
        if i%10==0:
            print("Best testing MAP", best_test_map)
            model.eval()
            [test_auc, test_ap] = evaluate(model, test_pos, test_neg)
            test_aucs.append(test_auc)
            test_maps.append(test_ap)
            if test_ap>best_test_map:
                best_test_map = test_ap
                torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("dro_passive/20NewsGroup/models/model_"+str(run)+"_"+str(gamma)+"_"+str(topic)+"_best_map.pth.tar"))
                
              
                
            
    losses = np.array(losses)
    test_maps = np.array(test_maps)
    test_aucs = np.array(test_aucs)
    np.save("dro_passive/20NewsGroup/logs/test_aucs_"+str(run)+"_"+str(gamma)+"_"+str(topic)+".npy", test_aucs)
    np.save("dro_passive/20NewsGroup/logs/test_maps_"+str(run)+"_"+str(gamma)+"_"+str(topic)+".npy", test_maps)
    np.save("dro_passive/20NewsGroup/logs/losses_"+str(run)+"_"+str(gamma)+"_"+str(topic)+".npy", losses)
    
    
    torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("dro_passive/20NewsGroup/models/model_"+str(run)+"_"+str(gamma)+"_"+str(topic)+"_last_map.pth.tar"))
            
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
