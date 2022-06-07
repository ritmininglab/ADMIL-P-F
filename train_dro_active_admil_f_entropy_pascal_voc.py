
import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)

bce_loss = nn.BCELoss()



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

def process_labeled_ins(pos_data, neg_data):
    all_feats = []
    all_labels = []
    all_bag_names = []
    for data in pos_data:
        feats, labels, bag_name = data[0], data[1], data[2]
        all_bag_names.append(bag_name)
        all_feats.extend(feats)
        all_labels.extend(labels)
    for data in neg_data:
        feats, labels, bag_name = data[0], data[1], data[2]
        all_bag_names.append(bag_name)
        all_feats.extend(feats)
        all_labels.extend(labels)

    all_feats = np.array(all_feats)
    all_labels = np.array(all_labels)
    all_bag_names = np.array(all_bag_names)
    return [all_feats, all_labels, all_bag_names]

if __name__=="__main__":
    [_, run, gamma, reg_coeff] = sys.argv
    no_query = 30
    patience = 0
    run = int(run)
    reg_coeff = float(reg_coeff)*1e-04
    gamma = float(gamma)*1e-06
    no_query = int(no_query)
    lr = 0.01
    mil_train_pos = np.load("Dataset/Pascal_VOC/train_pos.npy", allow_pickle = True)
    mil_train_neg = np.load("Dataset/Pascal_VOC/train_neg.npy", allow_pickle = True)

    lab_train_pos = np.load("Dataset/Pascal_VOC/added_pos_data_"+str(run)+"_"+str(no_query)+"_admil_f_entropy.npy", allow_pickle = True)
    lab_train_neg = np.load("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(no_query)+"_admil_f_entropy.npy", allow_pickle = True)
    

    lab_feats, lab_labels, _ = process_labeled_ins(lab_train_pos, lab_train_neg)

    test_pos = np.load("Dataset/Pascal_VOC/test_pos.npy", allow_pickle = True)
    test_neg = np.load("Dataset/Pascal_VOC/test_neg.npy", allow_pickle = True)

    bag_instance_mapping = {}
    bag_instance_mapping = get_mapping(bag_instance_mapping, mil_train_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, mil_train_neg)

    bag_instance_mapping = get_mapping(bag_instance_mapping, test_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, test_neg)

    max_iterations = 20000
    batch_size = 120

    model = network(input_dim = 4096)
    customobjective = DROLoss(gamma = gamma)
    model.to(device)
    customobjective.to(device)
    bce_loss.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 0.001)
    dro_losses = []
    reg_losses = []
    p_values = []
    test_maps = []
    test_aucs = []
    best_test_map = 0
    lab_ins_output = []
    pos_idx = list(range(len(mil_train_pos)))
    neg_idx = list(range(len(mil_train_neg)))

    for i in range(max_iterations):
        if patience>100:
           break
        model.train()
        
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        pos_data_batch = mil_train_pos[pos_idx[:int(batch_size/2)]]
        train_pos_feat, batch_pos_bag_names = pos_data_batch[:, 0],  pos_data_batch[:, 2]


        squeezed_pos_feats = []
        for x in train_pos_feat:
            squeezed_pos_feats.extend(x)
        squeezed_pos_feats = np.array(squeezed_pos_feats)


        neg_data_batch = mil_train_neg[neg_idx[:int(batch_size/2)]]

        train_neg_feat, batch_neg_bag_names = neg_data_batch[:, 0], neg_data_batch[:, 2]
        squeezed_neg_feats = []
        for x in train_neg_feat:
            squeezed_neg_feats.extend(x)
        squeezed_neg_feats = np.array(squeezed_neg_feats)

        train_feat = np.concatenate([squeezed_pos_feats, squeezed_neg_feats, lab_feats])
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

        lab_outputs = outputs[count:]
        lab_outputs = lab_outputs.reshape(-1, 1)
        ins_gt = torch.from_numpy(lab_labels)
        ins_gt = Variable(ins_gt).to(device)
        dro_loss = customobjective(outputs_positive, outputs_negative, p_values=p)
        reg_loss = bce_loss(lab_outputs, ins_gt.float())
        total_loss = dro_loss+reg_coeff*reg_loss
        total_loss.backward()
        optimizer.step()
        dro_losses.append(dro_loss.data.cpu())
        reg_losses.append(reg_loss.data.cpu())

        
        if i%10==0:
            patience+=1
            model.eval()
            [test_auc, test_ap] = evaluate(model, test_pos, test_neg)
            test_aucs.append(test_auc)
            test_maps.append(test_ap)
           
               
            if test_ap>best_test_map:
                patience =0
                
                best_test_map = test_ap

                torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("dro_active/Pascal_VOC/models/model_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+\
    "_admil_f_entropy_best_map.pth.tar"))
     
    
    dro_losses = np.array(dro_losses)
    reg_losses = np.array(reg_losses)
    test_maps = np.array(test_maps)
    test_aucs = np.array(test_aucs)
    np.save("dro_active/Pascal_VOC/logs/aucs_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", test_aucs)
    np.save("dro_active/Pascal_VOC/logs/maps_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", test_maps)
    np.save("dro_active/Pascal_VOC/logs/dro_losses_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", dro_losses)
    np.save("dro_active/Pascal_VOC/logs/reg_losses_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", reg_losses)
   
    