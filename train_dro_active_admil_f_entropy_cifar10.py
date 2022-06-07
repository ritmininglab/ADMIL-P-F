
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


def get_mapping(mapping, X, video_names):
    for i in range(len(X)):
        mapping[video_names[i]] = len(X[i])
    return mapping

def evaluate(model, X_pos, X_neg, y_pos, y_neg):
     pos_features = torch.from_numpy(X_pos)
     neg_features = torch.from_numpy(X_neg)
     input_dim = X_pos.shape[2]
     pos_features = pos_features.reshape(-1, input_dim)
     neg_features = neg_features.reshape(-1, input_dim)
     pos_feat = Variable(pos_features).to(device)
     neg_feat = Variable(neg_features).to(device)
     [pos_output, _] = model(pos_feat.float())
     [neg_output, _] = model(neg_feat.float())
     pos_output = pos_output.data.cpu().numpy().flatten()
     neg_output = neg_output.data.cpu().numpy().flatten()
     preds = np.concatenate([pos_output, neg_output])
     gts = np.concatenate([y_pos.flatten(), y_neg.flatten()])
     auc = roc_auc_score(gts, preds)
     ap = average_precision_score(gts, preds)
     return [ap,auc]

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
    patience = 0
    no_query = 150
    run = int(run)
    reg_coeff = float(reg_coeff)*1e-04
    gamma = float(gamma)*1e-06

    no_query = int(no_query)
    lr = 0.01

    mil_train_pos, mil_train_neg = np.load("Dataset/Cifar10/X_train_pos.npy"), np.load("Dataset/Cifar10/X_train_neg.npy")
    video_names_train_pos = np.array(["pos_"+str(i) for i in range(len((mil_train_pos)))])
    video_names_train_neg = np.array(["neg_"+str(i) for i in range(len(mil_train_neg))])


    X_test_pos, X_test_neg = np.load("Dataset/Cifar10/X_test_pos.npy"), np.load("Dataset/Cifar10/X_test_neg.npy")
    y_test_pos, y_test_neg = np.load("Dataset/Cifar10/y_test_pos.npy"), np.load("Dataset/Cifar10/y_test_neg.npy")
    input_dim = X_test_pos.shape[2]
    lab_train_pos = np.load("Dataset/Cifar/added_pos_data_"+str(run)+"_"+str(no_query)+"_admil_f_entropy.npy", allow_pickle = True)
    lab_train_neg = np.load("Dataset/Cifar/added_neg_data_"+str(run)+"_"+str(no_query)+"_admil_f_entropy.npy", allow_pickle = True)
    
    lab_feats, lab_labels, _ = process_labeled_ins(lab_train_pos, lab_train_neg)
    img_size = X_test_pos.shape[2]
    lab_feats = lab_feats.reshape(-1, input_dim)

    bag_instance_mapping = {}
    bag_instance_mapping = get_mapping(bag_instance_mapping, mil_train_pos, video_names_train_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, mil_train_neg, video_names_train_neg)

    max_iterations = 20000
    batch_size = 120

    model = network(input_dim = input_dim)
    customobjective = DROLoss(gamma = gamma)
    model.to(device)
    customobjective.to(device)
    bce_loss.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 0.001)

    dro_losses = []
    reg_losses = []
    p_values = []
    best_test_map = 0
    test_aucs = []
    test_maps = []
    lab_ins_output = []
    pos_idx = list(range(len(mil_train_pos)))
    neg_idx = list(range(len(mil_train_neg)))
    for i in range(max_iterations):
        if patience>100:
           break
        model.train()
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        train_pos_feat, batch_pos_bag_names = mil_train_pos[pos_idx[:int(batch_size/2)]], video_names_train_pos[pos_idx[:int(batch_size/2)]]


        squeezed_pos_feats = []
        for x in train_pos_feat:
            squeezed_pos_feats.extend(x)
        squeezed_pos_feats = np.array(squeezed_pos_feats)

        squeezed_pos_feats = squeezed_pos_feats.reshape(-1, input_dim)

        train_neg_feat, batch_neg_bag_names = mil_train_neg[neg_idx[:int(batch_size/2)]], video_names_train_neg[neg_idx[:int(batch_size/2)]]

        squeezed_neg_feats = []
        for x in train_neg_feat:
            squeezed_neg_feats.extend(x)
        squeezed_neg_feats = np.array(squeezed_neg_feats)
        squeezed_neg_feats = squeezed_neg_feats.reshape(-1, input_dim)

        train_feat = np.concatenate([squeezed_pos_feats, squeezed_neg_feats, lab_feats])
        train_feat = np.array(train_feat, dtype = np.float)
        train_feat = torch.from_numpy(train_feat)

        train_feat = Variable(train_feat, requires_grad = True).to(device)
        optimizer.zero_grad()
        [outputs, _] = model(train_feat.float())

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
            [test_ap, test_auc] = evaluate(model, X_test_pos, X_test_neg, y_test_pos, y_test_neg)
            test_aucs.append(test_auc)
            test_maps.append(test_ap)
            print("MAP Score in this iteration",i, test_ap)
            if test_ap>best_test_map:
                patience = 0
                best_test_map = test_ap
                torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("dro_active/Cifar10/models/model_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+\
    "_admil_f_entropy_best_map.pth.tar"))





    dro_losses = np.array(dro_losses)
    reg_losses = np.array(reg_losses)
    test_aucs = np.array(test_aucs)
    test_maps = np.array(test_maps)
    np.save("dro_active/Cifar10/logs/test_aucs_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", test_aucs)
    np.save("dro_active/Cifar10/logs/test_maps_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", test_maps)
    np.save("dro_active/Cifar10/logs/dro_losses_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", dro_losses)
    np.save("dro_active/Cifar10/logs/reg_losses_"+str(run)+"_"+str(gamma)+"_"+str(no_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy", reg_losses)



