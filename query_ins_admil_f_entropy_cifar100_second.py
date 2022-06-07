
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

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
    X_train_pos, X_train_neg = np.load("Dataset/Cifar100/X_train_pos.npy"), np.load("Dataset/Cifar100/X_train_neg.npy")
    y_train_pos, y_train_neg = np.load("Dataset/Cifar100/y_train_pos.npy"), np.load("Dataset/Cifar100/y_train_neg.npy")
    input_dim = X_train_pos.shape[2]
    gammas = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])*1e-06
    reg_coeffs = np.array([1, 10, 100, 1000, 10000])*1e-04
    curr_query = 300
    prev_query = 150
    best_test_map = 0
    for gamma in gammas:
        for reg_coeff in reg_coeffs:
            maps = np.load("dro_active/Cifar100/logs/test_maps_"+str(run)+"_"+str(gamma)+"_"+str(prev_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy")
            max_test_map = max(maps)
            if max_test_map>best_test_map:
                best_test_map = max_test_map
                best_gamma = gamma
                best_reg_coeff = reg_coeff
    
    prev_train_pos = np.load("Dataset/Cifar100/added_pos_data_"+str(run)+"_"+str(prev_query)+"_admil_f_entropy.npy.npy", allow_pickle = True)    
    prev_train_neg = np.load("Dataset/Cifar100/added_neg_data_"+str(run)+"_"+str(prev_query)+"_admil_f_entropy.npy.npy", allow_pickle = True)
        
    queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)
    
    model_path = "dro_active/Cifar100/models/model_"+str(run)+"_"+str(best_gamma)+"_"+str(prev_query)+"_"+str(best_reg_coeff)+"_admil_f_entropy_best_map.pth.tar"
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
    indices = np.argsort(closeness)
    queried_pos_data = []
    queried_neg_data = []
    total_queried = 0
    out_th = 0.2
    for index in indices:
        if total_queried>=curr_query-prev_query:
            break
        if closeness[index]>out_th:
            continue
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
        queried_neg_data = []
        
        

    np.save("Dataset/Cifar100/added_pos_data_"+str(run)+"_"+str(curr_query)+"_admil_f_entropy.npy",queried_pos_data)    
    np.save("Dataset/Cifar100/added_neg_data_"+str(run)+"_"+str(curr_query)+"_admil_f_entropy.npy", queried_neg_data)
  