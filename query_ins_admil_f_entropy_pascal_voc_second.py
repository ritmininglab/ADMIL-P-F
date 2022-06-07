
import numpy as np
import torch
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


def get_mapping(mapping, X):
    for i in range(len(X)):
        mapping[X[i][2]] = X[i][0].shape[0]
    return mapping

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

    train_pos, train_neg = np.load("Dataset/Pascal_VOC/train_pos.npy", allow_pickle = True), np.load("Dataset/Pascal_VOC/train_neg.npy", allow_pickle = True)
    bag_instance_mapping = {}
    bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)

    input_dim = 4096
    gammas = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])*1e-06
    reg_coeffs = np.array([1, 10, 100, 1000, 10000])*1e-04
    run = 1
    curr_query = 60
    prev_query = 30
    all_run_pos_output = []
    best_test_map = 0
    for gamma in gammas:
        for reg_coeff in reg_coeffs:
            maps = np.load("dro_active/Pascal_VOC/logs/maps_"+str(run)+"_"+str(gamma)+"_"+str(prev_query)+"_"+str(reg_coeff)+"_admil_f_entropy.npy")
            max_test_map = max(maps)
            if max_test_map>best_test_map:
                best_test_map = max_test_map
                best_gamma = gamma
                best_reg_coeff = reg_coeff
                

    prev_train_pos = np.load("Dataset/Pascal_VOCadded_pos_data_"+str(run)+"_"+str(prev_query)+"_max_conf.npy", allow_pickle = True)
    prev_train_neg = np.load("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(prev_query)+"_max_conf.npy", allow_pickle = True)

    queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)


    model_path = "dro_active/Pascal_VOC/models/model_"+str(run)+"_"+str(best_gamma)+"_"+str(prev_query)+"_"+str(best_reg_coeff)+"_admil_f_entropy_best_map.pth.tar"
    checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
    print("Loading Model from:", model_path)
    model = network(input_dim = input_dim)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    pos_features, pos_instance_labels = train_pos[:, 0], train_pos[:, 1]
    
    squeezed_pos_feats = []
    for x in pos_features:
        squeezed_pos_feats.extend(x)
    squeezed_pos_feats = np.array(squeezed_pos_feats)
    train_pos_feat = torch.from_numpy(squeezed_pos_feats)
    train_pos_feat = Variable(train_pos_feat).to(device)
    [pos_output, _] = model(train_pos_feat)
    pos_output = pos_output.data.cpu().numpy()
    pos_output = pos_output.flatten()
    closeness = np.abs(pos_output-0.5)
    indices = np.argsort(closeness)
    queried_pos_data = []
    queried_neg_data = []
    out_th = 0.2
    total_queried = 0
   
    for index in indices:
        if total_queried>=curr_query-prev_query:
            break
        if closeness[index]>out_th:
            continue
        count = 0
        for i in range(len(train_pos)):
            no_segments = bag_instance_mapping[train_pos[i][2]]
            upper = count+no_segments
        
            if index>=count and index<upper:
                instance_no = index-count
                if i in queried_instance_ids:
                    if instance_no in queried_instance_ids[i]:
                        count+=no_segments
                        continue
                        
                feat = train_pos[i][0][instance_no]
                label = train_pos[i][1][instance_no]

                bag_feat = feat.reshape(1,input_dim)
                bag_label = np.array([label]).reshape(1, 1)
                total_queried+=1
                if label==1:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_"+str(instance_no)]
                    queried_pos_data.append(np.array(temp))
                elif label==0:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_"+str(instance_no)]
                    queried_neg_data.append(np.array(temp))
                
                    
            count+=no_segments

    queried_pos_data = np.array(queried_pos_data)
    queried_neg_data = np.array(queried_neg_data)
    
    

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



    if len(queried_pos_data)+len(queried_neg_data)!=curr_query:
            print("Abnormal Going here")

    np.save("Dataset/Pascal_VOC/added_pos_data_"+str(run)+"_"+str(curr_query)+"_admil_f_entropy.npy",queried_pos_data)
    np.save("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(curr_query)+"_admil_f_entropy.npy", queried_neg_data)
   


    
    
