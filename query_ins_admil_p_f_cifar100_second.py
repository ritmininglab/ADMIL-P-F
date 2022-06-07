
import numpy as np
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)
from torch.autograd import Variable
from sklearn.cluster import KMeans

import copy

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



    
def get_retrieved_instances(bag_pos_output, selected_instances):
    for a in selected_instances:
        bag_id, instance_id = a
        instance_outputs = bag_pos_output[bag_id]
        if max(instance_outputs)!= instance_outputs[instance_id]:
            print("Retrived actual positive instance because of attention for bag", bag_id, "instance", instance_id)

            
def exploration_attention_aggressive(bag_pos_output, bag_pos_feats, X_train_pos, y_train_pos, gamma, no_query, queried_instance_ids, p_th = 0.0, out_th = 0.3, n_clusters=2):
    bag_dim = X_train_pos.shape[2]
    all_bag_highest_p = []
    all_bag_corr_out = []
    all_bag_corr_label = []
    all_bag_ids = []
    all_bag_ins_nos = []
    
    for i, bag_output in enumerate(bag_pos_output):
         bag_label = y_train_pos[i]
         if i in queried_instance_ids:
             queried_idxs = np.array(queried_instance_ids[i])
             queried_labels = np.array(bag_label[queried_idxs].flatten())
             neg_queried_idxs = queried_idxs[queried_labels==0]
             print("For bag ",i, "total prev neg queries:", len(neg_queried_idxs), "total prev pos queries:", len(queried_idxs)-len(neg_queried_idxs))
         else:
             neg_queried_idxs = np.array([])
             
         unqueried_idxs = list(set(range(len(bag_output)))-set(list(neg_queried_idxs)))
         unqueried_bag_output = bag_output[unqueried_idxs]
         unqueried_sub_output = unqueried_bag_output-max(unqueried_bag_output)
         p = np.zeros(len(bag_output))
         unq_p = np.exp(unqueried_sub_output/gamma)/sum(np.exp(unqueried_sub_output/gamma))
         p[unqueried_idxs] = unq_p
         
         all_bag_corr_out.append(bag_output[np.argwhere(p==max(p))[0][0]])
         all_bag_highest_p.append(max(p))

         all_bag_corr_label.append(bag_label[np.argwhere(p==max(p))[0][0]])
         all_bag_ids.append(i)
         all_bag_ins_nos.append(np.argwhere(p==max(p))[0][0])
    

    all_bag_highest_p = np.array(all_bag_highest_p)
    all_bag_corr_out = np.array(all_bag_corr_out)
    all_bag_corr_label = np.array(all_bag_corr_label).flatten()
    all_bag_ins_nos = np.array(all_bag_ins_nos)
    all_bag_ids = np.array(all_bag_ids) 
    
    indices = np.argsort(all_bag_corr_out)
    selected_ins_p = all_bag_highest_p[indices]
    selected_ins_out = all_bag_corr_out[indices]
    selected_bag_ids = all_bag_ids[indices]
    cand_bag_ids = selected_bag_ids[(selected_ins_out<out_th)& (selected_ins_p>p_th)]
    queried_so_far = 0
    cand_bag_ins_map = {}
    train_pos_data, train_neg_data = [], []
    pos_queried_bag_ins = []
    
    for bag_id in cand_bag_ids:
        print("\n____For bag", bag_id, "_______\n")
        if queried_so_far>=no_query:
            break
        bag_output = bag_pos_output[bag_id]
        
        if bag_id in queried_instance_ids:
            queried_instances = queried_instance_ids[bag_id]
            queried_labels = y_train_pos[bag_id][queried_instances].flatten()
            if 1 in queried_labels:
                print("Already found TP instance")
                continue
            all_indices = np.argsort(bag_output)[::-1]
            topk_count = 0
            for index in all_indices:
                if (topk_count+queried_so_far)>=no_query:
                    break
                if topk_count>=2:
                    break
                if index in queried_instances:
                    print("Skipping the queried negative instance")
                
                else:
                    feat = X_train_pos[bag_id][index]
                    label = y_train_pos[bag_id][index]
                    feat = feat.reshape(1, bag_dim)
                    bag_label = label.reshape(1, 1)
                    
                    topk_count+=1
                    if label==1:
                        print("Instance", index, "as positive")
                        temp = [feat, bag_label, 'added_pos_bag_'+str(bag_id)+"_pos_instance_explore_"+str(index)]
                        train_pos_data.append(np.array(temp))
                        pos_queried_bag_ins.append([bag_id, index])

                    elif label == 0:
                        print("Instance", index, "as negative")
                        temp = [feat, bag_label, 'added_pos_bag_'+str(bag_id)+"_neg_instance_explore_"+str(index)]
                        train_neg_data.append(np.array(temp))
                    else:
                        print("Abnormal")
                        
            queried_so_far+=topk_count
        else:
            top_k_instance_nos = np.argsort(bag_output)[::-1][:2]
            for instance_no in top_k_instance_nos:
                if queried_so_far>=no_query:
                    break
                
                feat = X_train_pos[bag_id][instance_no]
                label = y_train_pos[bag_id][instance_no]
                feat = feat.reshape(1, bag_dim)
                bag_label = label.reshape(1, 1)
                queried_so_far+=1
                if label==1:
                    print("Instance", instance_no, "as positive")
                    temp = [feat, bag_label, 'added_pos_bag_'+str(bag_id)+"_pos_instance_explore_"+str(instance_no)]
                    train_pos_data.append(np.array(temp))
    
                elif label == 0:
                     print("Instance", instance_no, "as negative")
                     temp = [feat, bag_label, 'added_pos_bag_'+str(bag_id)+"_neg_instance_explore_"+str(instance_no)]
                     train_neg_data.append(np.array(temp))
                else:
                    print("Abnormal")
           
    get_retrieved_instances(bag_pos_output, pos_queried_bag_ins)        
    train_pos_data = np.array(train_pos_data)
    train_neg_data = np.array(train_neg_data)

    print("New added pos", len(train_pos_data), "neg", len(train_neg_data))
    new_cand = copy.deepcopy(cand_bag_ins_map) 
    for k, v in queried_instance_ids.items():
        if k not in cand_bag_ins_map:
            cand_bag_ins_map[k] = []
        cand_bag_ins_map[k].extend(v)

    return [cand_bag_ins_map, new_cand, train_pos_data, train_neg_data]

 
def get_bag_info(bag_no, instance_no, X_train_pos, y_train_pos):
    input_dim = X_train_pos.shape[2]
    feat =  X_train_pos[bag_no][instance_no]
    label = y_train_pos[bag_no][instance_no]
    bag_feat = feat.reshape(1, input_dim)
    bag_label = np.array([label]).reshape(1, 1)
    return bag_feat, bag_label, label

 
def exploitation(ins_pos_outputs, no_query, X_train_pos, y_train_pos, queried_instances, out_th = 0.2):
    ins_pos_outputs = ins_pos_outputs.flatten()
    closeness = np.abs(ins_pos_outputs-0.5)
    indices = np.argsort(closeness)
    train_pos_data = []
    train_neg_data = []
    total_queried = 0

    for index in indices:
       
        if total_queried>=no_query:
            break
        if closeness[index]>out_th:
            continue
        for i in range(len(X_train_pos)):
            lower = i*32
            upper = (i+1)*32
            if index>=lower and index<upper:
                instance_no = index-lower
                if i in queried_instances:
                    if instance_no in queried_instances[i]:
                        continue
                    
                [bag_feat, bag_label, label] = get_bag_info(i, instance_no, X_train_pos, y_train_pos)
                total_queried+=1
                if label==1:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_exploit_"+str(instance_no)]
                    train_pos_data.append(np.array(temp))
                  

                elif label == 0:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_exploit_"+str(instance_no)]
                    train_neg_data.append(np.array(temp))
                   
                else:
                    print("Abnormal")
            

    train_pos_data = np.array(train_pos_data)
    train_neg_data = np.array(train_neg_data)
    return [train_pos_data, train_neg_data]
    
    
    
def getinstances(pos, neg):
    bag_infos = {}
    for data in pos:
        bag_name = data[2]
        if "added" in bag_name:
            [_, _, _, bag_no, _, _, _, instance_no] = bag_name.split("_")
            if int(bag_no) not in bag_infos: 
                bag_infos[int(bag_no)] = []
            bag_infos[int(bag_no)].append(int(instance_no))
    for data in neg:
        bag_name = data[2]
        if "added" in bag_name:
            [_, _, _, bag_no, _, _, _, instance_no] = bag_name.split("_")
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
  
    total_query = 300
    curr_query = 150
    prev_query = 150
    best_test_map = 0
    for gamma in gammas:
        for reg_coeff in reg_coeffs:
            maps = np.load("dro_active/Cifar100/logs/test_maps_"+str(run)+"_"+str(gamma)+"_"+str(prev_query)+"_"+str(reg_coeff)+"_admil_p_f.npy")
            max_test_map = max(maps)
            if max_test_map>best_test_map:
                best_test_map = max_test_map
                best_gamma = gamma
                best_reg_coeff = reg_coeff
  
    prev_train_pos = np.load("Dataset/Cifar100/added_pos_data_"+str(run)+"_"+str(prev_query)+"_admil_p_f.npy", allow_pickle = True)    
    prev_train_neg = np.load("Dataset/Cifar100/added_neg_data_"+str(run)+"_"+str(prev_query)+"_admil_p_f.npy", allow_pickle = True)
        
    queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)
    
    model_path = "dro_active/Cifar100/models/model_"+str(run)+"_"+str(best_gamma)+"_"+str(prev_query)+"_"+str(best_reg_coeff)+"_admil_p_f_best_map.pth.tar"
    checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
    print("Loading Model from:", model_path)
    model = network(input_dim = input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    pos_features = torch.from_numpy(X_train_pos)
    pos_features = pos_features.reshape(-1, input_dim)
    pos_feat = Variable(pos_features).to(device)
    [pos_output, processed_feats] = model(pos_feat.float())
    pos_output = pos_output.data.cpu().numpy()
    pos_output = pos_output.flatten()
    processed_feats = processed_feats.data.cpu().numpy()
    bag_pos_output = []
    bag_pos_feats = []
    for i in range(len(X_train_pos)):
            lower = i*32
            upper = (i+1)*32
            bag_pos_output.append(pos_output[lower: upper])
            bag_pos_feats.append(processed_feats[lower: upper])
            
            
    bag_pos_output = np.array(bag_pos_output)
    bag_pos_feats = np.array(bag_pos_feats)
    [sel_bag_ins_map, new_bag_ins_map, explore_pos_data, explore_neg_data] = exploration_attention_aggressive(bag_pos_output, bag_pos_feats, X_train_pos, y_train_pos, best_gamma, curr_query, queried_instance_ids)
    bags = [k for k,v in new_bag_ins_map.items()]
    print("Bags", bags, "len", len(bags))
    print("Total bags explored so far", len(sel_bag_ins_map))
    exploit_query_no = curr_query-len(explore_pos_data)-len(explore_neg_data)
    print("Number of instances to be exploited", exploit_query_no)
    if exploit_query_no>0:
        [exploit_pos_data, exploit_neg_data] = exploitation(bag_pos_output, exploit_query_no, X_train_pos, y_train_pos, sel_bag_ins_map)
       
        if len(exploit_pos_data)>0 and len(explore_pos_data)>0:
            queried_pos_data = np.concatenate([explore_pos_data, exploit_pos_data])
        elif len(exploit_pos_data)>0 and len(explore_pos_data)==0:
            queried_pos_data = exploit_pos_data
        elif len(exploit_pos_data)==0 and len(explore_pos_data)>0:
            queried_pos_data = explore_pos_data
        else:
            queried_pos_data=[]

        if len(exploit_neg_data)>0 and len(explore_neg_data)>0:
            queried_neg_data = np.concatenate([explore_neg_data, exploit_neg_data])
        elif len(exploit_neg_data)>0 and len(explore_neg_data)==0:
            queried_neg_data = exploit_neg_data
        elif len(exploit_neg_data)==0 and len(explore_neg_data)>0:
            queried_neg_data = explore_neg_data
        else:
            queried_neg_data = []


    else:
        queried_pos_data  = explore_pos_data
        queried_neg_data = explore_neg_data
        exploit_pos_data = []
        exploit_neg_data = []        
    if len(queried_pos_data)>0 and len(prev_train_pos)>0:
        train_pos_data = np.concatenate([queried_pos_data, prev_train_pos])
    elif len(queried_pos_data)==0 and len(prev_train_pos)>0:
        train_pos_data = prev_train_pos
    elif len(queried_pos_data)>0 and len(prev_train_pos)==0:
        train_pos_data = queried_pos_data
    else:
        train_pos_data = []

    if len(queried_neg_data)>0 and len(prev_train_neg)>0:
        train_neg_data = np.concatenate([queried_neg_data, prev_train_neg])
    elif len(queried_neg_data)==0 and len(prev_train_neg)>0:
        train_neg_data = prev_train_neg
    elif len(queried_neg_data)>0 and len(prev_train_neg)==0:
        train_neg_data = queried_neg_data
    else:
        train_neg_data = []
    
    if len(train_pos_data)+len(train_neg_data)!=total_query:
        print("Abnormal Going on here")

    np.save("Dataset/Cifar100/added_pos_data_"+str(run)+"_"+str(total_query)+"_admil_p_f.npy", train_pos_data)
    np.save("Dataset/Cifar100/added_neg_data_"+str(run)+"_"+str(total_query)+"_admil_p_f.npy", train_neg_data)

   

