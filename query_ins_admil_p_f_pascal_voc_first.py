

import numpy as np
import torch
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)


def get_bag_info(bag_no, instance_no, data):
    feat =  data[bag_no][0][instance_no]
    label = data[bag_no][1][instance_no]
    bag_feat = feat.reshape(1, feat.shape[0])
    bag_label = np.array([label]).reshape(1, 1)
    return bag_feat, bag_label, label

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

def exploitation(ins_pos_outputs, no_query, train_pos,  bag_instance_mapping, queried_instances, out_th = 0.2):
    closeness = np.abs(ins_pos_outputs-0.5)
    indices = np.argsort(closeness)
    train_pos_data = []
    train_neg_data = []
    total_queried = 0

    for index in indices:
        if total_queried>=no_query:
            break
        if closeness[index]>0.2:
            continue
        count = 0
        for i in range(len(train_pos)):
            no_segments = bag_instance_mapping[train_pos[i][2]]
            upper = count+no_segments
            
            if index>=count and index<upper:
                instance_no = index-count
                if i in queried_instances:
                    if instance_no in queried_instances[i]:
                        count+=no_segments
                        continue

                [bag_feat, bag_label, label] = get_bag_info(i, instance_no, train_pos)
                total_queried+=1
                if label==1:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_exploit_"+str(instance_no)]
                    train_pos_data.append(np.array(temp))

                elif label == 0:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_exploit_"+str(instance_no)]
                    train_neg_data.append(np.array(temp))
                else:
                    print("Abnormal")
            count+=no_segments


    train_pos_data = np.array(train_pos_data)
    train_neg_data = np.array(train_neg_data)
    return [train_pos_data, train_neg_data]


def exploration_aggressive(bag_pos_output,  train_pos,  gamma, no_query, p_th = 0.0, out_th = 0.3, bag_dim = 4096):
    labels = bag_pos_output[:, 2]
    outputs = bag_pos_output[:, 1]
    
    all_bag_highest_p = []
    all_bag_corr_out = []
    all_bag_corr_label = []
    all_bag_ids = []
    all_bag_ins_nos = []

    for i, bag_output in enumerate(outputs):
         bag_label = labels[i]
         sub_output = bag_output - max(bag_output)
         p = np.exp(sub_output/gamma)/sum(np.exp(sub_output/gamma))
         all_bag_corr_out.append(bag_output[np.argwhere(p==max(p))[0][0]])
         all_bag_highest_p.append(max(p))

         all_bag_corr_label.append(bag_label[np.argwhere(p==max(p))[0][0]])
         all_bag_ids.append(i)
         all_bag_ins_nos.append(np.argwhere(p==max(p))[0][0])


    all_bag_highest_p = np.array(all_bag_highest_p).flatten()
    all_bag_corr_out = np.array(all_bag_corr_out).flatten()
    all_bag_corr_label = np.array(all_bag_corr_label).flatten()
    all_bag_ins_nos = np.array(all_bag_ins_nos).flatten()
    all_bag_ids = np.array(all_bag_ids).flatten()

    indices = np.argsort(all_bag_corr_out)
    
    selected_ins_p = all_bag_highest_p[indices]
    selected_ins_out = all_bag_corr_out[indices]
    selected_bag_ids = all_bag_ids[indices]
    
    cand_bag_ids = selected_bag_ids[(selected_ins_out<out_th)& (selected_ins_p>p_th)]
   
    cand_bag_ins_map = {}
    train_pos_data, train_neg_data = [], []
    queried_so_far = 0
    for bag_id in cand_bag_ids:
        print("__\n For a bag", bag_id, "____\n")
        if queried_so_far>=no_query:
            break
        bag_output = outputs[bag_id]
       
        top_k_instance_nos = np.argsort(outputs[bag_id].flatten())[-2:]
        queried_so_far+=2
        for instance_no in top_k_instance_nos:
            feat = train_pos[bag_id][0][instance_no]
            label = labels[bag_id][instance_no]
            feat = feat.reshape(1, bag_dim)
            bag_label = label.reshape(1, 1)
            if label==1:
                print("Instance", instance_no, " as positive")
                temp = [feat, bag_label, 'added_pos_bag_'+str(bag_id)+"_pos_instance_explore_"+str(instance_no)]
                train_pos_data.append(np.array(temp))

            elif label == 0:
                 print("Instance", instance_no, "as negative")
                 temp = [feat, bag_label, 'added_pos_bag_'+str(bag_id)+"_neg_instance_explore_"+str(instance_no)]
                 train_neg_data.append(np.array(temp))
            else:
                print("Abnormal")
    train_pos_data = np.array(train_pos_data)
    train_neg_data = np.array(train_neg_data)

    return [cand_bag_ins_map, train_pos_data, train_neg_data]

def get_mapping(mapping, X):
    for i in range(len(X)):
        mapping[X[i][2]] = X[i][0].shape[0]
    return mapping


if __name__=="__main__":
        train_pos, train_neg = np.load("Dataset/Pascal_VOC/train_pos..npy", allow_pickle = True), np.load("Dataset/Pascal_VOC/train_neg"+".npy", allow_pickle = True)
        bag_instance_mapping = {}
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)
    
        input_dim = 4096
        gammas = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])*1e-06
        run = 1
        no_query = 30
        all_run_pos_output = []
        best_test_map = 0
        for gamma in gammas:
             eval_maps = np.load("dro_passive/Pascal_VOC/logs/test_maps_"+str(run)+"_"+str(gamma)+".npy")
             max_test_map = max(eval_maps)
             if max_test_map>best_test_map:
                 best_test_map = max_test_map
                 best_gamma = gamma
    
    
        model_path = "dro_passive/Pascal_VOC/models/model_"+str(run)+"_"+str(best_gamma)+"_best_map.pth.tar"
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
        [pos_outputs, _] = model(train_pos_feat)
        pos_outputs = pos_outputs.data.cpu().numpy().flatten()
        bag_pos_output = []
        count = 0
        
        for i in range(len(train_pos)):
            no_segments = bag_instance_mapping[train_pos[i][2]]
            bag_name = train_pos[i][2]
            upper = count+no_segments
            bag_pos_output.append([bag_name, pos_outputs[count: upper], train_pos[i][1]])
            count+=no_segments
    
    
    
        bag_pos_output = np.array(bag_pos_output)
        [sel_bag_ins_map, explore_pos_data, explore_neg_data] = exploration_aggressive(bag_pos_output, train_pos, best_gamma, no_query)
        print("Explore pos data", len(explore_pos_data), "explore neg data", len(explore_neg_data))
        bags = [k for k,v in sel_bag_ins_map.items()]
        print("Bags", bags, "len", len(bags))
    
        exploit_query_no = no_query-len(explore_pos_data)-len(explore_neg_data)
        if exploit_query_no>0:
            [exploit_pos_data, exploit_neg_data] = exploitation(pos_outputs, exploit_query_no, train_pos, bag_instance_mapping,  sel_bag_ins_map)
    
            if len(exploit_pos_data)>0 and len(explore_pos_data)>0:
                queried_pos_data = np.concatenate([explore_pos_data, exploit_pos_data])
    
            elif len(exploit_pos_data)>0 and len(explore_pos_data)==0:
                queried_pos_data = exploit_pos_data
    
            elif len(exploit_pos_data)==0 and len(explore_pos_data)>0:
                queried_pos_data = explore_pos_data
    
            elif len(exploit_pos_data)==0 and len(explore_pos_data)==0:
                queried_pos_data = []
    
            if len(exploit_neg_data)>0 and len(explore_neg_data)>0:
                queried_neg_data = np.concatenate([explore_neg_data, exploit_neg_data])
            elif len(exploit_neg_data)>0 and len(explore_neg_data)==0:
                queried_neg_data = exploit_neg_data
            elif len(exploit_neg_data)==0 and len(explore_neg_data)>0:
                queried_neg_data = explore_neg_data
            elif len(exploit_neg_data)==0 and len(explore_neg_data)==0:
                queried_neg_data = []
    
    
    
        else:
            queried_pos_data  = explore_pos_data
            queried_neg_data = explore_neg_data
            exploit_pos_data = []
            exploit_neg_data = []
        np.save("Dataset/Pascal_VOC/added_pos_data_"+str(run)+"_"+str(no_query)+"_admil_p_f.npy",queried_pos_data)
        np.save("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(no_query)+"_admil_p_f.npy", queried_neg_data)
        
        
        
                
