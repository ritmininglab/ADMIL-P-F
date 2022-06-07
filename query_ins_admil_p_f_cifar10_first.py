

import numpy as np
import torch
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)


def get_bag_info(bag_no, instance_no, X_train_pos, y_train_pos):
    feat =  X_train_pos[bag_no][instance_no]
    label = y_train_pos[bag_no][instance_no]
    input_dim = X_train_pos.shape[2]
    bag_feat = feat.reshape(1, input_dim)
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
        if closeness[index]>0.2:
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

    
    

def exploration_aggressive(bag_pos_output, bag_pos_feats, X_train_pos, y_train_pos, gamma, no_query, p_th = 0.0, out_th = 0.3):
    bag_dim = X_train_pos.shape[2]
    all_bag_highest_p = []
    all_bag_corr_out = []
    all_bag_corr_label = []
    all_bag_ids = []
    all_bag_ins_nos = []
    
    for i, bag_output in enumerate(bag_pos_output):
         bag_label = y_train_pos[i]
         sub_output = bag_output - max(bag_output)
         p = np.exp(sub_output/gamma)/sum(np.exp(sub_output/gamma))
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
    
    cand_bag_ins_map = {}
    train_pos_data, train_neg_data = [], []
    queried_so_far = 0
    for bag_id in cand_bag_ids:
        print("__\n For a bag", bag_id, "____\n")
        if queried_so_far>=no_query:
            break
        bag_output = bag_pos_output[bag_id]
        top_k_instance_nos = np.argsort(bag_output)[-2:]
        queried_so_far+=2
        for instance_no in top_k_instance_nos:
            feat = X_train_pos[bag_id][instance_no]
            label = y_train_pos[bag_id][instance_no]
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
    
    
                
if __name__=="__main__":
    run = 1
    X_train_pos, X_train_neg = np.load("Dataset/Cifar10/X_train_pos.npy"), np.load("Dataset/Cifar10/X_train_neg.npy")
    y_train_pos, y_train_neg = np.load("Dataset/Cifar10/y_train_pos.npy"), np.load("Dataset/Cifar10/y_train_neg.npy")
    input_dim = X_train_pos.shape[2]
    gammas = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])*1e-06
    no_query = 150
    all_run_pos_output = []
    best_test_map = 0
    for gamma in gammas:
         eval_maps = np.load("dro_passive/Cifar10/logs/maps_"+str(run)+"_"+str(gamma)+".npy")
         max_test_map = max(eval_maps)
         if max_test_map>best_test_map:
             best_test_map = max_test_map
             best_gamma = gamma
         
    
    model_path = "dro_passive/Cifar10/models/model_"+str(run)+"_"+str(best_gamma)+"_best.pth.tar"
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
    [sel_bag_ins_map, explore_pos_data, explore_neg_data] = exploration_aggressive(bag_pos_output, bag_pos_feats, X_train_pos, y_train_pos, best_gamma, no_query)
    print("Explore pos data", len(explore_pos_data), "explore neg data", len(explore_neg_data))
    bags = [k for k,v in sel_bag_ins_map.items()]
    print("Bags", bags, "len", len(bags))
        
    exploit_query_no = no_query-len(explore_pos_data)-len(explore_neg_data)
        
    if exploit_query_no>0:
        [exploit_pos_data, exploit_neg_data] = exploitation(bag_pos_output, exploit_query_no, X_train_pos, y_train_pos, sel_bag_ins_map)
            
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
    np.save("Dataset/Cifar10/added_pos_data_"+str(run)+"_"+str(no_query)+"_admil_p_f.npy",queried_pos_data)    
    np.save("Dataset/Cifar10/added_neg_data_"+str(run)+"_"+str(no_query)+"_admil_p_f.npy", queried_neg_data)
  
    