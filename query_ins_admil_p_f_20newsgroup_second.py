

import numpy as np
import torch
from torch.autograd import Variable
import copy
device = torch.device("cpu")#cuda:0" if torch.cuda.is_available() else "cpu")
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
        extract_feat = self.fc2(x)
        x = self.dropout(extract_feat)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return [x, extract_feat]

def exploitation(ins_pos_outputs, no_query, train_pos, bag_instance_mapping, queried_instances, conf_th = 0.2):
    closeness = np.abs(ins_pos_outputs-0.5)
    indices = np.argsort(closeness)
    train_pos_data = []
    train_neg_data = []
    total_queried = 0

    for index in indices:
        if total_queried>=no_query:
            break
        if closeness[index]>conf_th:
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


    
def get_retrieved_instances(bag_pos_output, selected_instances):
    for a in selected_instances:
        bag_id, instance_id = a
        instance_outputs = bag_pos_output[bag_id][1]
        if max(instance_outputs)!= instance_outputs[instance_id]:
            print("Retrived actual positive instance because of attention for bag", bag_id, "instance", instance_id)
            
def filter_ins(cluster_pts, queried_instances):
    idxs = []
    for pt in cluster_pts:
        if pt in queried_instances:
            continue
        idxs.append(pt)
    idxs = np.array(idxs)
    return idxs
        

              


def exploration_attention_aggressive(bag_pos_output, bag_pos_feats, train_pos, gamma, no_query, queried_instance_ids, p_th = 0.0, out_th = 0.3, bag_dim = 200):
     labels = bag_pos_output[:, 2]
     outputs = bag_pos_output[:, 1]
     all_bag_highest_p = []
     all_bag_corr_out = []
     all_bag_corr_label = []
     all_bag_ids = []
     all_bag_ins_nos = []   
     for i, bag_output in enumerate(outputs):
         bag_label = labels[i]
         if i in queried_instance_ids:
             queried_idxs = np.array(queried_instance_ids[i])
             queried_labels = np.array(labels[i][queried_idxs].flatten())
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
     
     cand_bag_ins_map = {}
     train_pos_data, train_neg_data = [], []
     queried_so_far = 0
    
     
     for bag_id in cand_bag_ids:
         print("\n ___________For a bag", bag_id, "__________\n")
         
         if queried_so_far>=no_query:
             break
         bag_output = outputs[bag_id].flatten()
         if bag_id in queried_instance_ids:
             queried_instances = queried_instance_ids[bag_id]
             queried_labels = labels[bag_id][queried_instances].flatten()
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
                     feat = train_pos[bag_id][0][index]
                     label = labels[bag_id][index]
                     feat = feat.reshape(1, bag_dim)
                     bag_label = label.reshape(1, 1)
                     topk_count+=1
                     if label==1:
                         print("Instance", index, "as positive")
                         temp = [feat, bag_label, "added_pos_bag_"+str(bag_id)+"_pos_instance_explore_"+str(index)]
                         train_pos_data.append(temp)
                     elif label==0:
                        print("Instance", index, "as negative")
                        temp = [feat, bag_label, "added_pos_bag_"+str(bag_id)+"_neg_instance_explore_"+str(index)]
                        train_neg_data.append(temp)
                     else:
                        print("Abnormal")
             queried_so_far+=topk_count
         else:
             topk_instance_nos = np.argsort(bag_output)[::-1][:2]
             for instance_no in topk_instance_nos:
                 if queried_so_far>=no_query:
                     break
                 feat = train_pos[bag_id][0][instance_no]
                 label = labels[bag_id][instance_no]
                 feat = feat.reshape(1, bag_dim)
                 bag_label = label.reshape(1, 1)
                 queried_so_far+=1
                 if label==1:
                         print("Instance", instance_no, "as positive")
                         temp = [feat, bag_label, "added_pos_bag_"+str(bag_id)+"_pos_instance_explore_"+str(instance_no)]
                         train_pos_data.append(temp)
                 elif label==0:
                        print("Instance", instance_no, "as negative")
                        temp = [feat, bag_label, "added_pos_bag_"+str(bag_id)+"_neg_instance_explore_"+str(instance_no)]
                        train_neg_data.append(temp)
                 else:
                     print("Abnormal")
                     
      
     train_pos_data = np.array(train_pos_data)
     train_neg_data = np.array(train_neg_data)
     print("New added pos", len(train_pos_data), "neg", len(train_neg_data))
     new_cand = copy.deepcopy(cand_bag_ins_map) 
     for k, v in queried_instance_ids.items():
         if k not in cand_bag_ins_map:
             cand_bag_ins_map[k] = []
         cand_bag_ins_map[k].extend(v)

     return [cand_bag_ins_map, new_cand, train_pos_data, train_neg_data]

     
     
        

    
 
def get_mapping(mapping, X):
    for i in range(len(X)):
        mapping[X[i][2]] = X[i][0].shape[0]
    return mapping

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
    topic_nos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    total_query = 30
    curr_query = 15
    prev_query = 15
    gammas = np.array([1, 100, 10000, 1000000, 100000000])*1e-06
    reg_coeffs = np.array([1, 10, 100, 1000])*1e-04
    topic_mapping = np.load("topic_map.npy", allow_pickle = True).item()
    all_topic_explore_pos, all_topic_explore_neg, all_topic_exploit_pos, all_topic_exploit_neg = [], [], [], []
    all_topic_explored_bags = []
    for topic_no in topic_nos:
        topic = topic_mapping[topic_no]
        train_pos = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_train_pos.npy", allow_pickle = True)
        train_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_train_neg.npy", allow_pickle = True)

        print("\n ________Working on a topic", topic, "__________\n")
        prev_train_pos = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(prev_query)+"_admil_p_f.npy", allow_pickle = True)
        prev_train_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(prev_query)+"_admil_p_f.npy", allow_pickle = True)
        
        if len(prev_train_pos)+len(prev_train_neg)==0:
             print("Skipping the topic", topic_no)
             np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(total_query)+"_admil_p_f.npy", [])
             np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(total_query)+"_admil_p_f.npy", [])
             all_topic_exploit_pos.append(0)
             all_topic_exploit_neg.append(0)
             all_topic_explore_pos.append(0)
             all_topic_explore_neg.append(0)
             continue
        if prev_query!=15:
            prev_prev_train_pos =  np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(prev_query-15)+"_admil_p_f.npy", allow_pickle = True)
            prev_prev_train_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(prev_query-15)+"_admil_p_f.npy", allow_pickle = True)
            if len(prev_prev_train_pos)+len(prev_prev_train_neg)==len(prev_train_pos)+len(prev_train_neg):
                 print("Skipping the topic here", topic_no)
                 np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(total_query)+"_admil_p_f.npy", prev_train_pos)
                 np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(total_query)+"_admil_p_f.npy", prev_train_neg)
                 all_topic_explore_pos.append(0)
                 all_topic_explore_neg.append(0)
                 all_topic_exploit_pos.append(0)
                 all_topic_exploit_neg.append(0)
                 continue
             
        best_map = 0
        for reg_coeff in reg_coeffs:
            for gamma in gammas:
                test_maps = np.load("dro_active/20NewsGroup/logs/test_maps_"+str(run)+"_"+str(gamma)+"_"+str(topic)+"_"+str(prev_query)+"_"+str(reg_coeff)+"_explore_exploit_attention.npy")
                if max(test_maps)>best_map:
                    best_map = max(test_maps)
                    best_gamma = gamma
                    best_reg_coeff = reg_coeff


       
         
        queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)
        bag_instance_mapping = {}
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)
        
        model_path = "dro_active/20NewsGroup/models/model_"+str(run)+"_"+str(best_gamma)+"_"+str(topic)+"_"+str(prev_query)+"_"+str(best_reg_coeff)+"_explore_exploit_attention_best_map.pth.tar"
        checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
        model = network(200)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        pos_feats, _, pos_instance_labels = train_pos[:, 0],  train_pos[:, 2], train_pos[:, 1]
        squeezed_pos_feats = []
        for x in pos_feats:
            squeezed_pos_feats.extend(x)
        squeezed_pos_feats = np.array(squeezed_pos_feats)
        train_pos_feat = torch.from_numpy(squeezed_pos_feats)
        train_pos_feat = Variable(train_pos_feat).to(device)
        [pos_outputs, pos_feats] = model(train_pos_feat.float())
        pos_outputs = pos_outputs.data.cpu().numpy().flatten()
        pos_feats = pos_feats.data.cpu().numpy()
        bag_pos_output = []
        bag_pos_feats = []
        count = 0
        for i in range(len(train_pos)):
            no_segments = bag_instance_mapping[train_pos[i][2]]
            bag_name = train_pos[i][2]
            upper = count+no_segments
            bag_pos_output.append([bag_name, pos_outputs[count: upper], train_pos[i][1]])
            bag_pos_feats.append(pos_feats[count: upper])
            count+=no_segments
            
        bag_pos_feats = np.array(bag_pos_feats)
        bag_pos_output = np.array(bag_pos_output)
        [sel_bag_ins_map, new_bag_ins_map, explore_pos_data, explore_neg_data] = exploration_attention_aggressive(bag_pos_output, bag_pos_feats, train_pos, best_gamma, curr_query, queried_instance_ids)
        print("Explore pos data", len(explore_pos_data), "explore neg data", len(explore_neg_data))
        bags = [k for k,v in new_bag_ins_map.items()]
        print("Bags", bags, "len", len(bags))
        all_topic_explored_bags.append(len(bags))
        print("Total bags explored so far", len(sel_bag_ins_map))
        exploit_query_no = curr_query-len(explore_pos_data)-len(explore_neg_data)
        all_topic_explore_pos.append(len(explore_pos_data))
        all_topic_explore_neg.append(len(explore_neg_data))
        
        if exploit_query_no>0:
            [exploit_pos_data, exploit_neg_data] = exploitation(pos_outputs, exploit_query_no, train_pos, bag_instance_mapping, sel_bag_ins_map)
            all_topic_exploit_pos.append(len(exploit_pos_data))
            all_topic_exploit_neg.append(len(exploit_neg_data))
            
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
            all_topic_exploit_pos.append(0)
            all_topic_exploit_neg.append(0)





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
        

        np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(total_query)+"_admil_p_f.npy", train_pos_data)
        np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(total_query)+"_admil_p_f.npy", train_neg_data)
      


                
            




