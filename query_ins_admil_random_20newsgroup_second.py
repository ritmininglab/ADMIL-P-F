

import numpy as np
import torch
from torch.autograd import Variable
import random
device = torch.device("cpu")#cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)



def get_mapping(mapping, X):
    for i in range(len(X)):
        mapping[X[i][2]] = X[i][0].shape[0]
    return mapping

def getinstances(pos, neg):
    bag_infos = {}
    for data in pos:

        bag_name = data[2]

        if "added" in bag_name:
            [_, _, _, bag_no, _, _,instance_no] = bag_name.split("_")
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
    random.seed(run)
    topic_nos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    curr_query = 30
    prev_query = 15

    gammas = np.array([1, 100, 10000, 1000000, 100000000])*1e-06
    reg_coeffs = np.array([1, 10, 100, 1000])*1e-04
    topic_mapping = np.load("topic_map.npy", allow_pickle = True).item()
    all_topic_pos, all_topic_neg = [], []
    for topic_no in topic_nos:

        topic = topic_mapping[topic_no]
        prev_train_pos = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(prev_query)+"_admil_random.npy", allow_pickle = True)
        prev_train_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(prev_query)+"_admil_random.npy", allow_pickle = True)


        queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)


        train_pos = np.load("Dataset/20NewsGroup/rep_"+str(1)+"_topic_"+topic+"_train_pos.npy", allow_pickle = True)
        train_neg = np.load("Dataset/20NewsGroup/rep_"+str(1)+"_topic_"+topic+"_train_neg.npy", allow_pickle = True)
        bag_instance_mapping = {}
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)

        
        indices = list(range(len(train_pos[:, 1].flatten())))
        random.shuffle(indices)

       
        queried_pos_data = []
        queried_neg_data = []
        total_queried = 0
        for index in indices:
            if total_queried>=curr_query-prev_query:
                break
            
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
                    bag_feat = feat.reshape(1, feat.shape[0])
                    bag_label = np.array([label]).reshape(1, 1)
                    total_queried+=1
                    if label==1:
                        temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_"+str(instance_no)]
                        queried_pos_data.append(np.array(temp))
                    elif label==0:
                        temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_"+str(instance_no)]
                        queried_neg_data.append(np.array(temp))
                    else:
                        print("Abnormal")

                count+=no_segments

        queried_pos_data = np.array(queried_pos_data)
        queried_neg_data = np.array(queried_neg_data)
        print("Added: For query", curr_query-prev_query, "fetched positive instance", len(queried_pos_data), "negative instance", len(queried_neg_data))
        all_topic_pos.append(len(queried_pos_data))
        all_topic_neg.append(len(queried_neg_data))
        print("Pos:", len(queried_pos_data), "Neg:", len(queried_neg_data))
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



        np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(curr_query)+"_admil_random.npy", queried_pos_data)
        np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(curr_query)+"_admil_random.npy", queried_neg_data)
       
    