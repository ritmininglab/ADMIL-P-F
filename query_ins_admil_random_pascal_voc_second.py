
import numpy as np
import torch
from torch.autograd import Variable
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    random.seed(run)
    curr_query = 60
    prev_query = 30
    all_run_pos_output = []
   
    prev_train_pos = np.load("Dataset/Pascal_VOC/Pasadded_pos_data_"+str(run)+"_"+str(prev_query)+"_admil_random.npy", allow_pickle = True)
    prev_train_neg = np.load("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(prev_query)+"_admil_random.npy", allow_pickle = True)

    queried_instance_ids = getinstances(prev_train_pos, prev_train_neg)
    
    indices = list(range(len(train_pos[:, 2].flatten())))
    random.shuffle(indices)
    queried_pos_data = []
    queried_neg_data = []
    out_th = 0.2
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

                bag_feat = feat.reshape(1,input_dim)
                bag_label = np.array([label]).reshape(1, 1)
                total_queried+=1
                if label==1:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_pos_instance_"+str(instance_no)]
                    queried_pos_data.append(np.array(temp))
                elif label==0:
                    temp = [bag_feat, bag_label, 'added_pos_bag_'+str(i)+"_neg_instance_"+str(instance_no)]
                    queried_neg_data.append(np.array(temp))
                else:
                    print("Something abnormal is going on here")

            count+=no_segments

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



    if len(queried_pos_data)+len(queried_neg_data)!=curr_query:
            print("Abnormal Going here")

    np.save("Dataset/Pascal_VOC/added_pos_data_"+str(run)+"_"+str(curr_query)+"_admil_random.npy",queried_pos_data)
    np.save("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(curr_query)+"_admil_random.npy", queried_neg_data)
    
    
    
