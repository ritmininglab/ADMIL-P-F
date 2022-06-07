
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


if __name__=="__main__":

    train_pos, train_neg = np.load("Dataset/Pascal_VOC/train_pos.npy", allow_pickle = True), np.load("Dataset/Pascal_VOC/train_neg.npy", allow_pickle = True)
    bag_instance_mapping = {}
    bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
    bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)

    input_dim = 4096
    
    run = 1
    random.seed(run)
    no_query = 30
    all_run_pos_output = []
   
    
    indices = list(range(len(train_pos[:, 1].flatten())))
    indices = random.sample(indices, no_query)
    queried_pos_data = []
    queried_neg_data = []
    out_th = 0.2
    count = 0
    for i in range(len(train_pos)):
        no_segments = bag_instance_mapping[train_pos[i][2]]
        upper = count+no_segments
        for index in indices:
           
            if index>=count and index<upper:
                instance_no = index-count
                feat = train_pos[i][0][instance_no]
                label = train_pos[i][1][instance_no]

                bag_feat = feat.reshape(1,input_dim)
                bag_label = np.array([label]).reshape(1, 1)
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
    np.save("Dataset/Pascal_VOC/added_pos_data_"+str(run)+"_"+str(no_query)+"_admil_random.npy",queried_pos_data)
    np.save("Dataset/Pascal_VOC/added_neg_data_"+str(run)+"_"+str(no_query)+"_admil_random.npy", queried_neg_data)
    


