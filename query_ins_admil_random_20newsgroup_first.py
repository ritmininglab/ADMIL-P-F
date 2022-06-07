

import numpy as np
import torch
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)
import random



def get_mapping(mapping, X):
    for i in range(len(X)):
        mapping[X[i][2]] = X[i][0].shape[0]
    return mapping



if __name__=="__main__":
    run = 1
    random.seed(run)
    topic_nos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    no_query = 15
    gammas = np.array([1, 100, 10000, 1000000, 100000000])*1e-06
    conf_th = 0.2
    topic_mapping = np.load("topic_map.npy", allow_pickle = True).item()
    all_topic_pos, all_topic_neg = [], []
    for topic_no in topic_nos:
        topic = topic_mapping[topic_no]
        print("\n ________Working on a topic", topic, "__________\n")
        train_pos = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_train_pos.npy", allow_pickle = True)
        train_neg = np.load("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_train_neg.npy", allow_pickle = True)
        bag_instance_mapping = {}
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_pos)
        bag_instance_mapping = get_mapping(bag_instance_mapping, train_neg)

        best_map = 0
        for gamma in gammas:
            test_maps =np.load("dro_passive/20NewsGroup/logs/test_maps_"+str(run)+"_"+str(gamma)+"_"+str(topic)+".npy")
            if max(test_maps)>best_map:
                best_map = max(test_maps)
                best_gamma = gamma
        model_path = "dro_passive/20NewsGroup/models/model_"+str(run)+"_"+str(best_gamma)+"_"+str(topic)+"_best_map.pth.tar"
        checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
       
        all_idxs = list(range(len(train_pos[:, 1].flatten())))
        indices = random.sample(all_idxs, no_query)

        
        queried_pos_data = []
        queried_neg_data = []
        count = 0
        for i in range(len(train_pos)):
            no_segments = bag_instance_mapping[train_pos[i][2]]
            upper = count+no_segments
            for index in indices:
                if index>=count and index<upper:
                    instance_no = index-count
                    feat = train_pos[i][0][instance_no]
                    label = train_pos[i][1][instance_no]
                    bag_feat = feat.reshape(1, feat.shape[0])
                    bag_label = np.array([label]).reshape(1, 1)
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
        all_topic_pos.append(len(queried_pos_data))
        all_topic_neg.append(len(queried_neg_data))

        print("Fetched positive instance", len(queried_pos_data), "negative instance", len(queried_neg_data))

        np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_pos_"+str(no_query)+"_admil_random.npy", queried_pos_data)
        np.save("Dataset/20NewsGroup/rep_"+str(run)+"_topic_"+topic+"_added_neg_"+str(no_query)+"_admil_random.npy", queried_neg_data)
        
    