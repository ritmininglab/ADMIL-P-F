#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:11:13 2021

@author: hiteshsapkota
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 09:31:03 2021

@author: hiteshsapkota
"""
import numpy as np

def getinstances(neg):
    bag_infos = {}
   
    for data in neg:
        
        bag_name = data[2]
        
        if "added" in bag_name:
            [_, _, _, bag_no, _, _, _, instance_no] = bag_name.split("_")
            if int(bag_no) not in bag_infos:
                bag_infos[int(bag_no)] = []
            bag_infos[int(bag_no)].append(int(instance_no))
    return bag_infos



def process_data(mil_train_data, lab_train_data):
    lab_ins_infos = getinstances(lab_train_data)
    processed_mil_data = []
    for i, data in enumerate(mil_train_data):
        if i in lab_ins_infos:
            queried_ins_nos = lab_ins_infos[i]
            all_instance_info = mil_train_data[i]
            feats, bag_name, labels = all_instance_info[0],  all_instance_info[2], all_instance_info[1]
            pro_feats, pro_labels = [], []
            for j, (f, l) in enumerate(zip(feats, labels)):
                if j in queried_ins_nos:
                    continue
                pro_feats.append(f)
                pro_labels.append(l)
            pro_feats, pro_labels = np.array(pro_feats), np.array(pro_labels)
            processed_mil_data.append(np.array([pro_feats, pro_labels, bag_name]))
            print("For bag", bag_name, "original", len(labels), "processed", len(pro_labels))
        else:
            processed_mil_data.append(data)
    processed_mil_data = np.array(processed_mil_data)
    return processed_mil_data
            
        
            
        
    
    
