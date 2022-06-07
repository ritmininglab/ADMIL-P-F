
import pickle
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
import cv2
from keras.applications.vgg16 import preprocess_input

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
    
def load_cifar_100_data():
    
    """Loading training dataset"""
    
    train_data = unpickle('raw_data/train')
   
    X_train = train_data[b'data']
    y_train_coarse = train_data[b'coarse_labels']
    train_image_names = train_data[b'filenames']
    
    """Loading testing dataset"""
    
    test_data = unpickle('raw_data/test')
    X_test = test_data[b'data']
    y_test_coarse = test_data[b'coarse_labels']
    test_image_names = test_data[b'filenames']
    
    X_train, y_train_coarse, train_image_names = np.array(X_train), np.array(y_train_coarse), np.array(train_image_names)
    X_test, y_test_coarse, test_image_names = np.array(X_test), np.array(y_test_coarse), np.array(test_image_names)
    
    return [X_train.reshape(-1, 32, 32, 3), y_train_coarse, train_image_names, X_test.reshape(-1, 32, 32, 3), y_test_coarse, test_image_names]
    
if __name__=="__main__":
    [train_data, train_labels, train_filenames, test_data, test_labels, test_filenames] = load_cifar_100_data()
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    increment = 1000
    print("Working on a training set ...........")
    for i in range(int(len(train_data)/increment)):
        print("Extracting features for the lot", i)
        x = train_data[i*increment: (i+1)*increment]
        x_new = np.array([cv2.resize(x[j], (224, 224)) for j in range(0, len(x))]).astype('float32')
        vgg_train_input = preprocess_input(x_new)
        train_features = model.predict(vgg_train_input)
        np.save("raw_data/temp/train_processed_lot_"+str(i)+".npy", train_features)
        
    print("Working in a testing set ..............")
    for i in range(int(len(test_data)/increment)):
        print("Extracting features for the lot", i)
        x = test_data[i*increment: (i+1)*increment]
        x_new = np.array([cv2.resize(x[j], (224, 224)) for j in range(0, len(x))]).astype('float32')
        vgg_test_input = preprocess_input(x_new)
        test_features = model.predict(vgg_test_input)
        np.save("raw_data/temp/test_processed_lot_"+str(i)+".npy", test_features)
        
    np.save("raw_data/temp/train_file_names.npy", train_filenames)
    np.save("raw_data/temp/test_file_names.npy", test_filenames)
    np.save("raw_data/temp/train_labels.npy", train_labels)
    np.save("raw_data/temp/test_labels.npy", test_labels)
