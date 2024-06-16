import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class SaDataset(Dataset):
    def __init__(self, dir_to_img, file, transform, train=True):
        self.dir_to_img = dir_to_img
        self.dataset = pd.read_csv(file)
        self.transform = transform
        self.train = train
        if train:
            le = LabelEncoder()
            le.fit_transform(self.dataset.label)
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            self.class_to_idx = le_name_mapping            
                
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        if self.train:
            label = self.dataset.label[idx]
            img_name = os.path.join(self.dir_to_img, label, self.dataset.image_names[idx])
            img_name = self.dataset.image_path[idx]
            img = plt.imread(img_name)
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img)
            img_tr = self.transform(image = img)['image']
            return np.array(img_tr), self.class_to_idx[label]
        else:
            # img_name = os.path.join(self.dir_to_img, label, self.dataset.image_names[idx])
            img_name = self.dataset.image_path[idx]
            img = plt.imread(img_name)
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img)
            img_tr = self.transform(image = img)['image']
            return np.array(img_tr)

    
    def show_image(self, idx, label='Unknown'):
        if self.train:
            label =  self.dataset.label[idx]
            img_name = self.dataset.image_path[idx]
            img = plt.imread(img_name)
            # img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            plt.imshow(img)
            plt.title(label)
            plt.show()
        else:    
            img_name = os.path.join(self.dir_to_img, self.dataset.image_path[idx])
            img_name = self.dataset.image_path[idx]
            img = plt.imread(img_name)
            # img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            # img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            plt.imshow(img)
            plt.title(label)
            plt.show()     