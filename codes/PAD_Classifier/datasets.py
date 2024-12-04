import pandas as pd
import numpy as np
import glob
import os
from PIL import Image
import torch
import torchvision
from sklearn.model_selection import train_test_split

class WarsawCroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.dataset = self.create_image_dataframe()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_height, self.image_width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index]))
        return (
            self.preprocess(img.convert('L')), self.dataset["image_label"][index], self.dataset["image_file_name"][index]
            )
    
    def create_image_dataframe(self):
        image_dict = {"train_test_split":[],"image_label":[],"image_file_name":[],"image_file_path":[],"image_type":[]}
        for (root, subdirs, files) in os.walk(self.image_path):
            for subdir in subdirs:
                img_path = os.path.join(root, subdir)
                if subdir == 'train': 
                    split = 'Train'
                elif subdir == 'test-known':
                    split = 'Test'
                elif subdir == 'test-unknown':
                    split = 'Test-unknown'         
                img_files = glob.glob(img_path + self.file_type)
                for file in img_files:
                    file_name_splits = os.path.basename(file).split('_')
                    if file_name_splits[1] == 'REAL': label = 0
                    elif file_name_splits[1] == 'PRNT': label = 1
                    image_dict["train_test_split"].append(split)
                    image_dict["image_label"].append(label)
                    image_dict["image_file_name"].append(os.path.basename(file))
                    image_dict["image_file_path"].append(img_path)
                    image_dict["image_type"].append(file_name_splits[1])
        image_df = pd.DataFrame(image_dict)
        return image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image
    
class ClarksonCroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.dataset = self.create_image_dataframe()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_height, self.image_width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index]))
        return (
            self.preprocess(img.convert('L')), self.dataset["image_label"][index], self.dataset["image_file_name"][index]
            )
    
    def create_image_dataframe(self):
        image_dict = {"train_test_split":[],"image_label":[],"image_file_name":[],"image_file_path":[],"image_type":[]}
        for (root, subdirs, files) in os.walk(self.image_path):
            for subdir in subdirs:
                for (root, subsubdirs, files) in os.walk(os.path.join(self.image_path,subdir)):
                    #img_path = os.path.join(root, subdir)
                    for subsubdir in subsubdirs:
                        if subsubdir == 'Live': label = 0
                        else: label = 1
                        img_path = os.path.join(root, subsubdir)
                        img_files = glob.glob(img_path + self.file_type)
                        for file in img_files:
                            image_dict["train_test_split"].append(subdir)
                            image_dict["image_label"].append(label)
                            image_dict["image_file_name"].append(os.path.basename(file))
                            image_dict["image_file_path"].append(img_path)
                            image_dict["image_type"].append(subsubdir)
        image_df = pd.DataFrame(image_dict)
        return image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image


class NotreDameCroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.dataset = self.create_image_dataframe()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_height, self.image_width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index]))
        return (
            self.preprocess(img.convert('L')), self.dataset["image_label"][index], self.dataset["image_file_name"][index]
            )
    
    def create_image_dataframe(self):
        image_dict = {"train_test_split":[],"image_label":[],"image_file_name":[],"image_file_path":[],"image_type":[]}
        for (root, subdirs, files) in os.walk(self.image_path):
            for subdir in subdirs:
                for (root, subsubdirs, files) in os.walk(os.path.join(self.image_path,subdir)):
                    #img_path = os.path.join(root, subdir)
                    for subsubdir in subsubdirs:
                        if subsubdir == 'live': label = 0
                        else: label = 1
                        img_path = os.path.join(root, subsubdir)
                        img_files = glob.glob(img_path + self.file_type)
                        for file in img_files:
                            image_dict["train_test_split"].append(subdir)
                            image_dict["image_label"].append(label)
                            image_dict["image_file_name"].append(os.path.basename(file))
                            image_dict["image_file_path"].append(img_path)
                            image_dict["image_type"].append(subsubdir)
        image_df = pd.DataFrame(image_dict)
        return image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image
    
class IIITDWVUCroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.dataset = self.create_image_dataframe()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_height, self.image_width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index]))
        return (
            self.preprocess(img.convert('L')), self.dataset["image_label"][index], self.dataset["image_file_name"][index]
            )
    
    def create_image_dataframe(self):
        image_dict = {"train_test_split":[],"image_label":[],"image_file_name":[],"image_file_path":[],"image_type":[]}
        for (root, subdirs, files) in os.walk(self.image_path):
            for subdir in subdirs:
                for (root, subsubdirs, files) in os.walk(os.path.join(self.image_path,subdir)):
                    #img_path = os.path.join(root, subdir)
                    for subsubdir in subsubdirs:
                        if subsubdir == 'Live': label = 0
                        else: label = 1
                        img_path = os.path.join(root, subsubdir)
                        img_files = glob.glob(img_path + self.file_type)
                        for file in img_files:
                                image_dict["train_test_split"].append(subdir)
                                image_dict["image_label"].append(label)
                                image_dict["image_file_name"].append(os.path.basename(file))
                                image_dict["image_file_path"].append(img_path)
                                image_dict["image_type"].append(subsubdir)
        image_df = pd.DataFrame(image_dict)
        return image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image

class LivDet2020CroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.dataset = self.create_image_dataframe()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_height, self.image_width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index]))
        return (
            self.preprocess(img.convert('L')), self.dataset["image_label"][index], self.dataset["image_file_name"][index]
            )

    def create_image_dataframe(self):
        temp_df = pd.read_csv(os.path.join(self.image_path, "Test_Segmentation_Segmented.csv"))
        temp_df['image_file_name'] = temp_df["ImgPath"].str.split("/", expand=True)[3].astype(str)
        image_df = pd.DataFrame(columns=["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        image_df['image_file_path'] = temp_df['ImgPath'].apply(lambda x: os.path.join(self.image_path, x.split('/')[-2]))
        image_df['image_file_name'] = temp_df['ImgPath'].apply(lambda x: x.split('/')[-1])
        image_df["train_test_split"] = "Test"
        image_df["image_label"] = temp_df['Label'].apply(lambda x:  0 if x == 'Live' else 1)
        return image_df

    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image
