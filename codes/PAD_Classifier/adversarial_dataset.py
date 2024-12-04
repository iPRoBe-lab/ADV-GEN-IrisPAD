import pandas as pd
import numpy as np
import glob
import os
from PIL import Image
import torch
import torchvision
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import torchvision.models as models

transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])

def get_features(images, device, pretrained_model):
    if pretrained_model == 'densenet121':
            model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
            model = model.to(device)
            model = torch.nn.DataParallel(model)
            features = model.module.features(images.to(device)).squeeze(0)
    elif pretrained_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            modules = list(model.children())[:-1]
            resnet_features = torch.nn.Sequential(*modules)
            resnet_features = resnet_features.to(device)
            resnet_features = torch.nn.DataParallel(resnet_features)
            features = resnet_features(images.to(device)).squeeze(0)
    elif pretrained_model == 'vgg19':
            model = models.vgg19(weights='VGG19_Weights.DEFAULT')
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
            model = model.to(device)
            model = torch.nn.DataParallel(model)
            features = model.module.features(images.to(device)).squeeze(0)
    return features.detach().cpu().numpy().reshape(-1)

def k_means_clustering(image_files, image_path, device, n_clusters = 10, n_images_per_cluster=20, pretrained_model='densenet121'):
    print("Model used for K-means Clustering:"+pretrained_model)
    adversarial_image_features = []
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        image = Image.open(img_path).convert('L')
        image = transform(image).repeat(3, 1, 1).unsqueeze(0)
        feature_array = get_features(image, device, pretrained_model) 
        adversarial_image_features.append(feature_array)
    #adversarial_images = torch.stack(adversarial_images, dim=0)
    #adversarial_image_features = get_features(adversarial_images, device)
    adversarial_image_features = np.array(adversarial_image_features)
    #print("Shape of Adv. Image Features:" + str(adversarial_image_features.shape))
    # Flatten the images to 1D arrays
    #adversarial_image_features = adversarial_image_features.reshape(adversarial_image_features.shape[0], -1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(adversarial_image_features)

    # Dictionary to hold the indices of selected images
    selected_indices = []
    # Select closest images from each cluster
    for cluster_idx in range(n_clusters):
        # Get indices of all images in this cluster
        cluster_indices = np.where(kmeans.labels_ == cluster_idx)[0]
        # Get the corresponding images
        cluster_images = adversarial_image_features[cluster_indices]
        # Compute distances to the cluster centroid
        centroid = kmeans.cluster_centers_[cluster_idx].reshape(1, -1)
        distances = np.linalg.norm(cluster_images - centroid, axis=1)
        # Get indices of the 20 closest images
        closest_indices = cluster_indices[np.argsort(distances)[:n_images_per_cluster]]
        # Add to the selected indices
        selected_indices.extend(closest_indices)

    # Convert selected indices to a numpy array
    selected_indices = np.array(selected_indices)
    return selected_indices

class WarsawCroppedIrisImagesAdversarial(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type, adv_image_path, adv_file_type, device, transform=True):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.adv_image_path = adv_image_path
        self.adv_file_type = adv_file_type
        self.device = device
        self.orig_dataset = self.create_image_dataframe()
        if transform:
            print("Using Adversarial Transformation..")
            self.adversarial_dataset = self.get_adversarial_images_w_transform()
        
        self.dataset = pd.concat([self.orig_dataset, self.adversarial_dataset], ignore_index=True)
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
                    split = 'Test-Unknown'         
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
    
    def get_adversarial_images_w_transform(self):
        #adversarial_image_df = pd.DataFrame(columns=["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        transformed_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Transformed_Images_WO_Training") + self.adv_file_type)]
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Synthetic_Images") + self.adv_file_type)]
        data = pd.DataFrame(transformed_images, columns=["transformed_img_file_name"])
        data['transformed_image_index'] = data["transformed_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["transformed_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['transformed_PA_label'] = data["transformed_img_file_name"].str[-10:-9].astype(int)
        data['transformed_PA_score'] = data["transformed_img_file_name"].str[-8:-4].astype(float)
        synthetic_image_df = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        synthetic_image_df['synthetic_image_index'] = synthetic_image_df["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        synthetic_image_df['synthetic_PA_label'] = synthetic_image_df["synthetic_img_file_name"].str[-10:-9].astype(int)
        synthetic_image_df['synthetic_PA_score'] = synthetic_image_df["synthetic_img_file_name"].str[-8:-4].astype(float)
        data = pd.merge(left = data, right = synthetic_image_df, how='inner', 
                   left_on=data["transformed_image_index"], right_on=synthetic_image_df["synthetic_image_index"])
        data.sort_values(by=['transformed_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data["transformed_image_path"] = os.path.join(self.adv_image_path, "Transformed_Images_WO_Training")
        data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        
        # Select transformed adversarial PA
        adversarial_PA_transformed = data[(data["image_label"]==1) & (data["transformed_PA_score"]<0.5)
                                          ].reset_index()
        adversarial_PA_transformed["image_type"] = "Tranformed_Adversarial_PA"
        adversarial_PA_transformed = pd.DataFrame(adversarial_PA_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) #& (data["transformed_PA_score"]>=0.1) 
                              & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # select transformed adversarial bonafide
        adversarial_bonafide_transformed = data[(data["image_label"]==0) & (data["transformed_PA_score"]>=0.5)
                                                ].reset_index()
        adversarial_bonafide_transformed['image_type'] = "Tranformed_Adversarial_Bonafide"
        #k_means_selected_bt = k_means_clustering(adversarial_bonafide_transformed["transformed_img_file_name"], os.path.join(self.adv_image_path, "Transformed_Images_WO_Training"),
        #                                      device =self.device, n_clusters=5, n_images_per_cluster=8)
        #adversarial_bonafide_transformed = adversarial_bonafide_transformed.filter(items=k_means_selected_bt, axis=0)

        adversarial_bonafide_transformed = pd.DataFrame(adversarial_bonafide_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select adversarial bonafide
        adversarial_bonafide = data[(data["image_label"]==0) #& (data["transformed_PA_score"]<=0.9) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([#adversarial_image_df, 
        #                                  adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image


class ClarksonCroppedIrisImagesAdversarial(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type, adv_image_path, adv_file_type, device, transform=False, pretrained_mdoel = 'densenet121'):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.adv_image_path = adv_image_path
        self.adv_file_type = adv_file_type
        self.device = device
        self.pretrained_model = pretrained_mdoel
        self.orig_dataset = self.create_image_dataframe()
        if transform:
            print("Using Adversarial Transformation..")
            self.adversarial_dataset = self.get_adversarial_images_w_transform()
        else:
            print("Using Adversarial Generation..")
            self.adversarial_dataset = self.get_adversarial_images_wo_transform()
        self.dataset = pd.concat([self.orig_dataset, self.adversarial_dataset], ignore_index=True)
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
    
    def get_adversarial_images_w_transform(self):
        #adversarial_image_df = pd.DataFrame(columns=["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        transformed_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Transformed_Images_WO_Training") + self.adv_file_type)]
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Synthetic_Images") + self.adv_file_type)]
        data = pd.DataFrame(transformed_images, columns=["transformed_img_file_name"])
        data['transformed_image_index'] = data["transformed_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["transformed_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['transformed_PA_label'] = data["transformed_img_file_name"].str[-10:-9].astype(int)
        data['transformed_PA_score'] = data["transformed_img_file_name"].str[-8:-4].astype(float)
        synthetic_image_df = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        synthetic_image_df['synthetic_image_index'] = synthetic_image_df["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        synthetic_image_df['synthetic_PA_label'] = synthetic_image_df["synthetic_img_file_name"].str[-10:-9].astype(int)
        synthetic_image_df['synthetic_PA_score'] = synthetic_image_df["synthetic_img_file_name"].str[-8:-4].astype(float)
        data = pd.merge(left = data, right = synthetic_image_df, how='inner', 
                   left_on=data["transformed_image_index"], right_on=synthetic_image_df["synthetic_image_index"])
        data.sort_values(by=['transformed_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data["transformed_image_path"] = os.path.join(self.adv_image_path, "Transformed_Images_WO_Training")
        data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        
        # Select transformed adversarial PA
        adversarial_PA_transformed = data[(data["image_label"]==1) & (data["transformed_PA_score"]<0.5)
                                          ].reset_index()
        adversarial_PA_transformed["image_type"] = "Tranformed_Adversarial_PA"
        adversarial_PA_transformed = pd.DataFrame(adversarial_PA_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) #& (data["transformed_PA_score"]>=0.5) 
                              & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # select transformed adversarial bonafide
        adversarial_bonafide_transformed = data[(data["image_label"]==0) & (data["transformed_PA_score"]>=0.5)
                                                ].reset_index()
        adversarial_bonafide_transformed['image_type'] = "Tranformed_Adversarial_Bonafide"
        k_means_selected_bt = k_means_clustering(adversarial_bonafide_transformed["transformed_img_file_name"], os.path.join(self.adv_image_path, "Transformed_Images_WO_Training"),
                                              device =self.device, n_clusters=5, n_images_per_cluster=10, pretrained_model=self.pretrained_model)
        adversarial_bonafide_transformed = adversarial_bonafide_transformed.filter(items=k_means_selected_bt, axis=0)

        adversarial_bonafide_transformed = pd.DataFrame(adversarial_bonafide_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select adversarial bonafide
        adversarial_bonafide = data[(data["image_label"]==0) #& (data["transformed_PA_score"]<0.5) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([adversarial_image_df, 
        #                                 adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df
    
    def get_adversarial_images_wo_transform(self):
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path) + self.adv_file_type)]
        data = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        data['synthetic_image_index'] = data["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["synthetic_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['synthetic_PA_score'] = data["synthetic_img_file_name"].str[-8:-4].astype(float)
        data.sort_values(by=['synthetic_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data['image_type'] = data['image_label'].apply(lambda x: 'Live' if x == 0 else 'Spoof')
        #data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        data["synthetic_image_path"] = self.adv_image_path

        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], self.adv_image_path, #os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])

        
        adversarial_bonafide = data[(data["image_label"]==0) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], self.adv_image_path,# os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([adversarial_image_df, adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image


class IIITDWVUCroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type, adv_image_path, adv_file_type, device, transform=False):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.adv_image_path = adv_image_path
        self.adv_file_type = adv_file_type
        self.device = device
        self.orig_dataset = self.create_image_dataframe()
        if transform:
            print("Using Adversarial Transformation..")
            self.adversarial_dataset = self.get_adversarial_images_w_transform()
        else:
            print("Using Adversarial Generation..")
            self.adversarial_dataset = self.get_adversarial_images_wo_transform()
        self.dataset = pd.concat([self.orig_dataset, self.adversarial_dataset], ignore_index=True)
        
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
    
    def get_adversarial_images_w_transform(self):
        #adversarial_image_df = pd.DataFrame(columns=["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        transformed_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Transformed_Images_WO_Training") + self.adv_file_type)]
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Synthetic_Images") + self.adv_file_type)]
        data = pd.DataFrame(transformed_images, columns=["transformed_img_file_name"])
        data['transformed_image_index'] = data["transformed_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["transformed_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['transformed_PA_label'] = data["transformed_img_file_name"].str[-10:-9].astype(int)
        data['transformed_PA_score'] = data["transformed_img_file_name"].str[-8:-4].astype(float)
        synthetic_image_df = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        synthetic_image_df['synthetic_image_index'] = synthetic_image_df["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        synthetic_image_df['synthetic_PA_label'] = synthetic_image_df["synthetic_img_file_name"].str[-10:-9].astype(int)
        synthetic_image_df['synthetic_PA_score'] = synthetic_image_df["synthetic_img_file_name"].str[-8:-4].astype(float)
        data = pd.merge(left = data, right = synthetic_image_df, how='inner', 
                   left_on=data["transformed_image_index"], right_on=synthetic_image_df["synthetic_image_index"])
        data.sort_values(by=['transformed_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data["transformed_image_path"] = os.path.join(self.adv_image_path, "Transformed_Images_WO_Training")
        data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        
        # Select transformed adversarial PA
        adversarial_PA_transformed = data[(data["image_label"]==1) & (data["transformed_PA_score"]<0.5)
                                          ].reset_index()
        adversarial_PA_transformed["image_type"] = "Tranformed_Adversarial_PA"
        adversarial_PA_transformed = pd.DataFrame(adversarial_PA_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) #& (data["transformed_PA_score"]>=0.1) 
                              & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # select transformed adversarial bonafide
        adversarial_bonafide_transformed = data[(data["image_label"]==0) & (data["transformed_PA_score"]>=0.5)
                                                ].reset_index()
        adversarial_bonafide_transformed['image_type'] = "Tranformed_Adversarial_Bonafide"
        #k_means_selected_bt = k_means_clustering(adversarial_bonafide_transformed["transformed_img_file_name"], os.path.join(self.adv_image_path, "Transformed_Images_WO_Training"),
        #                                      device =self.device, n_clusters=5, n_images_per_cluster=10)
        #adversarial_bonafide_transformed = adversarial_bonafide_transformed.filter(items=k_means_selected_bt, axis=0)

        adversarial_bonafide_transformed = pd.DataFrame(adversarial_bonafide_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select adversarial bonafide
        adversarial_bonafide = data[(data["image_label"]==0) #& (data["transformed_PA_score"]<=0.9) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([#adversarial_image_df, 
        #                                  adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df
    
    def get_adversarial_images_wo_transform(self):
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path) + self.adv_file_type)]
        data = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        data['synthetic_image_index'] = data["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["synthetic_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['synthetic_PA_score'] = data["synthetic_img_file_name"].str[-8:-4].astype(float)
        data.sort_values(by=['synthetic_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data['image_type'] = data['image_label'].apply(lambda x: 'Live' if x == 0 else 'Spoof')
        #data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        data["synthetic_image_path"] = self.adv_image_path

        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], self.adv_image_path, #os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])

        
        adversarial_bonafide = data[(data["image_label"]==0) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], self.adv_image_path,# os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=10, n_images_per_cluster=20, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([adversarial_image_df, adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df
    
    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image
    
class NotreDameCroppedIrisImages(torch.utils.data.Dataset):
    def __init__(self, image_path, image_height, image_width, file_type, adv_image_path, adv_file_type, device, transform=True):
        super().__init__()
        self.image_path = os.path.join(image_path)
        self.image_height = image_height
        self.image_width = image_width
        self.file_type = file_type
        self.adv_image_path = adv_image_path
        self.adv_file_type = adv_file_type
        self.device = device
        self.orig_dataset = self.create_image_dataframe()
        if transform:
            print("Using Adversarial Transformation..")
            self.adversarial_dataset = self.get_adversarial_images_w_transform()
        else:
            print("Using Adversarial Generation..")
            self.adversarial_dataset = self.get_adversarial_images_wo_transform()
        
        self.dataset = pd.concat([self.orig_dataset, self.adversarial_dataset], ignore_index=True)
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
    
    def get_adversarial_images_w_transform(self):
        #adversarial_image_df = pd.DataFrame(columns=["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        transformed_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Transformed_Images_WO_Training") + self.adv_file_type)]
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path, "Synthetic_Images") + self.adv_file_type)]
        data = pd.DataFrame(transformed_images, columns=["transformed_img_file_name"])
        data['transformed_image_index'] = data["transformed_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["transformed_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['transformed_PA_label'] = data["transformed_img_file_name"].str[-10:-9].astype(int)
        data['transformed_PA_score'] = data["transformed_img_file_name"].str[-8:-4].astype(float)
        synthetic_image_df = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        synthetic_image_df['synthetic_image_index'] = synthetic_image_df["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        synthetic_image_df['synthetic_PA_label'] = synthetic_image_df["synthetic_img_file_name"].str[-10:-9].astype(int)
        synthetic_image_df['synthetic_PA_score'] = synthetic_image_df["synthetic_img_file_name"].str[-8:-4].astype(float)
        data = pd.merge(left = data, right = synthetic_image_df, how='inner', 
                   left_on=data["transformed_image_index"], right_on=synthetic_image_df["synthetic_image_index"])
        data.sort_values(by=['transformed_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data["transformed_image_path"] = os.path.join(self.adv_image_path, "Transformed_Images_WO_Training")
        data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        
        # Select transformed adversarial PA
        adversarial_PA_transformed = data[(data["image_label"]==1) & (data["transformed_PA_score"]<0.5)
                                          ].reset_index()
        adversarial_PA_transformed["image_type"] = "Tranformed_Adversarial_PA"
        adversarial_PA_transformed = pd.DataFrame(adversarial_PA_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) #& (data["transformed_PA_score"]>=0.1) 
                              & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=5, n_images_per_cluster=25, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # select transformed adversarial bonafide
        adversarial_bonafide_transformed = data[(data["image_label"]==0) & (data["transformed_PA_score"]>=0.5)
                                                ].reset_index()
        adversarial_bonafide_transformed['image_type'] = "Tranformed_Adversarial_Bonafide"
        #k_means_selected_bt = k_means_clustering(adversarial_bonafide_transformed["transformed_img_file_name"], os.path.join(self.adv_image_path, "Transformed_Images_WO_Training"),
        #                                      device =self.device, n_clusters=5, n_images_per_cluster=10)
        #adversarial_bonafide_transformed = adversarial_bonafide_transformed.filter(items=k_means_selected_bt, axis=0)

        adversarial_bonafide_transformed = pd.DataFrame(adversarial_bonafide_transformed[["train_test_split", "image_label", "transformed_img_file_name", "transformed_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        
        # Select adversarial bonafide
        adversarial_bonafide = data[(data["image_label"]==0) #& (data["transformed_PA_score"]<=0.9) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=5, n_images_per_cluster=25, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([#adversarial_image_df, 
        #                                  adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df
    
    def get_adversarial_images_wo_transform(self):
        synthetic_images = [os.path.basename(img) for img in glob.glob(os.path.join(self.adv_image_path) + self.adv_file_type)]
        data = pd.DataFrame(synthetic_images, columns=["synthetic_img_file_name"])
        data['synthetic_image_index'] = data["synthetic_img_file_name"].str.split("-", expand=True)[1].astype(int)
        data['image_label'] = data["synthetic_img_file_name"].str.split("-", expand=True)[2].str[-1].astype(int)
        data['synthetic_PA_score'] = data["synthetic_img_file_name"].str[-8:-4].astype(float)
        data.sort_values(by=['synthetic_image_index'], inplace=True, ignore_index=True)
        data["l2_norm"] = np.load(os.path.join(self.adv_image_path, "l2_norm.npy"))
        data["train_test_split"] = 'Train'
        data['image_type'] = data['image_label'].apply(lambda x: 'Live' if x == 0 else 'Spoof')
        #data["synthetic_image_path"] = os.path.join(self.adv_image_path, "Synthetic_Images")
        data["synthetic_image_path"] = self.adv_image_path

        
        # Select generated adversarial PA
        adversarial_PA = data[(data["image_label"]==1) & (data["synthetic_PA_score"]<0.1)]
        adversarial_PA = adversarial_PA[(adversarial_PA["l2_norm"]<0.01)].reset_index()
        adversarial_PA['image_type'] = "Synthetic_Adversarial_PA"
        k_means_selected_PA = k_means_clustering(adversarial_PA["synthetic_img_file_name"], self.adv_image_path, #os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=5, n_images_per_cluster=25, pretrained_model=self.pretrained_model)
        adversarial_PA = adversarial_PA.filter(items=k_means_selected_PA, axis=0)
        
        adversarial_PA = pd.DataFrame(adversarial_PA[["train_test_split", "image_label", "synthetic_img_file_name", "synthetic_image_path","image_type"]].values,
                                      columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])

        
        adversarial_bonafide = data[(data["image_label"]==0) 
                                    & (data["synthetic_PA_score"]>0.9)]
        adversarial_bonafide = adversarial_bonafide[(adversarial_bonafide["l2_norm"]<0.01)].reset_index()
        adversarial_bonafide['image_type'] = "Synthetic_Adversarial_Bonafide"
        k_means_selected_bonafide = k_means_clustering(adversarial_bonafide["synthetic_img_file_name"], self.adv_image_path,# os.path.join(self.adv_image_path, "Synthetic_Images"),
                                              device =self.device, n_clusters=5, n_images_per_cluster=25, pretrained_model=self.pretrained_model)
        adversarial_bonafide = adversarial_bonafide.filter(items=k_means_selected_bonafide, axis=0)
        
        adversarial_bonafide = pd.DataFrame(adversarial_bonafide[["train_test_split", "image_label", "synthetic_img_file_name", 
                                                                           "synthetic_image_path","image_type"]].values, 
                                                                           columns= ["train_test_split","image_label","image_file_name","image_file_path","image_type"])
        adversarial_image_df = pd.concat([adversarial_PA, adversarial_bonafide], ignore_index=True, axis=0)        
        #adversarial_image_df = pd.concat([adversarial_image_df, adversarial_PA_transformed, adversarial_bonafide_transformed], ignore_index=True, axis=0)
        return adversarial_image_df


    def get_PIL_image(self, index):
        image = Image.open(os.path.join(self.dataset["image_file_path"][index], self.dataset["image_file_name"][index])).convert('L')
        return image