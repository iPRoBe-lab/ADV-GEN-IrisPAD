import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
import pandas as pd
from  datasets import WarsawCroppedIrisImages, IIITDWVUCroppedIrisImages, ClarksonCroppedIrisImages, NotreDameCroppedIrisImages, LivDet2020CroppedIrisImages
from utils import make_folder
from evaluation import Evaluation


def get_evaluation(args, loader, device, split):
    if args.pretrained_model == 'densenet121':
            model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    elif args.pretrained_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
    elif args.pretrained_model == 'resnet101':
            model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
    elif args.pretrained_model == 'vgg19':
            model = models.vgg19(weights='VGG19_Weights.DEFAULT')
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
        
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.modelPath)['model_state_dict'])
    model = model.to(device)
    model.eval()

    imagesScores=[]
    imageNames = []
    trueLabels = []
    for images, labels, imageName in loader:
            
        images = images.repeat(1,3,1,1) # for NIR images having one channel
        #tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
        images = images.to(device)

        # Output from single binary CNN model
        output = model(images).squeeze()
        predict_score = torch.sigmoid(output.detach()).cpu().numpy()

        # Normalization of output score between [0,1]
        #PAScore = np.minimum(np.maximum((PAScore+15)/35,0),1)
        #PAScore = np.round(np.minimum(np.maximum((PAScore+34)/48,0),1), decimals=2)
        imagesScores.extend(predict_score)
        imageNames.extend(imageName)
        trueLabels.extend(labels)
    evaluation = Evaluation(imageNames, trueLabels, imagesScores, args.output_path, split+"_thresh_"+str(args.threshold))
    evaluation.calculate_ACER(np.float64(args.threshold))  

def main(args):
    if args.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    make_folder(args.output_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    make_folder(args.output_path, args.version)
    args.output_path = os.path.join(args.output_path, args.version)

    args.modelPath = os.path.join(args.modelPath, args.version, "models", "iris_pa_densenet121_best.pth")
    
    # Load data
    print("Start loading data...")
    
    if args.dataset == 'Clarkson':
        args.image_path = os.path.join(args.image_path, args.dataset)
        iris_data = ClarksonCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.JPG')
    elif args.dataset == 'Warsaw':
        args.image_path = os.path.join(args.image_path, args.dataset)
        iris_data = WarsawCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.png')
    elif args.dataset == 'NotreDame':
        args.image_path = os.path.join(args.image_path, args.dataset)
        iris_data = NotreDameCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.tiff')
    elif args.dataset == 'IIITD_WVU':
        args.image_path = os.path.join(args.image_path, args.dataset)
        iris_data = IIITDWVUCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.bmp')
    elif args.dataset == 'LivDet2020':
        iris_data = LivDet2020CroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.png')
    
    print("Length of Dataset:" + str(len(iris_data)))
    
    # Create train and test loader
    train_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Train'].index)
    test_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Test'].index)
    train_data = torch.utils.data.Subset(iris_data, train_indices)
    test_data = torch.utils.data.Subset(iris_data, test_indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle=False, num_workers=1)
    print("Length of Train Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(train_data), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 1)])))
    #print(iris_data.dataset.head(10))
    print("Length of Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(test_data), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 1)])))
    get_evaluation(args, train_loader, device, split="Train")
    get_evaluation(args, test_loader, device, split="Test")
    
    if len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown')])>0:
        print("Length of Unknown Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown') & (iris_data.dataset["image_label"] == 1)])))
        unk_test_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Test-unknown'].index)
        unk_test_data = torch.utils.data.Subset(iris_data, unk_test_indices)
        unk_test_loader = torch.utils.data.DataLoader(unk_test_data, batch_size = args.batch_size, shuffle=False, num_workers=1)
        get_evaluation(args, unk_test_loader, device, split="U-Test")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # specify to enable cuda
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="0,7", help="GPU devices to be used")
    # image specification
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--file_type', type=str, default='/*.bmp')
    parser.add_argument('--image_path', default= '/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/data/livdet-iris-2017-complete-cropped',type=str)
    #parser.add_argument('--image_path', default= '/research/iprobe/datastore/datasets/iris/livdet-iris/LivDet-Iris-2020/Segmented',type=str)
    
    parser.add_argument('--dataset', default= 'Clarkson',type=str, choices=['Warsaw', 'Clarkson', 'NotreDame', 'IIITD_WVU', 'LivDet2020'])
    parser.add_argument('--modelPath',  default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/DNetPAD_Modified/outputs',type=str)
    parser.add_argument('--output_path',  default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/DNetPAD_Modified/inference',type=str)
    parser.add_argument('--version',  default='Clarkson_resnet50_adv_v1_1_retest',type=str)
    parser.add_argument('--pretrained_model', type=str, default= 'resnet50', choices=['densenet121', 'resnet50','resnet101', 'vgg19'])
    #parser.add_argument('-resultFolder',  default='Clarkson',type=str)
    parser.add_argument('--threshold', default=-1)

    args = parser.parse_args()
    main(args)