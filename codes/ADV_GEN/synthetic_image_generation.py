import argparse
import os
import torch
import numpy as np
from autoencoder_v6 import ConvolutionalAutoencoder
from utils import make_folder, denorm, tensor_to_PIL
import torchvision
import torchvision.models as models
from torchvision.utils import save_image


class Tester():
    def __init__(self, test_data, test_loader, args, device):
        self.test_data = test_data
        self.test_loader = test_loader
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.num_channels =args.num_channels
        self.latent_dim = args.latent_dim
        self.device = device
        self.args = args
        self.save_model_path = args.save_model_path
        self.synthetic_img_path = args.synthetic_img_path
        self.trained_model = args.trained_model
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),        
        ])
        self.PAD_normalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
        self.define_augmentation_parameters()
        self.load_trained_model()
        self.load_Standard_PAD()

    def define_augmentation_parameters(self):
        #self.augmentation_param_range = {
        #    'translate_x':[-30,30],'translate_y':[-30,30], 
        #    'shear_x':[-10,10], 'shear_y':[-10,10], 
        #    'scale_factor':[0.5,1.5], 'rotation':[-30,30], 
        #    'sharpness_factor':[0.1,2,0], 'brightness_factor':[0.1,2,0], 'contrast_factor':[0.1,2,0], 'saturation_factor':[0.1,2,0], 
        #    'hue_factor':[-0.5,0.5], 'solarize':[0,255], 'posterize':[0,8]
        #}

        self.augmentation_params ={
            'translate_x': np.arange(-15.0,16.0, step=1),  # Number of pixels to shifted along x-axis
            'translate_y': np.arange(-15.0,16.0, step=1),  # Number of pixels to shifted along y-axis
            'shear_x': np.arange(-5.0,6.0, step=1),    # Shear angle along x-axis
            'shear_y': np.arange(-5.0,6.0, step=1),      # Shear angle along y-axis
            'rotation': np.arange(-10.0,11.0, step=1),     # Angle (degrees) to be rotated
            'solarize': np.arange(0.7,1.0, step=0.1), 
            #'posterize': np.arange(0.0,9.0, step=1),      # Does not work on tensor image normalized between (0,1).
            'scale_factor': np.arange(0.9,1.2, step=0.1),
            'sharpness_factor': np.arange(0.5,1.6, step=0.1),  # 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
            'brightness_factor': np.arange(0.5,1.6, step=0.1), # 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
            'contrast_factor': np.arange(0.5,1.6, step=0.1),   # 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2. of 2.
        }
        self.num_aug_params = len(list(self.augmentation_params.keys()))
        self.min_aug_params = torch.Tensor(list({key: min(value) for key, value in self.augmentation_params.items()}.values()))
        self.max_aug_params = torch.Tensor(list({key: max(value) for key, value in self.augmentation_params.items()}.values()))
        
    def get_augentation_params_input(self, batch_size): # Randomly sample the augmentation parameters for training and testing
        augmentation_params_input = []
        for _, key in enumerate(self.augmentation_params):
            #np.random.seed(self.args.seed)
            values = np.round(np.random.choice(self.augmentation_params[key], batch_size), decimals=1)
            augmentation_params_input.append(values)
        return torch.Tensor(np.array(augmentation_params_input).T)
    
    def get_scaled_aug_params(self, augmentation_params_input):
        return ((augmentation_params_input-self.min_aug_params)/(self.max_aug_params - self.min_aug_params))

    def apply_augmentation(self, image, aug_param):
        aug_keys = list(self.augmentation_params.keys())
        #transformed_image = image.detach().clone()
        transformed_image = torchvision.transforms.functional.affine(image, angle=aug_param[aug_keys.index('rotation')].item(),
                                                                 translate=(aug_param[aug_keys.index('translate_x')], aug_param[aug_keys.index('translate_y')]),
                                                                 shear=(aug_param[aug_keys.index('shear_x')], aug_param[aug_keys.index('shear_y')]), 
                                                                 scale=aug_param[aug_keys.index('scale_factor')])
        #transformed_image = torchvision.transforms.functional.rotate(transformed_image, angle=aug_param[aug_keys.index('rotation')].item())
        transformed_image = torchvision.transforms.functional.solarize(transformed_image, aug_param[aug_keys.index('solarize')])
        #transformed_image = torchvision.transforms.functional.posterize(transformed_image, int(aug_param[aug_keys.index('posterize')].item()))
        transformed_image = torchvision.transforms.functional.adjust_sharpness(transformed_image, aug_param[aug_keys.index('sharpness_factor')])
        transformed_image = torchvision.transforms.functional.adjust_brightness(transformed_image, aug_param[aug_keys.index('brightness_factor')])
        transformed_image = torchvision.transforms.functional.adjust_contrast(transformed_image, aug_param[aug_keys.index('contrast_factor')])
        return transformed_image

    def load_trained_model(self):
        self.model = ConvolutionalAutoencoder(self.image_height, self.image_width, self.num_channels, self.num_aug_params, self.latent_dim).to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(os.path.join(self.save_model_path, self.trained_model)))
        self.model.eval()
    
    
    def load_Standard_PAD(self):
        if self.args.pretrained_model == 'densenet121':
            PAD = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            num_features = PAD.classifier.in_features
            PAD.classifier = torch.nn.Linear(num_features, 1)
        elif self.args.pretrained_model == 'resnet50':
            PAD = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            PAD.fc = torch.nn.Linear(PAD.fc.in_features, 1)
        elif self.args.pretrained_model == 'vgg19':
            PAD = models.vgg19(weights='VGG19_Weights.DEFAULT')
            PAD.classifier[6] = torch.nn.Linear(PAD.classifier[6].in_features, 1)

        self.PAD = torch.nn.DataParallel(PAD)
        self.PAD.load_state_dict(torch.load(self.args.standard_PAD_model)['model_state_dict'])
        self.PAD = PAD.to(self.device) 
        self.PAD.eval()
        
    def get_PAD_output(self, images):
        images = self.PAD_normalize(images).repeat(1,3,1,1)
        PAD_output = torch.sigmoid(self.PAD(images).squeeze())
        return PAD_output
    
    def generate_synthetic_image(self):
        make_folder(self.synthetic_img_path, "Transformed_Images_WO_Training")
        make_folder(self.synthetic_img_path, "Synthetic_Images")
        step=0
        l1_norm = []
        l2_norm = []
        for data in self.test_loader:
            images, labels, image_file_name = data
            current_batch_size = images.shape[0]
            augmentation_params_input = (self.get_augentation_params_input(current_batch_size))
            torch.save(augmentation_params_input, os.path.join(self.args.synthetic_img_path, 'augmentation_params_input.pt'))
            labels = labels.type(torch.FloatTensor).to(self.device)
                    
            transformed_images = torchvision.transforms.Lambda(lambda images: torch.stack(
                         [self.apply_augmentation(images[i],augmentation_params_input[i]) for i in range(current_batch_size)]))(images)

            transformed_images_PA_score = self.get_PAD_output(transformed_images.to(self.device)).detach().cpu()
            transformed_images_label = torch.where(transformed_images_PA_score>=np.float64(self.args.threshold), 1, 0)
            
            images = self.normalize(images)    
            images = images.to(self.device)
            aug_params_input_scaled = self.get_scaled_aug_params(augmentation_params_input).to(self.device)
                
            output = self.model(images, aug_params_input_scaled)
            
            synthetic_images_PA_score = self.get_PAD_output(denorm(output)).detach().cpu()
            synthetic_images_label = torch.where(synthetic_images_PA_score>=np.float64(args.threshold), 1, 0)
            for i in range(0, current_batch_size):
                transformed_img = transformed_images[i]
                synthetic_img = denorm(output[i].to("cpu"))
                l1_norm.append(torch.nn.functional.l1_loss(synthetic_img, transformed_img, reduction = 'mean').item())
                l2_norm.append(torch.nn.functional.mse_loss(synthetic_img, transformed_img, reduction = 'mean').item())
                
                save_image(transformed_img, 
                       os.path.join(self.synthetic_img_path, "Transformed_Images_WO_Training", 'transformed-{}-GT{}-{}-{}-{:.2f}.jpg'.format(self.args.batch_size*step+i,labels[i].int(),
                                                                                                                                    image_file_name[i], 
                                                                                                                                    transformed_images_label[i].int(),
                                                                                                                                    transformed_images_PA_score[i])))
                save_image(synthetic_img, 
                       os.path.join(self.synthetic_img_path, "Synthetic_Images", 'synthetic-{}-GT{}-{}-{}-{:.2f}.jpg'.format(self.args.batch_size*step+i, labels[i].int(),
                                                                                                                    image_file_name[i], 
                                                                                                                    synthetic_images_label[i].int(), 
                                                                                                                    synthetic_images_PA_score[i])))

                #ssim_score = calculate_ssim_score(np.array(tensor_to_PIL(transformed_img)), np.array(tensor_to_PIL(synthetic_img)))
                #ssim_score_list.append(ssim_score)
            step=step+1      
        np.save(os.path.join(self.synthetic_img_path, "l2_norm.npy"), l2_norm)
        np.save(os.path.join(self.synthetic_img_path, "l1_norm.npy"), l1_norm)     

def main(args):
    # For fast training
    if args.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Load data
    print("Start loading data...")
    args.image_path = os.path.join(args.image_path, args.dataset)
    
    if args.dataset == 'Clarkson':
        from datasets import ClarksonCroppedIrisImages
        iris_data = ClarksonCroppedIrisImages(args.image_path, args.image_height, args.image_width, args.file_type)
    elif args.dataset == 'Warsaw':
        from datasets import WarsawCroppedIrisImages
        iris_data = WarsawCroppedIrisImages(args.image_path, args.image_height, args.image_width, args.file_type)
    elif args.dataset == 'IIITD_WVU':
        from datasets import IIITDWVUCroppedIrisImages
        iris_data = IIITDWVUCroppedIrisImages(args.image_path, args.image_height, args.image_width, args.file_type)
    elif args.dataset == 'NotreDame':
        from datasets import NotreDameCroppedIrisImages
        iris_data = NotreDameCroppedIrisImages(args.image_path, args.image_height, args.image_width, args.file_type)
    print("Length of Dataset:" + str(len(iris_data)))
    
    # Create train and test loader
    train_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Train'].index)
    test_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Test'].index)
    train_data = torch.utils.data.Subset(iris_data, train_indices)
    test_data = torch.utils.data.Subset(iris_data, test_indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Length of Train Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(train_data), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 1)])))

    print("Length of Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(test_data), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 1)])))


    # Retrieve current working directory
    curr_wd = os.path.dirname(__file__)
    args.save_model_path = os.path.join(curr_wd, args.save_model_path, args.version)
    make_folder(os.path.join(curr_wd, args.synthetic_img_path), args.version)
    args.synthetic_img_path = os.path.join(curr_wd, args.synthetic_img_path, args.version)
    make_folder(args.synthetic_img_path, args.trained_model, replace=True)
    args.synthetic_img_path = os.path.join(args.synthetic_img_path, args.trained_model)
    
    print("Generating images from trained models...")
    tester = Tester(iris_data, train_loader, args, device)
    tester.generate_synthetic_image()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating synthetic images with trained ADV_GEN model")
    # specify to enable cuda
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="1,2", help="GPU devices to be used")
    
    # image specification
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--file_type', type=str, default='/*.JPG', help="Image format of Dataset Used")
    
    # path details
    parser.add_argument('--image_path', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/data/livdet-iris-2017-complete-cropped', help="Path to cropped images used for training and testing")
    parser.add_argument('--dataset', type=str, default='Clarkson', choices=['Clarkson', 'Warsaw', 'IIITD_WVU', 'NotreDame'], help="Dataset to be used")
    parser.add_argument('--pretrained_model', type=str, default= 'densenet121', choices=['densenet121', 'resnet50', 'vgg19'])
    parser.add_argument('--standard_PAD_model', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/DNetPAD_Modified/outputs/Clarkson_densenet121_v1/models/iris_pa_densenet121_best.pth')
    parser.add_argument('--save_model_path', type=str, default='models')
    parser.add_argument('--synthetic_img_path', type=str, default='synthetic_train_images', help="path to store generated synthetic imaages")
#    parser.add_argument('--synthetic_img_path', type=str, default='synthetic_train_images')
    

    # details of training paramaters
    parser.add_argument('--num_classes', type=int, default=2, help="Live and Spoof")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=512)


    # Other parameters
    parser.add_argument('--version', type=str, default= 'Clarkson_conv_AE_v6_1_densenet121')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default= '500_conv_AE.pth', help="ADV_GEN model to be used to generate synthetic images")
    parser.add_argument('--threshold', type=float, default= 0.5)
    args = parser.parse_args()
    main(args)