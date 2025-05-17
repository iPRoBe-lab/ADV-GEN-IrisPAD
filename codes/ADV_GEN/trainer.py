import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import torch
from autoencoder import ConvolutionalAutoencoder
from torchvision.utils import save_image
import torchvision.models as models
from utils import tensor_to_PIL, denorm
import torchvision

class Trainer():
    def __init__(self, iris_data, train_loader, test_loader, args, device):
        
        self.iris_data = iris_data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.num_channels = args.num_channels
        self.latent_dim = args.latent_dim
        self.args = args
        self.device = device
        self.exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
        self.sampled_test_data = self.iris_data.dataset[self.iris_data.dataset["train_test_split"] == 'Test'].groupby("image_label").sample(n=self.args.num_samples, replace=False, 
                                                                        random_state=100).reset_index()
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),        
        ])
        self.PAD_normalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
        self.define_augmentation_parameters()
        self.build_autoencoder()
        self.load_standard_PAD()

    
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
            'contrast_factor': np.arange(0.5,1.6, step=0.1),   # 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
        }
        self.num_aug_params = len(list(self.augmentation_params.keys()))
        self.min_aug_params = torch.Tensor(list({key: min(value) for key, value in self.augmentation_params.items()}.values()))
        self.max_aug_params = torch.Tensor(list({key: max(value) for key, value in self.augmentation_params.items()}.values()))
        
    def get_augentation_params_input(self, batch_size): # Randomly sample the augmentation parameters for training and testing
        augmentation_params_input = []
        for _, key in enumerate(self.augmentation_params):
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

    def build_autoencoder(self):
        # Initialize weights
        def weights_init(module):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(module.weight)
                #module.bias.data.fill_(0.01)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                #module.bias.data.fill_(0.01)
            
        # Initialize the autoencoder
        self.model = ConvolutionalAutoencoder(self.image_height, self.image_width, self.num_channels, self.num_aug_params, self.latent_dim).to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model_criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, 
                                          betas=[self.args.beta1, self.args.beta2]
                                         )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=self.args.gamma)
        self.classifier_criterion = torch.nn.BCEWithLogitsLoss()

        if self.args.load_model is None:
            self.model.apply(weights_init)
        if self.args.load_model is not None:
            self.model.load_state_dict(torch.load(os.path.join(self.args.save_model_path, '{}_conv_AE.pth'.format(self.args.load_model))))
            self.optimizer.load_state_dict(torch.load(os.path.join(self.args.save_model_path, '{}_conv_AE_opt.pth'.format(self.args.load_model))))



    def load_standard_PAD(self):
        if self.args.pretrained_model == 'densenet121':
            PAD = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            num_features = PAD.classifier.in_features
            PAD.classifier = torch.nn.Linear(num_features, 1)
        elif self.args.pretrained_model == 'resnet101':
            PAD = models.resnet101(weights='ResNet101_Weights.DEFAULT')
            PAD.fc = torch.nn.Linear(PAD.fc.in_features, 1)
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
        for param in self.PAD.parameters():
            param.requires_grad = False

    def get_PAD_output(self, images):
        images = self.PAD_normalize(images).repeat(1,3,1,1)
        PAD_output = self.PAD(images).squeeze()
        return PAD_output
    

    def train(self):
        if self.args.load_model:
            start = self.args.load_model
        else:
            start = 0
        
        test_augmentation_params_input = (self.get_augentation_params_input(self.args.num_samples*self.args.num_classes))
        # Train the autoencoder
        for epoch in range(start, self.args.num_epochs):  
            if (epoch+1) % 100==1:
                print("\nLearning Rate:" + str(self.lr_scheduler.get_last_lr()))
            
            train_loss = []
            test_loss = []
            self.model.train()            
            for data in self.train_loader:
                images, labels, _ = data
                current_batch_size = images.shape[0]
                augmentation_params_input = (self.get_augentation_params_input(current_batch_size))
                
                # Apply augmentation paramaters to images to comapare with generated images
                transformed_images = torchvision.transforms.Lambda(lambda images: torch.stack(
                    [self.apply_augmentation(images[i],augmentation_params_input[i]) for i in range(current_batch_size)]))(images)
                
                images = self.normalize(images)
                transformed_images = self.normalize(transformed_images)
                
                images = images.to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)
                aug_params_input_scaled = self.get_scaled_aug_params(augmentation_params_input).to(self.device)
                transformed_images = transformed_images.to(self.device)
                
                output= self.model(images, aug_params_input_scaled)
                reconstruction_loss = self.model_criterion(output, transformed_images)
                
                PAD_output = self.get_PAD_output(denorm(output))
                classifier_loss = self.classifier_criterion(PAD_output, 1-labels)
                self.optimizer.zero_grad()
                total_loss = reconstruction_loss + self.args.lambda_adv*classifier_loss
                total_loss.backward()
                self.optimizer.step()
                train_loss.append(total_loss.item())
            self.lr_scheduler.step()
            # After completion of an epoch, check model performance on test data
            self.model.eval()
            with torch.no_grad():
                bonafide_test_missclassified = 0
                PA_test_missclassified = 0
                #missclassification_accuracy = 0
                #total_samples = 0
                bonafide_missclassifications = 0
                PA_missclassifications = 0
                bonafide_samples = 0
                PA_samples = 0
                for data in self.test_loader:
                    images, labels, _ = data
                    current_batch_size = images.shape[0]
                    augmentation_params_input = (self.get_augentation_params_input(current_batch_size))
                    labels = labels.type(torch.FloatTensor).to(self.device)
                    bonafide_mask = (labels == 0)
                    PA_mask = (labels == 1)
                    bonafide_samples += bonafide_mask.sum().item()
                    PA_samples += PA_mask.sum().item()
                    #total_samples += current_batch_size

                    
                    transformed_images = torchvision.transforms.Lambda(lambda images: torch.stack(
                         [self.apply_augmentation(images[i],augmentation_params_input[i]) for i in range(current_batch_size)]))(images)

                    PAD_output_trans_images = self.get_PAD_output(transformed_images.to(self.device))
                    PAD_label_trans_images = torch.where((torch.sigmoid(PAD_output_trans_images).detach())>=self.args.threshold, 1.0, 0.0)
                    
                    #trans_test_missclassified += torch.sum((PAD_label_trans_images == 1-labels).int())
                    bonafide_test_missclassified += torch.sum(PAD_label_trans_images[bonafide_mask] == (1-labels[bonafide_mask])).int().item()
                    PA_test_missclassified += torch.sum(PAD_label_trans_images[PA_mask] == (1-labels[PA_mask])).int().item()
                    
                    images = self.normalize(images)
                    transformed_images = self.normalize(transformed_images)

                    images = images.to(self.device)
                    aug_params_input_scaled = self.get_scaled_aug_params(augmentation_params_input).to(self.device)                
                    transformed_images = transformed_images.to(self.device)
                    output = self.model(images, aug_params_input_scaled)
                    reconstruction_loss = self.model_criterion(output, transformed_images)
                    
                    PAD_output = self.get_PAD_output(denorm(output))
                    #label_output = torch.argmax(PAD_output, dim=1).type(torch.LongTensor).to(self.device)
                    output_PAScore = torch.sigmoid(PAD_output).detach()
                    
                    label_output = torch.where(output_PAScore>=self.args.threshold, 1.0, 0.0)
                    #correct_missclassifications = torch.sum((label_output == 1-labels).int())
                    #missclassification_accuracy += correct_missclassifications.item()
                    bonafide_missclassifications += torch.sum(label_output[bonafide_mask] == (1-labels[bonafide_mask])).int().item()
                    PA_missclassifications += torch.sum(label_output[PA_mask] == (1-labels[PA_mask])).int().item()
                    
                    classifier_loss = self.classifier_criterion(PAD_output, 1-labels)
                    total_loss = reconstruction_loss + self.args.lambda_adv*classifier_loss
                    test_loss.append(total_loss.item())
                    
            #self.lr_scheduler.step()
            print('Epoch [{}/{}], Train Loss: {:.8f}, Test Loss: {:.8f}, Transformed Test BPCER: {:.4f}, Transformed Test APCER: {:.4f}, Synthetic BPCER: {:.4f}, Synthetic APCER: {:.4f}'.format(
                    epoch+1, self.args.num_epochs, np.mean(train_loss), np.mean(test_loss), 
                    (bonafide_test_missclassified/bonafide_samples), (PA_test_missclassified/PA_samples),
                    (bonafide_missclassifications/bonafide_samples), (PA_missclassifications/PA_samples)))
            with open(os.path.join(self.args.log_path, self.exec_summary_file_name), 'a') as txt_file_object:
                    txt_file_object.write('\n>Epoch [{}/{}], Train Loss: {:.8f}, Test Loss: {:.8f}, Transformed Test BPCER: {:.4f}, Transformed Test APCER: {:.4f}, Synthetic BPCER: {:.4f}, Synthetic APCER: {:.4f})'.format(
                    epoch+1, self.args.num_epochs, np.mean(train_loss), np.mean(test_loss), 
                    (bonafide_test_missclassified/bonafide_samples), (PA_test_missclassified/PA_samples),
                    (bonafide_missclassifications/bonafide_samples), (PA_missclassifications/PA_samples)))
            txt_file_object.close()

            if (epoch+1) % self.args.model_save_epoch== 0:
                # Save the model
                torch.save(self.model.state_dict(), os.path.join(self.args.save_model_path, '{}_conv_AE.pth'.format(epoch + 1)))
                torch.save(self.optimizer.state_dict(), os.path.join(self.args.save_model_path, '{}_conv_AE_opt.pth'.format(epoch + 1)))
                
                # Plot sample synthetic images
                fig, axs = plt.subplots(4, self.args.num_samples, figsize=(30,24), squeeze=True)
                fig.suptitle("After epoch %d \n"%(epoch + 1), fontsize=34)
                row=0
                col=0
                for i in range(self.args.num_classes*self.args.num_samples):
                    if self.sampled_test_data["image_label"][i]==0:
                        label = 'Live'
                    else:
                        label= 'Spoof'
                    test_image = (self.iris_data.__getitem__(self.sampled_test_data["index"][i])[0]).to(self.device)
                    transformed_test_image = self.apply_augmentation(test_image,test_augmentation_params_input[i])
                    
                    output_image = self.model(self.normalize(test_image.unsqueeze(0)), 
                                              self.get_scaled_aug_params(test_augmentation_params_input[i].unsqueeze(0)))
                    
                    #test_image_predicted_label = torch.argmax(self.get_PAD_output(transformed_test_image), dim=1)[0].item()
                    #output_image_predicted_label = torch.argmax(self.get_PAD_output(denorm(output_image)), dim=1)[0].item()
                    test_image_PAScore = torch.sigmoid(self.get_PAD_output(transformed_test_image)).detach().cpu().numpy()
                    test_image_PAScore = np.round(test_image_PAScore, decimals=2)
                    output_image = denorm(output_image.detach()[0])
                    output_image_PAScore = torch.sigmoid(self.get_PAD_output(output_image)).detach().cpu().numpy()
                    output_image_PAScore = np.round(output_image_PAScore, decimals=2)
                    #axs[row][col].imshow(self.apply_augmentation(self.iris_data.get_PIL_image(self.sampled_test_data["index"][i]),
                    #                                             test_augmentation_params_input[i]), cmap='gray')
                    axs[row][col].imshow(tensor_to_PIL(transformed_test_image), cmap='gray')
                    
                    axs[row][col].set_title(label + "-"+ str(test_image_PAScore), fontsize=28)
                    axs[row][col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[row+1][col].imshow(tensor_to_PIL(output_image), cmap='gray')
                    axs[row+1][col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[row+1][col].set_title("Gen. " + label + " "+ str(output_image_PAScore), fontsize=28)
                    col=col+1
                    if col==self.args.num_samples:
                        row=row+2
                        col=0
                fig.tight_layout()
                plt.savefig(os.path.join(self.args.sample_path, '{}_sample_image.jpg'.format(epoch + 1)))
                plt.close()



                





