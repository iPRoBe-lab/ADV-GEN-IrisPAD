import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
 
# Define the autoencoder architecture
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, image_height=224, image_width=224, num_channels =1, num_aug_params=12, latent_dim=512):
        super(ConvolutionalAutoencoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        self.num_aug_params = num_aug_params
        self.latent_dim = latent_dim

        self.aug_param_process = nn.Sequential(
            nn.Linear(self.num_aug_params, self.image_height*self.image_width),
            #nn.LeakyReLU(),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_channels+1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            #nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            #n.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),
            #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            #nn.ReLU(),
            #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #n.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(512*7*7, self.latent_dim),
            #nn.ReLU(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(),
            #nn.Linear(4096,self.latent_dim)
        )

        self.decoder_input_process = nn.Sequential(
        nn.Linear(self.latent_dim, 512*7*7),
        #nn.LeakyReLU(),
        #nn.Linear(4096, 512*7*7)
    )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(32, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
         
    def forward(self, images, aug_params_input):
        aug_params = self.aug_param_process(aug_params_input)
        aug_params = aug_params.view(-1, 1, 224, 224)
        encoder_input = torch.cat((images , aug_params), dim=1)
        encoder_output = self.encoder(encoder_input)
        decoder_input = self.decoder_input_process(encoder_output)
        decoder_input = decoder_input.view(-1, 512, 7, 7)
        decoder_output = self.decoder(decoder_input)
        return decoder_output