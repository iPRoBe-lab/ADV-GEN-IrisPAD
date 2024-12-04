import argparse
import os
import torch
from datetime import date
import json
from trainer import Trainer
from utils import make_folder
import numpy as np

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
    make_folder(os.path.join(curr_wd, args.save_model_path), args.version)
    make_folder(os.path.join(curr_wd, args.sample_path), args.version)
    make_folder(os.path.join(curr_wd, args.log_path), args.version)
    args.save_model_path = os.path.join(curr_wd, args.save_model_path, args.version)
    args.sample_path = os.path.join(curr_wd, args.sample_path, args.version)
    args.log_path = os.path.join(curr_wd, args.log_path, args.version)
    
#    if args.load_model_path is not None:
#        args.load_model_path = os.path.join(curr_wd,args.load_model_path)
    
    args.threshold = np.float64(args.threshold)
    # Create a file to log output
    exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
    with open(os.path.join(args.log_path, exec_summary_file_name), 'w') as txt_file_object:
        txt_file_object.write("Parameters:\n")
        txt_file_object.write(json.dumps(vars(args)))
        #txt_file_object.write("\nlearning_rate: %s; beta1: %s; beta2:%s"%(args.learning_rate, args.beta1, args.beta2))
        #txt_file_object.write("\nlambda_adv: %s; gamma: %s; latent_dim:%s"%(args.lambda_adv, args.gamma, args.latent_dim))
        txt_file_object.write("\nExecution Summary:")
        txt_file_object.close()

    
    print("Start training the autoencoder...")
    trainer = Trainer(iris_data, train_loader, test_loader, args, device)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ADV_GEN")
    # specify to enable cuda
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="4,5,6,7", help="GPU devices to be used")
    
    # image specification
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--file_type', type=str, default='/*.JPG', help="Training image format")
    
    # path details
    parser.add_argument('--image_path', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/data/livdet-iris-2017-complete-cropped', help="Path to cropped images used for training and testing")
    parser.add_argument('--dataset', type=str, default='Clarkson', choices=['Clarkson', 'Warsaw', 'IIITD_WVU', 'NotreDame'], help="Dataset to be used")
    parser.add_argument('--pretrained_model', type=str, default= 'densenet121', choices=['densenet121', 'resnet101', 'vgg19'], help="Pretrained model to be used as backbone architecture")
    parser.add_argument('--standard_PAD_model', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/DNetPAD_Modified/outputs/Clarkson_densenet121_v1/models/iris_pa_densenet121_best.pth', help="Path to the Standard PAD Classifier model")
    parser.add_argument('--save_model_path', type=str, default='models', help="path to store ADV_GEN model")
    parser.add_argument('--sample_path', type=str, default='sample_images', help="path to store samples of generated images by ADV_GEN during training")
    parser.add_argument('--log_path', type=str, default='logs', help="path to store log of the run")
    #parser.add_argument('--load_model_path', type=str, default=None)

    # details of training paramaters
    parser.add_argument('--num_classes', type=int, default=2, help="Live and Spoof")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--num_epochs', type=int, default=500, help="number of iterations")
    parser.add_argument('--latent_dim', type=int, default=512, help="Latent dimension (encoder output)")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=.0001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam betas[0]')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value Learning Rate Schedular')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Weight to classifier loss')

    # Other parameters
    parser.add_argument('--version', type=str, default= 'Clarkson_conv_AE_v6_1_densenet121', help='Name of folder which stores all outputs of this run')
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--load_model', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to use in Standard PAD classifier to determine BPCER and APCER of generated images over epochs')

    args = parser.parse_args()
    main(args)