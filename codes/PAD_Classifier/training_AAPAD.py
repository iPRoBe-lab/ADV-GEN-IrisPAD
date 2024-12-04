import argparse
import os
import torch
from datetime import date
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from adversarial_dataset import ClarksonCroppedIrisImagesAdversarial
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
    iris_data = ClarksonCroppedIrisImagesAdversarial(args.image_path, args.image_height, args.image_width, args.file_type,
                                                   args.adv_image_path, args.adv_file_type, device, args.transform, args.pretrained_model)
    print("Length of Dataset:" + str(len(iris_data)))

    print("Length of Train Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train')]), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 1)])))

    print("Length of Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 1)])))

    print("No. of Adversarial Images- Transformed Bonafide: {:d}, Synthetic Bonafide: {:d}, Transformed PA: {:d}, Synthetic PA: {:d}".format(len(iris_data.dataset.loc[(iris_data.dataset["image_type"] == "Tranformed_Adversarial_Bonafide")]), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["image_type"] == 'Synthetic_Adversarial_Bonafide')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["image_type"] == 'Tranformed_Adversarial_PA')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["image_type"] == 'Synthetic_Adversarial_PA')])))
    
    # Create train, val and test loader
    train_set_indices = np.array(iris_data.dataset[iris_data.dataset["train_test_split"]=='Train'].index)
    labels = iris_data.dataset["image_label"].values
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=42)
    #train_indices, val_indices = train_test_split(train_indices, test_size=args.val_split, random_state=42)
    train_indices, val_indices = next(stratified_split.split(train_set_indices, labels[train_set_indices]))
    train_indices = train_set_indices[train_indices]
    val_indices = train_set_indices[val_indices]
    test_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Test'].index)
    
    train_data = torch.utils.data.Subset(iris_data, train_indices)
    val_data = torch.utils.data.Subset(iris_data, val_indices)
    test_data = torch.utils.data.Subset(iris_data, test_indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("After train and validation split:")
    print("Selected bonafides in train set:"+str((labels[train_indices]==0).sum()))
    print("Selected PAs in train set:"+str((labels[train_indices]==1).sum()))
    print("Selected bonafides in val set:"+str((labels[val_indices]==0).sum()))
    print("Selected PAs in val set:"+str((labels[val_indices]==1).sum()))
    print(iris_data.dataset.filter(items=train_indices, axis=0)["train_test_split"].unique())
    print(iris_data.dataset.filter(items=val_indices, axis=0)["train_test_split"].unique())
    
    if len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown')])>0:
        print("Length of Unknown Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown') & (iris_data.dataset["image_label"] == 1)])))
        unk_test_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Test-unknown'].index)
        unk_test_data = torch.utils.data.Subset(iris_data, unk_test_indices)
        unk_test_loader = torch.utils.data.DataLoader(unk_test_data, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
        print("Length of unknown test loader:" + str(len(unk_test_loader)))
    else: unk_test_loader = None
    
    dataloader = {'train':train_loader, 'val':val_loader, 'test':test_loader, 'test_unknown':unk_test_loader}
        
    # Retrieve current working directory and create required folders to save model, outputs and logs
    curr_wd = os.path.dirname(__file__)
    make_folder(os.path.join(curr_wd, args.output_path), args.version)
    args.output_path = os.path.join(curr_wd, args.output_path, args.version)
    make_folder(os.path.join(curr_wd, args.output_path), "validation_results", replace=True)
    make_folder(os.path.join(curr_wd, args.output_path), "test_results")
    make_folder(os.path.join(curr_wd, args.output_path), "logs")
    make_folder(os.path.join(curr_wd, args.output_path), "models")

    #make_folder(os.path.join(curr_wd, args.save_model_path), args.version)
    #args.save_model_path = os.path.join(curr_wd, args.save_model_path, args.version)
    #make_folder(os.path.join(curr_wd, args.log_path), args.version)
    #args.log_path = os.path.join(curr_wd, args.log_path, args.version)
    
    # Create a file to log output
    exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
    with open(os.path.join(args.output_path, "logs", exec_summary_file_name), 'w') as txt_file_object:
        txt_file_object.write("Parameters:\n")
        txt_file_object.write(json.dumps(vars(args)))
        #txt_file_object.write("\nlearning_rate: %s; beta1: %s; beta2:%s"%(args.learning_rate, args.beta1, args.beta2))
        #txt_file_object.write("\nlambda_adv: %s; gamma: %s; latent_dim:%s"%(args.lambda_adv, args.gamma, args.latent_dim))
        txt_file_object.write("\nExecution Summary:")
        txt_file_object.close()

    
    print("Start training the AAPAD for iris PA detection to include original and adversarial images during training...")
    trainer = Trainer(iris_data, dataloader, args, device)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating Augmented Adversarial Iris Images Using DenseNet121")
    # specify to enable cuda
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="4,5,6,7", help="GPU devices to be used")
    
    # image specification
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--file_type', type=str, default='/*.JPG', help='Image type of original images')
    parser.add_argument('--adv_file_type', type=str, default='/*.jpg', help='Image type of the synthetic adversarial images')
    
    # path details
    parser.add_argument('--image_path', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/data/livdet-iris-2017-complete-cropped/Clarkson')
    parser.add_argument('--adv_image_path', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/Adversarial_Transformation/synthetic_train_images/Clarkson_conv_AE_v6_1_densenet121/500_conv_AE.pth')
    parser.add_argument('--transform', type=str, default=True, help='whether transformation parameters used in generated images or not')
    #parser.add_argument('--save_model_path', type=str, default='models')
    parser.add_argument('--output_path', type=str, default='outputs', help='Store the outputs of the run')
    #parser.add_argument('--log_path', type=str, default='logs')
    #parser.add_argument('--load_model_path', type=str, default=None)    # To train model from a checkpoint

    # details of training paramaters
    parser.add_argument('--num_classes', type=int, default=2, help="Live and Spoof")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--num_epochs', type=int, default=50, help="number of iterations")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.20, help='Validation split')
    parser.add_argument('--learning_rate', type=float, default=.0001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam betas[0]')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value Learning Rate Schedular')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay in optimizer')
    
    # Other parameters
    parser.add_argument('--pretrained_model', type=str, default= 'densenet121', choices=['densenet121', 'resnet50', 'vgg19'], help='Pretrained model to be used as backbone architecture')
    parser.add_argument('--version', type=str, default= 'Clarkson_densenet121_adv_v1_1', help='Folder name to distinguish between different runs')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--load_model_epoch', type=int, default=0)
    

    args = parser.parse_args()
    main(args)