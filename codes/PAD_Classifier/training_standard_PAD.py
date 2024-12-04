import argparse
import os
import torch
from datetime import date
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from datasets import WarsawCroppedIrisImages, ClarksonCroppedIrisImages, NotreDameCroppedIrisImages, IIITDWVUCroppedIrisImages
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
        iris_data = ClarksonCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.JPG')
    elif args.dataset == 'Warsaw':
        iris_data = WarsawCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.png')
    elif args.dataset == 'NotreDame':
        iris_data = NotreDameCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.tiff')
    elif args.dataset == 'IIITD_WVU':
        iris_data = IIITDWVUCroppedIrisImages(args.image_path, args.image_height, args.image_width, '/*.bmp')
    print("Length of Dataset:" + str(len(iris_data)))

    print("Length of Train Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train')]), 
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Train') & (iris_data.dataset["image_label"] == 1)])))

    print("Length of Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test') & (iris_data.dataset["image_label"] == 1)])))


    
    # Create train, val and test loader
    train_set_indices = np.array((iris_data.dataset[iris_data.dataset["train_test_split"]=='Train'].index))
    labels = iris_data.dataset["image_label"].values
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=42)
    
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
    print(iris_data.dataset.filter(items=test_indices, axis=0)["train_test_split"].unique())
    if len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown')])>0:
        print("Length of Unknown Test Data: {:d} (Bonafide: {:d}, PA: {:d})".format(len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown')]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown') & (iris_data.dataset["image_label"] == 0)]),
                                                                   len(iris_data.dataset.loc[(iris_data.dataset["train_test_split"] == 'Test-unknown') & (iris_data.dataset["image_label"] == 1)])))
        unk_test_indices = list(iris_data.dataset[iris_data.dataset["train_test_split"]=='Test-unknown'].index)
        unk_test_data = torch.utils.data.Subset(iris_data, unk_test_indices)
        unk_test_loader = torch.utils.data.DataLoader(unk_test_data, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
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

    
    # Create a file to log output
    exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
    with open(os.path.join(args.output_path, "logs", exec_summary_file_name), 'w') as txt_file_object:
        txt_file_object.write("Parameters:\n")
        txt_file_object.write(json.dumps(vars(args)))
        #txt_file_object.write("\nlearning_rate: %s; beta1: %s; beta2:%s"%(args.learning_rate, args.beta1, args.beta2))
        #txt_file_object.write("\nlambda_adv: %s; gamma: %s; latent_dim:%s"%(args.lambda_adv, args.gamma, args.latent_dim))
        txt_file_object.write("\nExecution Summary:")
        txt_file_object.close()

    
    print("Start training Standard PAD for iris PA detection...")
    trainer = Trainer(iris_data, dataloader, args, device)
    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Standard PAD Classifier for a dataset")
    # specify to enable cuda
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="4,5,6,7", help="GPU devices to be used")
    
    # image specification
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--num_channels', type=int, default=1)
    #parser.add_argument('--file_type', type=str, default='/*.JPG')
    
    # path details
    parser.add_argument('--image_path', type=str, default='/research/iprobe-paldebas/Research_Work/Adversarial_Augmentation_DNetPAD/data/livdet-iris-2017-complete-cropped')
    parser.add_argument('--dataset', type=str, default='Clarkson', choices=['Clarkson', 'Warsaw', 'NotreDame', 'IIITD_WVU'], help='Dataset to be used for training')
    parser.add_argument('--output_path', type=str, default='outputs', help='Store the outputs of the run')


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
    parser.add_argument('--pretrained_model', type=str, default= 'densenet121', choices=['densenet121', 'resnet50','resnet101', 'vgg19'], help='Pretrained model to be used as backbone architecture')
    parser.add_argument('--version', type=str, default= 'Clarkson_densenet121_v1')
    parser.add_argument('--load_model', type=str, default=None)     # Need to update this parameter to start training from a checkpoint
    

    args = parser.parse_args()
    main(args)