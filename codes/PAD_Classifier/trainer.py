import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import torch
from evaluation import Evaluation
import torchvision.models as models
import argparse
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from datasets import WarsawCroppedIrisImages, ClarksonCroppedIrisImages, NotreDameCroppedIrisImages, IIITDWVUCroppedIrisImages
from utils import make_folder

class Trainer():
    def __init__(self, iris_data, dataloader, args, device):
        
        self.iris_data = iris_data
        self.dataloader = dataloader
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.num_channels = args.num_channels
        self.args = args
        self.device = device
        self.exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
        self.build_model()
        
    def build_model(self):
        if self.args.pretrained_model == 'densenet121':
            model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
        elif self.args.pretrained_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
        elif self.args.pretrained_model == 'resnet101':
            model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
        elif self.args.pretrained_model == 'vgg19':
            model = models.vgg19(weights='VGG19_Weights.DEFAULT')
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
        
        self.model = model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model_criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, 
                                          betas=[self.args.beta1, self.args.beta2]
                                         )  
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size =50, gamma=self.args.gamma)   

    def train(self):
        if self.args.load_model is None:
            start = 0
        else:   # To start training from last checkpoint
            start = self.model.load_state_dict(self.args.load_model['epoch'])
            self.model.load_state_dict(self.args.load_model['model_state_dict'])
            self.optimizer.load_state_dict(self.args.load_model['optimizer_state_dict'])
        print("Training using Adam optimizer:")
        best_accuracy = 0
        best_loss= float('inf')
        best_epoch = -1
        best_model = None
        train_loss_hist = []
        val_loss_hist = [] 
        for epoch in range(start, self.args.num_epochs):  
            for phase in ['train', 'val']:
                predict_scores = []
                true_labels = []
                epoch_loss = 0
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                with torch.set_grad_enabled(phase=='train'):
                    for data  in self.dataloader[phase]:    
                        images, labels, _ = data
                        images = images.repeat(1,3,1,1).to(self.device)
                        labels = labels.type(torch.FloatTensor).to(self.device)
                        output= self.model(images).squeeze()
                        loss = self.model_criterion(output, labels)
                        predict_scores.extend(torch.sigmoid(output.detach()).cpu().numpy())
                        true_labels.extend(labels.cpu().numpy().astype(int))
                        epoch_loss +=loss.item()
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        
                predict = predict_scores >= np.float64(0.5)
                predict_labels =[]
                [predict_labels.append(int(predict[i])) for i in range(len(predict))]
                correct_predictions = np.sum(np.array(predict_labels) == true_labels)
                if phase == 'train':
                    train_acc = float(correct_predictions)/float(len(true_labels))
                    train_loss = epoch_loss/len(self.dataloader['train'])    
                    train_loss_hist.append(train_loss)
                elif phase == 'val':
                    val_acc = float(correct_predictions)/float(len(true_labels))
                    val_loss = epoch_loss/len(self.dataloader['val'])  
                    val_loss_hist.append(val_loss)  
                    
                    if (val_loss<=best_loss or val_acc > best_accuracy):
                        best_loss = val_loss
                        best_accuracy =val_acc
                        best_epoch = epoch+1
                        best_model = self.model.state_dict()
                        torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            }, os.path.join(self.args.output_path, "models", 'iris_pa_best.pth'))

                        print("\nEvaluating validation data after epoch {}".format(epoch+1))
                        evaluation = Evaluation(None, true_labels, predict_scores, os.path.join(self.args.output_path, "validation_results"), "validation_epoch_"+str(epoch + 1))
                        best_threshold = evaluation.calculate_ACER()        
            self.lr_scheduler.step()
            print('Epoch [{}/{}], Train Loss: {:.8f}, Train Acc.: {:.2f}, Val Loss: {:.8f}, Val Acc.: {:.2f}'.format(
                epoch+1, self.args.num_epochs, train_loss, train_acc, val_loss, val_acc))
            with open(os.path.join(self.args.output_path, "logs", self.exec_summary_file_name), 'a') as txt_file_object:
                    txt_file_object.write('\n>Epoch [{}/{}], Train Loss: {:.8f}, Train Acc.: {:.2f}, Val Loss: {:.8f}, Val Acc.: {:.2f}'.format(
                    epoch+1, self.args.num_epochs, train_loss, train_acc, val_loss, val_acc))
            txt_file_object.close()

        print("Training completed.")
        # Save last model
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.args.output_path, "models", 'iris_pa_last_chkpt.pth'))

        # Plotting of train and test loss
        plt.figure()
        plt.xlabel('Epoch Count')
        plt.ylabel('Loss')
        plt.plot(np.arange(0, self.args.num_epochs), train_loss_hist[:], color='r')
        plt.plot(np.arange(0, self.args.num_epochs), val_loss_hist[:], 'b')
        plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
        plt.savefig(os.path.join(self.args.output_path, "models", 'Loss.jpg'))
        plt.close()

        print("Best epoch:" + str(best_epoch))
        print("Threshold on validation data using best model:" + str(best_threshold))  
        with open(os.path.join(self.args.output_path, "logs", self.exec_summary_file_name), 'a') as txt_file_object:
                    txt_file_object.write('\n\nBest Epoch: {}, Best Threshold: {:.3f}, Best Validation Loss: {:.8f}'.format(
                    best_epoch, best_threshold, best_loss))  
        
        print("\nEvaluation on Test Data Ater Training:")
        self.model.load_state_dict(best_model)
        self.model.eval()
        with torch.no_grad():
            for phase in ['test', 'test_unknown']:
                predict_scores_test=[]
                image_names_test = []
                true_labels_test = []
                test_loss = 0
                if self.dataloader[phase] is not None:
                    for data in self.dataloader[phase]:  
                        images, labels, image_names = data
                        images = images.repeat(1,3,1,1).to(self.device)
                        labels = labels.type(torch.FloatTensor).to(self.device)
                        output= self.model(images).squeeze()
                        loss = self.model_criterion(output, labels)
                        test_loss += loss.item()
                        predict_scores_test.extend(torch.sigmoid(output.detach()).cpu().numpy())
                        true_labels_test.extend(labels.cpu().numpy())
                        image_names_test.extend(image_names)
                    test_loss = test_loss/len(self.dataloader[phase])
                    print('\nTest Loss ({}): {:.8f}'.format(phase, test_loss))
                    with open(os.path.join(self.args.output_path, "logs", self.exec_summary_file_name), 'a') as txt_file_object:
                        txt_file_object.write('\n\nTest Loss ({}): {:.8f}'.format(phase, test_loss))
                        txt_file_object.close()

                    evaluation = Evaluation(image_names_test, true_labels_test, predict_scores_test, os.path.join(self.args.output_path, "test_results"), phase)
                    evaluation.calculate_ACER(best_threshold)     

