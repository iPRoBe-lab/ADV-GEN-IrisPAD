import os
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
import pandas as pd

class Evaluation:
    def __init__(self, image_names, true_label, predict_score, output_path, label):
        self.image_names = image_names
        self.predict_score = predict_score
        self.true_label = true_label
        self.output_path = output_path
        self.label = label
        self.create_score_histogram()
        self.plot_roc_curve()

    # Histogram plot
    def create_score_histogram(self):
        self.live = []
        [self.live.append(self.predict_score[i]) for i in range(len(self.true_label)) if (self.true_label[i] == 0)]
        self.spoof = []
        [self.spoof.append(self.predict_score[j]) for j in range(len(self.true_label)) if (self.true_label[j] == 1)]
        bins = np.linspace(np.min(np.array(self.spoof + self.live)), np.max(np.array(self.spoof + self.live)), 60)
        plt.figure()
        plt.hist(self.live, bins, alpha=0.5, label='Bonafide', density=True, edgecolor='black', facecolor='g')
        plt.hist(self.spoof, bins, alpha=0.5, label='PA', density=True, edgecolor='black',facecolor='r' )
        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('Scores')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_path, self.label +"_Histogram.jpg"))
        plt.close()
    
    def get_threshold(self, fprs, thresholds, fpr):
        # Getting threshold for particular fpr
       threshold = 0
       for x in range(0, fprs.size):
         if fprs[x] >= fpr:
             break
         threshold = thresholds[x]
       return threshold

    def plot_roc_curve(self):
        (fprs, tprs, thresholds) = roc_curve(self.true_label, self.predict_score)
        plt.figure()
        plt.semilogx(fprs, tprs)
        plt.grid(True, which="major")
        #plt.legend(loc='lower right', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([0.001, 0.002, 0.01, 0.1, 1])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Detection Rate')
        plt.xlim((0.0005, 1.01))
        plt.ylim((0, 1.02))
        plt.plot([0.002, 0.002], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.plot([0.001, 0.001], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.plot([0.01, 0.01], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.savefig(os.path.join(self.output_path, self.label +"_ROC.jpg"))
        plt.close()

        #Plot Raw ROC curves
        plt.figure()
        plt.plot(fprs, tprs)
        plt.grid(True, which="major")
        #plt.legend(self.label, loc='lower right', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([0.01, 0.1, 1])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Detection Rate')
        plt.xlim((0.0005, 1.01))
        plt.ylim((0, 1.02))
        plt.savefig(os.path.join(self.output_path, self.label +"_RawROC.jpg"))
        plt.close()
    
        # Calculation of TDR at 0.2% , 0.1%, 1% and  5% FDR
        with open(os.path.join(self.output_path , self.label +'_TDR-ACER.csv'), mode='w+') as fout:
            fprArray = [0.002,0.001, 0.01, 0.05]
            for fpr in fprArray:
                tpr = np.interp(fpr, fprs, tprs)
                threshold = self.get_threshold(fprs, thresholds, fpr)
                fout.write("TDR @ FDR, threshold: %f @ %f ,%f\n" % (tpr, fpr, threshold))
                #print("TDR @ FDR, threshold: %f @ %f ,%f " % (tpr, fpr, threshold))

    def calculate_ACER(self,minThreshold=-1):    
        # Calculation of APCER, BPCER and ACER
        with open(os.path.join(self.output_path , self.label +'_TDR-ACER.csv'), mode='a') as fout:
            if minThreshold == -1:
                minACER= 10000
                for thresh in np.arange(0,1,0.025):
                    APCER = np.count_nonzero(np.less(self.spoof,thresh))/len(self.spoof)
                    BPCER = np.count_nonzero(np.greater_equal(self.live,thresh))/len(self.live)
                    ACER = (APCER + BPCER)/2
                    if ACER < minACER:
                        minThreshold = thresh
                        minAPCER = APCER
                        minBPCER = BPCER
                        minACER = ACER
                fout.write("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (minAPCER, minBPCER, minACER, minThreshold))
                print("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (minAPCER, minBPCER, minACER, minThreshold))
            else:
                APCER = np.count_nonzero(np.less(self.spoof, minThreshold)) / len(self.spoof)
                BPCER = np.count_nonzero(np.greater_equal(self.live, minThreshold)) / len(self.live)
                ACER = (APCER + BPCER) / 2
                fout.write("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (APCER, BPCER, ACER, minThreshold))
                print("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f" % (APCER, BPCER, ACER, minThreshold))
        
        predict = self.predict_score >= minThreshold
        predict_label =[]
        [predict_label.append(int(predict[i])) for i in range(len(predict))]
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(self.true_label, np.array(predict_label))).plot().figure_.savefig(os.path.join(self.output_path, self.label +"_conf_matrix.jpg"))
        plt.close()
        if self.image_names is not None:
            pd.DataFrame({"img_file_name":self.image_names, "true_label":self.true_label, "PA_Score":self.predict_score, "predicted_label":predict_label}).to_csv(os.path.join(self.output_path, self.label +"_match_score.csv"))

        return minThreshold

