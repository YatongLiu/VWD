import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def Evalutions(model_name,y_test,y_pred):
    model = [] # 保存模型名称
    classes = [] # 保存标签名称
    lst1 = []
    hammingloss = 0    
    # Classes = y_test.toarray().shape[1]
    hammingloss = round(hamming_loss(y_test,y_pred),4)
    # fig = plt.figure(figsize=(20,13),dpi=80)
    for i in range(y_test.toarray().shape[1]):
        model.append(model_name)
        classes.append('Class_'+str(i))
        sw = compute_sample_weight(class_weight='balanced',y=y_test.toarray()[:,i])
        cm = confusion_matrix(y_test.toarray()[:,i], y_pred.toarray()[:,i],sample_weight=sw)
        
        '''
        conf_matrix = pd.DataFrame(cm, index=['0','1'], columns=['0','1'])
                
        fig1 = fig.add_subplot(2,3,i+1)
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
        plt.title('Confusion matrix on Class {}'.format(i))
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig('result/Figure/{0}_confusion.png'.format(model_name), bbox_inches='tight')
        '''
        Accuracy = round(accuracy_score(y_test.toarray()[:,i], y_pred.toarray()[:,i],sample_weight=sw),4)
        Precision = round(precision_score(y_test.toarray()[:,i], y_pred.toarray()[:,i],sample_weight=sw),4)
        Recall = round(recall_score(y_test.toarray()[:,i], y_pred.toarray()[:,i],sample_weight=sw),4)
        f1 = round(f1_score(y_test.toarray()[:,i], y_pred.toarray()[:,i],sample_weight=sw),4)
        lst1.append([Accuracy,Precision,Recall,f1])
        
    return hammingloss,model,classes,lst1

def AUPRC(model_name,y_test,y_score):
    Classes = y_test.toarray().shape[1]
    au_prc,thre = [],[]
    fig = plt.figure(figsize=(20,10),dpi=80)
    for i in range(Classes):
        precision, recall, thresholds = precision_recall_curve(y_test.toarray()[:,i], y_score.toarray()[:,i])
        thre.append([precision, recall, thresholds])
        average_precision = average_precision_score(y_test.toarray()[:,i], y_score.toarray()[:,i])
        au_prc.append(round(average_precision,2))
        
        fig2 = fig.add_subplot(2,3,i+1)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.title('PRC on Class {0}: AP={1:0.2f}'.format(i,average_precision))
        plt.savefig("result\Figure\{0}_PRC.png".format(model_name))
        
    return au_prc,thre

def AUROC(model_name,y_test,y_score):
    Classes = y_test.toarray().shape[1]
    au_roc = []
    # fig = plt.figure(figsize=(20,10),dpi=80)
    for i in range(Classes):
        fpr, tpr, thresholds = roc_curve(y_test.toarray()[:,i], y_score.toarray()[:,i], pos_label=1)
        AUC_ROC = roc_auc_score(y_test.toarray()[:,i], y_score.toarray()[:,i])
        au_roc.append(round(AUC_ROC,2))
        '''
        fig1 = fig.add_subplot(2,3,i+1)
        plt.plot(fpr, tpr, color='darkorange',lw=2, label='{0} (area = {1:.2f})'.format(model_name,AUC_ROC)) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('ROC curve on Class {}'.format(i))
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.legend(loc="lower right")
        plt.savefig("result/Figure/{0}_ROC.png".format(model_name))
        '''
    return au_roc

'''
def get_genotype(genotypes):
    gtype = {'Not Known': 0,
           'Heterozygous': 1,
           'Linked inheritance': 2,
           'Compound Heterozygous': 3,
           'Homologous': 4}
    
    get_gtype = []
    for genotype in genotypes:
        for key,value in gtype.items():
            get_gtype.append(value)
    
    return np.array(get_gtype)
'''
