import numpy as np
from torch import nn
import torch.distributions as D
import torch
import tqdm
from sklearn.manifold import TSNE
from torch.nn.functional import softmax

from sklearn.metrics import classification_report,accuracy_score,silhouette_score,davies_bouldin_score,normalized_mutual_info_score
import seaborn as sns
import pandas as pd
import torchextractor as tx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


class Linear(torch.nn.Module):
    def __init__(self,n_out=10):
        super().__init__()
        self.model=torch.nn.Sequential(torch.nn.LazyLinear(n_out))
        
    def forward(self,x):
        return self.model(x)
        

def mapping(cl,y,n_cluster):
    """map[k]=label so we map the cluster k-th to the label"""

    map={}

    acc={}
    mode=int(stats.mode(y)[0][0])
    #print(f"[DEBUG] {mode}")
    for k in range(n_cluster):
        if len(y[cl==k]):
            
            v = int(stats.mode(y[cl==k])[0][0])
            #v=y[cl==k].mode().values
            map[k]=v
            acc[k]=(y[cl==k]==v).sum()/len(y[cl==k]) 
        
        else:
            map[k]=mode
            print(f"[WARNING] cluster {k} matched no classes. It's now linked to the mode of the labels. This could lead to decrease in metrics.")
    return map,acc

def just_report(y_true,y_pred):
    """just compute the report"""
    class_report=classification_report(y_true, y_pred)
    class_report_dict=classification_report(y_true, y_pred,output_dict=True)
    df = pd.DataFrame(class_report_dict).transpose()
    accuracy=accuracy_score(y_true,y_pred)

    return class_report,df,accuracy


def lin_evaluate(z_train,y_train,z_test,y_test,BS=512,lr=1e-3,epochs=10,device="cuda"):
        
    lin_z=Linear()
    lin_z=lin_z.to(device)
        
    opt_z=torch.optim.Adam(lin_z.parameters(),lr=lr)
    criterion=torch.nn.CrossEntropyLoss()
        
    z_train_dataset=torch.utils.data.TensorDataset(torch.Tensor(z_train),torch.Tensor(y_train).long())
    z_train_dataloader=torch.utils.data.DataLoader(z_train_dataset,shuffle=True,batch_size=BS)
    z_loss,z_acc=train_linear(model=lin_z,
                        criterion=criterion,
                        optim=opt_z,epochs=epochs,
                        train_dataloader=z_train_dataloader,device=device)
    
    y_true,y_pred=test_linear(lin_z,z_test,y_test,device=device)
    class_report, df, lin_acc=just_report(y_true,y_pred)
    
    model_stuff={"model":lin_z, "loss": z_loss, "train_acc":z_acc}
    
    return class_report,df,lin_acc,model_stuff
    

    
def test_linear(model,z_test,labels,BS=512,device="cuda"):

    z_dataset=torch.utils.data.TensorDataset(torch.Tensor(z_test),torch.Tensor(labels).long())
    z_dataloader=torch.utils.data.DataLoader(z_dataset,shuffle=True,batch_size=BS)

    labels=[] #they are real labels
    preds=[] #they are prediciton from model

    with torch.no_grad():
        for x,y in z_dataloader:

            x,y=x.to(device),y.to(device)
            y_pred=model(x)
            labels.append(y.cpu())
            preds.append(y_pred.cpu().argmax(dim=1))

    labels=torch.cat(labels)
    preds=torch.cat(preds)

    return labels.numpy(),preds.numpy()
    
    
    
def train_linear(model,criterion,optim,epochs,train_dataloader,device="cuda"):


    model.train()
    loss_history=[]
    acc_history=[]

    pbar=tqdm.tqdm(range(epochs))
    for epoch in pbar:
        for x,y in train_dataloader:

            x,y=x.to(device),y.to(device)

            optim.zero_grad()
            y_pred=model(x)

            if y.shape[-1]==1:
                y=y.squeeze(dim=-1) #handle labels with shape (bs,1) instead of just (bs,)

            loss=criterion(y_pred,y)
            loss.backward()
            optim.step()
            loss_history.append(loss.item())

            ##acc
            acc=(y_pred.argmax(dim=1)==y).sum()/len(y)
            acc_history.append(acc.cpu())
        pbar.set_description(f"[LINEAR] epoch {epoch} loss: {loss_history[-1]} acc: {acc_history[-1]}")

    return loss_history,acc_history



def kNN_evaluation(z_train,y_train,z_test,y_test,neighbours=5):
    neigh = KNeighborsClassifier(neighbours)
    neigh.fit(z_train,y_train)
    y_pred_neigh=neigh.predict(z_test)
    class_report, df, neigh_accuracy=just_report(y_test,y_pred_neigh)
    
    return class_report,df,neigh_accuracy


def stat_evaluation(cl_test,y_test,map):
    
    print(map)
    y_pred=[map[i] for i in cl_test]
    return just_report(y_test,y_pred)



def model_evaluation(z_train,y_train,z_test,y_test,cl_train,cl_test,n_cluster=10,device="cpu"):
    
    print("[INFO] compute mappings and stat accuracy")
    map,acc=mapping(cl_train,y_train,n_cluster=n_cluster)
    
    stat_report,stat_df,stat_acc=stat_evaluation(cl_test,y_test,map)
    
    print(stat_report)
    
    print("[INFO] compute kNN accuracy")
    
    knn_report,knn_df,knn_acc=kNN_evaluation(z_train,y_train,z_test,y_test)
    print(knn_report)
    
    print("[INFO] compute linear accuracy")
    
    lin_report,lin_df,lin_acc,model_stuff=lin_evaluate(z_train,y_train,z_test,y_test,device=device)
    print(lin_report)
    
    outputs={"stat_df":stat_df,"stat_acc":stat_acc,"knn_df":knn_df,"kNN_acc":knn_acc,"lin_df":lin_df,"lin_acc":lin_acc}
    return outputs