import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import classification_report
from scipy.optimize import linear_sum_assignment
from utils.utils import linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans
import pandas as pd
import wandb
import seaborn as sns
from scipy import stats

class Linear(torch.nn.Module):
    def __init__(self,n_out=10):
        super().__init__()
        self.model=torch.nn.Sequential(torch.nn.LazyLinear(n_out))
        
    def forward(self,x):
        return self.model(x)




def just_report(y_true,y_pred):
    """just compute the report"""
    class_report=classification_report(y_true, y_pred)
    class_report_dict=classification_report(y_true, y_pred,output_dict=True)
    df = pd.DataFrame(class_report_dict).transpose()
    accuracy=accuracy_score(y_true,y_pred)

    return class_report,df,accuracy

def evaluate_linear(z_sampled_train,labels_train,BS=512,lr=1e-3,epochs=10,device="cpu"):

    lin_z=Linear()
    lin_z=lin_z.to(device)

    opt_z=torch.optim.Adam(lin_z.parameters(),lr=lr)
    criterion=torch.nn.CrossEntropyLoss()

    z_train_dataset=torch.utils.data.TensorDataset(z_sampled_train,labels_train)
    z_train_dataloader=torch.utils.data.DataLoader(z_train_dataset,shuffle=True,batch_size=BS)

    z_loss,z_acc=train_linear(model=lin_z,criterion=criterion,optim=opt_z,epochs=epochs,train_dataloader=z_train_dataloader)

    return lin_z,z_loss,z_acc




def test_linear(model,z_sampled,labels,BS=512,device="cpu"):

    z_dataset=torch.utils.data.TensorDataset(torch.Tensor(z_sampled),torch.Tensor(labels))
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



def train_linear(model,criterion,optim,epochs,train_dataloader,device="cpu"):


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




def get_semisupervised_latents(train_dataloader,model,BS,device="cuda:0",n_semi=3000):
    x_semi=np.array([])
    z_semi=np.array([])
    h_semi=np.array([])
    y_semi=np.array([])
    c_semi=np.array([])

    with torch.no_grad():

        n_semi=3000
        passes=n_semi//BS
        c=0
        for x_i,x_j,y in tqdm.tqdm(train_dataloader):

            x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)
            x_i_recon,z_i,h_i,c_i = model(x_i)

            x_semi=np.append(x_semi,x_i_recon.cpu().numpy())
            z_semi=np.append(z_semi,z_i.cpu().numpy())
            h_semi=np.append(h_semi,h_i.cpu().numpy())
            y_semi=np.append(y_semi,y.cpu().numpy())
            c_semi=np.append(c_semi,c_i.cpu().numpy())

            c+=1
            if c==passes:
                break

    z_semi=z_semi.reshape((n_semi,-1))
    h_semi=h_semi.reshape((n_semi,-1))
    x_semi=x_semi.reshape((n_semi,-1))
    y_semi=y_semi.reshape((n_semi,-1))
    c_semi=c_semi.reshape((n_semi,-1))
    
    return z_semi,h_semi,x_semi,y_semi,c_semi


def get_test_latents(test_dataloader,len_dataset,model,BS,device="cuda:0"):
    x_test=np.array([])
    z_test=np.array([])
    h_test=np.array([])
    y_test=np.array([])
    c_test=np.array([])


    with torch.no_grad():

        for x_i,x_j,y in tqdm.tqdm(test_dataloader):

            x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)
            x_i_recon,z_i,h_i,c_i = model(x_i)

            x_test=np.append(x_test,x_i_recon.cpu().numpy())
            z_test=np.append(z_test,z_i.cpu().numpy())
            h_test=np.append(h_test,h_i.cpu().numpy())
            y_test=np.append(y_test,y.cpu().numpy())
            c_test=np.append(c_test,c_i.cpu().numpy())

    l=len_dataset
    z_test=z_test.reshape((l,-1))
    h_test=h_test.reshape((l,-1))
    x_test=x_test.reshape((l,-1))
    y_test=y_test.reshape((l,-1))
    c_test=c_test.reshape((l,-1))
        
    return z_test,h_test,x_test,y_test,c_test


def mapping(cl,y,n_cluster):
    """map[k]=label so we map the cluster k-th to the label"""

    map={}

    acc={}

    for k in range(n_cluster):
        if len(y[cl==k]):
            
            v = int(stats.mode(y[cl==k])[0][0])
            #v=y[cl==k].mode().values
            map[k]=v
            acc[k]=(y[cl==k]==v).sum()/len(y[cl==k]) 

    return map,acc

def evaluate_kmeans_baseline(model,n_k,batch_size,train_dataloader,test_dataloader,len_dataset,wandb_log=True):
    
    BS=batch_size
    z_semi,h_semi,x_semi,y_semi,c_semi=get_semisupervised_latents(train_dataloader,model,BS)
    z_test,h_test,x_test,y_test,c_test=get_test_latents(test_dataloader,len_dataset,model,BS)
    
    ## Kmeans
    
    kmodel=KMeans(n_clusters=n_k)

    kmodel.fit(z_semi)

    y_pred=kmodel.predict(z_test)
    
    y_pred=np.argmax(c_test,-1)
    
    maps,acc=mapping(y_pred,y_test.squeeze(),n_k)
    y_pred_aligned=[maps[i] for i in y_pred]
    
    print(classification_report(y_test,y_pred_aligned))

    if wandb_log:
        wandb.log({"Report":classification_report(y_test,y_pred_aligned,output_dict=True)})

    print(f"Accuracy {accuracy_score(y_test,y_pred_aligned)}")

    fig,ax=plt.subplots(figsize=(15,10))
    sns.heatmap(confusion_matrix(y_test, y_pred_aligned,normalize="true"),annot=True,ax=ax,linewidth=1.,linecolor="white")
    
    if wandb_log:
        wandb.log({"kMeans heatmap":wandb.Image(fig)})