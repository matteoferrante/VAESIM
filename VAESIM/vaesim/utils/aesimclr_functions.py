import numpy as np
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms, models
from torchvision.transforms import *

import matplotlib.pyplot as plt
import tqdm

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer

from sklearn.metrics.cluster import normalized_mutual_info_score
from utils.utils import linear_assignment

from scipy import stats
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
from torch.nn.functional import softmax
from IPython.display import clear_output
import wandb
from utils.aesimclr_evaluation import *

import medmnist
from medmnist import INFO, Evaluator
from monai import transforms as m_transforms

import pandas as pd

from classes.SIMCLR import SimCLR
from utils.datasets import ContrastiveDataset
from utils.losses import DynamicSimCLR_Loss
from torchsummary import summary
from os.path import join as opj
from torchvision.utils import make_grid 

def find_optimal_cluster(model,train_dataloader,device,plot_tsne=True,wandb_log=True,k=(2,20)):

    z_train=[]
    h_train=[]

    with torch.no_grad():
        for x_i,x_j,y in tqdm.tqdm(train_dataloader):
            x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)
            x_i_recon,z_i,h_i,c_i = model(x_i)
            z_train+=z_i.cpu().numpy().tolist()
            h_train+=h_i.cpu().numpy().tolist()


    fig,ax=plt.subplots(1,1,figsize=(8,8))

    
    kmodel = KMeans()
    visualizer = KElbowVisualizer(kmodel, k=k,ax=ax)

    h_train=np.array(h_train)
    
    visualizer.fit(h_train)        # Fit the data to the visualizer
    
    
    visualizer.show()        # Finalize and render the figure

    optimal=visualizer.elbow_value_

    if optimal is None:
        optimal=int(k[0]+0.5*(k[1]-k[0]))
        print(f"[WARNING] Failed to detect optimal number of cluster, choosing minimum number as {optimal}")
    else:
        print(f"[INFO] Optimal number of clusters is {optimal}, fitting final kMeans")
    final_model=KMeans(n_clusters=optimal)
    final_model.fit(h_train)

    if wandb_log:
        wandb.log({"KMeans Elbow": wandb.Image(fig)})

    plot_kmeans(final_model,h_train,n_clusters=optimal)

    return final_model,optimal



def plot_kmeans(kmeans,X,n_clusters,wandb_log=True):
    
    labels=kmeans.predict(X)

    tsne = TSNE(n_components=2,perplexity=50,random_state=42)
    x_tsne=tsne.fit_transform(X)

    #cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray'}

    cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray',10:'tab:red', 11: 'tab:orange', 12: 'tab:green', 13: 'cyan', 14: 'lime', 15:'seagreen',16: 'tab:green',17:'tab:purple',18:'tab:pink',19:'aquamarine',20:'tab:olive',21:'tab:gray',22:'mediumseagreen',23:'brown',24:'tab:cyan',25:'darkmagenta',26:'beige',27:'coral',28:'lightgreen',29:'skyblue',30:'crimson' }


    fig, ax = plt.subplots(figsize=(4,4))

    for i in range(n_clusters):
        ix = np.where(labels == i)
        plt.scatter(x_tsne[:,0][ix],x_tsne[:,1][ix], c = cdict[i], label = i, s = 20,alpha=0.7,linewidth=1.)

    ax.legend()
    ax.set_title("tSNE represantation of the latent space")

    if wandb_log:
        wandb.log({"tSNE KMeans": wandb.Image(fig)})
    



def plot_features(model,val_dataloader,num_classes,latent_dim,device="cuda:0",plot_tsne=True,wandb_log=True):
    
    #TODO log images to wandb!

    feats=np.array([]).reshape((0,latent_dim))
    labels=np.array([])
    model.eval()
    model.to(device)
    with torch.no_grad():
        c=0 #counter for n_samples
        for x1,x2,y in val_dataloader:
            
            x1=x1.to(device)
            x1_recon,z,h,c=model(x1)
            h=h.cpu().numpy()
            
            
            feats = np.append(feats,h,axis = 0)
            labels= np.append(labels,y.squeeze(),axis=0)
            
            c+=1
            
            
    
    
    if plot_tsne:
        tsne = TSNE(n_components=2,perplexity=50,random_state=42)
        x_feats=tsne.fit_transform(feats)


        #cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray'}
        cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray',10:'tab:red', 11: 'tab:orange', 12: 'tab:green', 13: 'cyan', 14: 'lime', 15:'seagreen',16: 'tab:green',17:'tab:purple',18:'tab:pink',19:'aquamarine',20:'tab:olive',21:'tab:gray',22:'mediumseagreen',23:'brown',24:'tab:cyan',25:'darkmagenta',26:'beige',27:'coral',28:'lightgreen',29:'skyblue',30:'crimson' }

        fig, ax = plt.subplots(figsize=(4,4))

        for i in range(num_classes):
            ix = np.where(labels == i)
            plt.scatter(x_feats[:,0][ix],x_feats[:,1][ix], c = cdict[i], label = i, s = 20,alpha=0.7,linewidth=1.)

        ax.legend()
        ax.set_title("tSNE represantation of the latent space")
    
        if wandb_log:
            wandb.log({"tSNE": wandb.Image(fig)})
    
    
        
    
    plt.show()






def train_epoch(model,train_dataloader,device="cuda:0",optimizer=None,criterion=None,recon_criterion=None,epoch=None):
    loss_tmp=[]
    recon_loss_tmp=[]
    sim_loss_tmp=[]
    
    model.train()
    with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:

        tepoch.set_description(f"Epoch {epoch}")
        for batch in tepoch:
            optimizer.zero_grad()
            x_i,x_j,y=batch

            x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)
            
            # positive pair, with encoding
            x_i_recon,z_i,h_i,c_i= model(x_i)
            x_j_recon,z_j,h_j,c_j = model(x_j)

            
            #compute loss
            sim_loss = criterion(z_i, z_j)
            recon_loss=recon_criterion(x_i_recon,x_i)+recon_criterion(x_j_recon,x_j)
            loss=sim_loss+0.1*recon_loss
            #loss=sim_loss

            loss.backward()

            optimizer.step()

            tepoch.set_postfix({"loss": loss.item(),"recon_loss":recon_loss.item(), "sim_loss":sim_loss.item()})
            loss_tmp.append(loss.item())
            sim_loss_tmp.append(sim_loss.item())
            recon_loss_tmp.append(recon_loss.item())

    return np.mean(loss_tmp),np.mean(sim_loss_tmp),np.mean(recon_loss_tmp)
    

def validation_epoch(model,val_dataloader,device="cuda:0",criterion=None,recon_criterion=None,epoch=None,wandb_log=True):
    loss_tmp=[]
    recon_loss_tmp=[]
    sim_loss_tmp=[]
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(val_dataloader, unit="batch") as tepoch:

            tepoch.set_description(f"Val {epoch}")
            for batch in tepoch:

                x_i,x_j,y=batch

                x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)

                # positive pair, with encoding
                x_i_recon,z_i,h_i,c_i = model(x_i)
                x_j_recon,z_j,h_j,c_j = model(x_j)


                #compute loss
                sim_loss = criterion(z_i, z_j)
                recon_loss=recon_criterion(x_i_recon,x_i)+recon_criterion(x_j_recon,x_j)
                loss=sim_loss+0.1*recon_loss
                
                #loss=sim_loss
            tepoch.set_postfix({"loss": loss.item(),"recon_loss":recon_loss.item(), "sim_loss":sim_loss.item()})
            loss_tmp.append(loss.item())
            sim_loss_tmp.append(sim_loss.item())
            recon_loss_tmp.append(recon_loss.item())

    
    if wandb_log:
        ### save ims to wandb
        x_i=x_i[:8]
        x_i_recon=x_i_recon[:8]

        x_img=torch.cat([x_i,x_i_recon]).cpu()

        #print(x_img.shape)
                
        vis=make_grid(x_img,n_row=2).permute(1,2,0).numpy()*255.
        #vis = build_montages(images, (target_shape[1], target_shape[1]), (10, 10))[0]  #only works with square images!

        log = {f"Image": wandb.Image(vis)}
        wandb.log(log)




    return np.mean(loss_tmp),np.mean(sim_loss_tmp),np.mean(recon_loss_tmp)

def train_epoch_phase2(model,train_dataloader,device="cuda:0",optimizer=None,criterion=None,recon_criterion=None,epoch=None):
    
    
    loss_tmp=[]
    recon_loss_tmp=[]
    sim_loss_tmp=[]
    
    model.train()
    with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:

        tepoch.set_description(f"Epoch {epoch}")
        for batch in tepoch: 
            optimizer.zero_grad()
            x_i,x_j,y=batch

            x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)
            
            # positive pair, with encoding
            x_i_recon,z_i,h_i,c_i = model(x_i)
            x_j_recon,z_j,h_j,c_j= model(x_j)

            
            #compute loss
            sim_loss = criterion(z_i, z_j)
            recon_loss=recon_criterion(x_i_recon,x_i)+recon_criterion(x_j_recon,x_j)
            #loss=sim_loss+0.5*recon_loss
            loss=recon_loss
            
            loss.backward()

            optimizer.step()
            
            model.update_cluster_means(h_i.detach(),c_i.detach())
            
            
            tepoch.set_postfix({"loss": loss.item(),"recon_loss":recon_loss.item(), "sim_loss":sim_loss.item()})
            loss_tmp.append(loss.item())
            sim_loss_tmp.append(sim_loss.item())
            recon_loss_tmp.append(recon_loss.item())

    return np.mean(loss_tmp),np.mean(sim_loss_tmp),np.mean(recon_loss_tmp)




def save(model,outdir,name,optimizer,scheduler):
    
    torch.save(model.state_dict(),opj(outdir,f"model_epoch_{name}.pt"))
    torch.save(model.basis,opj(outdir,f"model_basis_{name}.pt"))
    torch.save(model.kmeans,opj(outdir,f"model_kmeans_{name}.pt"))

    torch.save(optimizer.state_dict(),opj(outdir,f"optimizer_{name}.pt"))
    torch.save(scheduler.state_dict(),opj(outdir,f"optimizer_{name}.pt"))
    
    


def get_pneumonia_array():
    data_flag = f'pneumoniamnist'
    # data_flag = 'breastmnist'
    download = True


    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])






    train_dataset = DataClass(split='train', transform=None, download=download)
    val_dataset = DataClass(split='val', transform=None, download=download)
    test_dataset = DataClass(split='test', transform=None, download=download)

    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    for x,y in tqdm.tqdm(train_dataset):
        x_train.append(x)
        y_train.append(y)


    for x,y in tqdm.tqdm(val_dataset):
        x_val.append(x)
        y_val.append(y)

    for x,y in tqdm.tqdm(test_dataset):
        x_test.append(x)
        y_test.append(y)



    print(len(x_train),len(y_train),len(x_val),len(y_val),len(x_test),len(y_test))
    
    return x_train,y_train,x_val,y_val,x_test,y_test
    

def get_mnist_array():
    train_dataset=MNIST('mnist_train', train=True, download=True,
                                transform=None)

    val_dataset=MNIST('mnist_test', train=False, download=True,
                                transform=None)


    test_dataset=MNIST('mnist_test', train=False, download=True,
                                transform=None)



    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    for x,y in tqdm.tqdm(train_dataset):
        x_train.append(x)
        y_train.append(y)


    for x,y in tqdm.tqdm(val_dataset):
        x_val.append(x)
        y_val.append(y)

    for x,y in tqdm.tqdm(test_dataset):
        x_test.append(x)
        y_test.append(y)



    print(len(x_train),len(y_train),len(x_val),len(y_val),len(x_test),len(y_test))
    
    return x_train,y_train,x_val,y_val,x_test,y_test

