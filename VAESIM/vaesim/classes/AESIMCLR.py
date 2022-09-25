
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
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd





## TODO: Better division between "hard" and "soft" labels (kmeans or soft similarity)

class Linear(torch.nn.Module):
    def __init__(self,n_out=10):
        super().__init__()
        self.model=torch.nn.Sequential(torch.nn.LazyLinear(n_out))
        
    def forward(self,x):
        return self.model(x)




class ProjectionHead(nn.Module):
    def __init__(self,latent_dim,hidden_dim,out_features):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(latent_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,out_features))
        
    def forward(self,x):
        return self.model(x)

class AESIMCLR(nn.Module):
    def __init__(self,latent_dim,encoder,decoder,n_basis=10):
        super().__init__()
        self.encoder = encoder
        self.decoder= decoder
        self.n_basis=n_basis
        
        self.projector = ProjectionHead(latent_dim, 2*latent_dim,latent_dim)
        self.warmup=True
        
        self.basis=None
        self.wandb_log=False
        self.temperature=1.
        self.latent_dim=latent_dim
        self.kmeans=None
        self.dist=None
        self.mappings=None
        


    def init_basis_from_kmeans_centroids(self,centroids,visualize=True):
        
    

        self.basis=torch.zeros(self.latent_dim,self.n_basis)
        
        for (i,c) in enumerate(centroids):
            self.basis[:,i]=torch.Tensor(c)

        if visualize:
            
            
            fig,ax=plt.subplots()
            print(f"Computing t-SNE to visualize from {self.latent_dim} to 2 dim - this could take a while..")
            tsne=TSNE(n_components=2, perplexity=50,random_state=42,learning_rate='auto',init='random').fit_transform(self.basis.T)
            ax.scatter(tsne[:,0],tsne[:,1])
            
            if self.wandb_log:
                wandb.log({"init": wandb.Image(ax)})
            
            plt.show()
        
        
    def init_basis(self,z,visualize=True):
        """Init the basis, used in the first forward pass
        
        The columns of the B (basis) matrix are updated as random latent representations
        
        @param z: Tensor, latent vectors
        @param visualize: bool, if True visualize tSNE representations of centroids
        
        """
        
        
        #pick n_basis random idx and set their rapresentations as columns
        
        idxs=torch.randint(0,z.shape[0],(self.n_basis,))
        
        self.basis=torch.zeros(self.latent_dim,self.n_basis)
        for i in range(self.n_basis):
            self.basis[:,i]=z[idxs[i]]
        #self.basis=torch.rand((self.latent_dim,self.n_basis))
        
        
        
        if visualize:
            
            
            fig,ax=plt.subplots()
            print(f"Computing t-SNE to visualize from {self.latent_dim} to 2 dim - this could take a while..")
            tsne=TSNE(n_components=2, perplexity=50,random_state=42,learning_rate='auto',init='random').fit_transform(self.basis.T)
            ax.scatter(tsne[:,0],tsne[:,1])
            
            if self.wandb_log:
                wandb.log({"init": wandb.Image(ax)})
            
            plt.show()
            
            
    def init_basis_kmeans(self,trainloader,l):
        """
        Initialize the basis using the kmeans algorithm
        
        """
        
        h_semi=np.array([])


        with torch.no_grad():
            for x_i,x_j,y in tqdm.tqdm(trainloader):

                x_i,x_j,y=x_i.to(device),x_j.to(device),y.to(device)
                x_i_recon,z_i,h_i,c_i = model(x_i)

                h_semi=np.append(h_semi,h_i.cpu().numpy())
        
        
        h_semi=h_semi.reshape((l,-1))
        
        kmodel = KMeans(n_clusters=self.n_basis)
        kmodel.fit(h_semi)
        basis=torch.Tensor(kmodel.cluster_centers_).T
        self.basis=basis
        
        
    def compute_similarity(self,z):
        """
        Compute the similarity between the latent representations z and the prototypes matrix B
        compute the similarity between each sample and the mean vector of the cluster
        
        @param z: Tensor, latent reprs
        
        return similarity, Tensor with shape BS,K
        """

        if self.basis is None:
            self.init_basis(z.detach())
        sim= z@self.basis.type_as(z)

        return sim
        
        

    def forward(self,x):
        h = self.encoder(x)
        
        zp = self.projector(torch.squeeze(h))
        
        if self.warmup:
            c=torch.ones((h.shape[0],self.n_basis))/self.n_basis
            
            xr=self.decoder(h)
            
        else:
            sim=self.compute_similarity(h)
            c=softmax(sim/self.temperature,dim=1)
        
            c=c.to(h.get_device())
            
            xr= self.decoder(h,c)

            ##update means?
        
        return xr,zp,h,c

    
    
    
    
    
    #### EVALUATION FUNCTION BELOW HERE
    
    def clusterize(self,soft_sim):
        
        """
        clustering function
        
        @param: soft_sim: Tensor, softmax of the similarity
        
        
        return cl: Tensor with shape (BS,)
        """
        
        cl=soft_sim.argmax(dim=1)
            
        return cl
    
    
    
    
    
    
    def get_similarity_matrix(self):
        B=self.normalize_basis()
        return (B.T@B)
    
    def normalize_basis(self):
        """Normalize each prototype by its mean"""
        
        N=torch.norm(self.basis,dim=0)

        B=torch.zeros_like(self.basis)
        for i in range(self.basis.shape[1]):
            B[:,i]=self.basis[:,i]/N[i]
        return B
    
    
    def update_cluster_means(self,z,soft_sim):
        
        """This is the most important function in the code, update the values of the columns of B matrix
        @param z: Tensor, latent reprs
        @param soft_sim: Tensor, softmax of similarity
        
        1) Get the cluster
        2) For each cluster compute the mean of the vectors that belongs to it
        3) If reinit -> montecarlo sampling and reinitialize the prototypes
        
        return cl: Tensor
        
        """
            
        ##map each vector into his cluster
        
        
        #cl=self.clusterize(soft_sim)
        
        cl=soft_sim.argmax(-1)
        
        #update the basis matrix columns according to the mean of the vector that belongs to that cluster
        for i in range(self.n_basis):
            
             #Here I check there is no NaN in the mean vector
            if not torch.isnan(z[cl==i].mean(axis=0)).any():
                
                new_mean=z[cl==i].mean(axis=0)
                
                self.basis[:,i]=new_mean
                    
                
                
        
        return cl
    
    def mapping(self,cl,y):
        """map[k]=label so we map the cluster k-th to the label"""

        y=y.squeeze()
        map={}

        acc={}

        for k in range(self.n_basis):
            if len(y[cl==k]):
                v=y[cl==k].mode().values
                map[k]=v.item()
                a=(y[cl==k]==v).sum()/len(y[cl==k]) 
                acc[k]=a.item()
    
    
        #fix missing keys
        
        val,counts=np.unique(list(map.values()),return_counts=True)
        mode=val[np.argmax(counts)]
        
        map=defaultdict(lambda:mode,map)
        
        return map,acc
        
        
    def compute_mappings(self,train_dataloader,n_batch=30,label="soft"):
        
        model=self
        device=self.device
        i=0
        with torch.no_grad():




            stack=[]
            labels=[]
            z_list=[]
            h_list=[]
            cl_list=[]
            k_list=[]
            
            for data in tqdm.tqdm(train_dataloader):
                x,y=data
                #stack.append(x)
                labels.append(y)
                h=model.encoder(x.to(device))
                z=model.projector(torch.squeeze(h))
                #z_mean,z_log_var = model.encoder(x.to(device))
                #model.q_z = D.normal.Normal(z_mean, torch.exp(0.5 * z_log_var))
                #device=z_mean.get_device()
                #z = model.q_z.rsample()
                
                sim=model.compute_similarity(z)
                soft_sim=softmax(sim,dim=1)
                cl=model.clusterize(soft_sim)
                hh=np.array(h.cpu().numpy(), dtype=np.double)
                k=model.kmeans.predict(hh)
                
                
                
                z_list.append(z)
                h_list.append(h)
                cl_list.append(cl)
                k_list.append(k)
                i+=1
                if i==(n_batch-1):
                    break

            

            

            
            z=torch.cat(z_list)
            h=torch.cat(h_list)
            cl=torch.cat(cl_list)
            labels=torch.cat(labels)
            kl=np.concatenate(k_list)

            
            
           
            
            




            ##map each vector into his cluster
            #cl=sim.argmax(-1)
        print(f"Computing mappings using {len(cl)} samples from the training set")
        
        if label=="soft":
            print(f"[INFO] computing mappings using argmax of soft label")
            mappings,accuracy=self.mapping(cl,labels)
        
        elif label=="hard":
            print(f"[INFO] computing mappings using kmeans prediction")
            mappings,accuracy=self.mapping(kl,labels)
            
        else:
            print(f"[ERROR] please specicify soft or hard as label argument")
        
        return mappings,accuracy,z.cpu(),h.cpu(),cl.cpu(),kl,labels.cpu()
    
    
    def compute_report(self,test_dataloader,mode="standard",label="soft"):
        
        model=self
        device=self.device
        
        if mode=="hierarchical":
            mappings=self.hierarchical_mappings
        else:
            mappings=self.mappings
        
        ## classification report
        recon=[]
        y_pred=[]
        labels=[]
        z_sampled=[]
        h_sampled=[]
        clusters=[]
        k_clusters=[]
        
        with torch.no_grad():
            for x,y in tqdm.tqdm(test_dataloader):


                x_rec,z,h,sim=model(x.to(device))
                cluster=sim.argmax(dim=1)
                hh=np.array(h.cpu().numpy(), dtype=np.double)
                k=model.kmeans.predict(hh)
                #cluster=model.clusterize(soft_sim)

                if label=="soft":
                    mapped=[mappings[cluster[i].cpu().item()] for i in range(len(cluster))]
                elif label=="hard":
                    mapped=[mappings[k[i]] for i in range(len(k))]

                y_pred.append(mapped)

                clusters.append(cluster.cpu())
                k_clusters.append(k)
                labels.append(y)
                z_sampled.append(z.detach().cpu())
                h_sampled.append(h.detach().cpu())                                                                         

                                                                                          

        
        
        
        y_pred=[item for sublist in y_pred for item in sublist]
        
        
        #y_pred=torch.cat(y_pred)
        labels=torch.cat(labels).numpy()
        z_sampled=torch.cat(z_sampled).numpy()
        h_sampled=torch.cat(h_sampled).numpy()
        clusters=torch.cat(clusters).numpy()
        k_clusters=np.concatenate(k_clusters)
        #z_sampled=np.array(z_sampled)
        
        
        
        print(f"Computing metrics using {len(clusters)} samples from the test set")    

        class_report=classification_report(labels, y_pred)
        class_report_dict=classification_report(labels, y_pred,output_dict=True)
        df = pd.DataFrame(class_report_dict).transpose()
        accuracy=accuracy_score(labels,y_pred)
        
        return class_report,z_sampled,h_sampled,labels,clusters,k_clusters,accuracy,df
        
        
        
    def compute_tsne(self,z_sampled,labels):
        tsne=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(np.squeeze(z_sampled))
         
        cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray',10:'tab:red', 11: 'tab:orange', 12: 'tab:green', 13: 'cyan', 14: 'lime', 15:'seagreen',16: 'tab:green',17:'tab:purple',18:'tab:pink',19:'aquamarine',20:'tab:olive',21:'tab:gray',22:'mediumseagreen',23:'brown',24:'tab:cyan',25:'darkmagenta',26:'beige',27:'coral',28:'lightgreen',29:'skyblue',30:'crimson' }

        fig, ax = plt.subplots(figsize=(10,10))
        for g in np.unique(labels):
            ix = np.where(labels == g)[0]
            
            if len(ix):
                ax.scatter(tsne[:,0][ix], tsne[:,1][ix], c = cdict[g], label = g, s =20,alpha=0.7,linewidth=1.)
        #plt.scatter(tsne[:,0],tsne[:,1],label=labels,c= cdict[g])
        ax.legend()
        ax.set_title("tSNE represantation of the latent space")
        plt.show()
        return ax

    
    
    def cluster_metrics(self,z_sampled,clusters,labels):
        
        print(z_sampled.shape,clusters.shape,labels.shape)
        
        pred_labels=[self.mappings[i] for i in clusters]

        try:
            sil_score=silhouette_score(z_sampled.squeeze(),clusters.squeeze())
            sil_score_mappings=silhouette_score(z_sampled.squeeze(),pred_labels) ## silhouette score with labels

        except:
            print("Can't compute silhouette score")
            sil_score=0
            sil_score_mappings=0
            
        try:
            di_score=davies_bouldin_score(z_sampled.squeeze(),clusters.squeeze())
        except:
            print("Can't compute silhouette score")
            di_score=0
        
        try:
            nmi_score=normalized_mutual_info_score(pred_labels,labels)
        except: 
            nmi_score=None
        
        
        
        return sil_score,di_score,nmi_score,sil_score_mappings
    
    def correlation_matrix(self):
        
        hs=self.get_similarity_matrix()
        #fig, ax = plt.subplots(figsize=(15,12))
        h=sns.clustermap(hs, method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12))
        

        return h._figure,h
    
    
    
    def compute_tsne_training(self,z_sampled_train,h_sampled_train,cl_train,kl_train,labels_train,zoom=1.,threedimensional=False,label="soft"):
        
        labels_train=labels_train.squeeze()
        if label=="soft":
            kl_mapped=[self.mappings[i] for i in cl_train]
            cluster_labels=cl_train.squeeze()
        elif label=="hard":
            kl_mapped=[self.mappings[i] for i in kl_train]
            cluster_labels=kl_train
        else:
            print(f"[WARNING] Please specify either soft of hard as label parameter")
                
        print(f"[INFO] Mapped using {label} approach found {set(kl_mapped)}")
                
        plt.rcParams['axes.grid'] = False
        plt.rcParams['xtick.bottom']=False
        plt.rcParams['xtick.labelbottom']=False

        plt.rcParams['ytick.left']=False
        plt.rcParams['ytick.labelleft']=False
        plt.rcParams['font.size']=24

        tsne=TSNE(n_components=2,learning_rate='auto',init='random')
        tsne=tsne.fit_transform(np.squeeze(z_sampled_train.squeeze()))
                         #tsne=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(np.squeeze(z_sampled_train.squeeze()))

        cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray',10:'tab:red', 11: 'tab:orange', 12: 'tab:green', 13: 'cyan', 14: 'lime', 15:'seagreen',16: 'tab:green',17:'tab:purple',18:'tab:pink',19:'aquamarine',20:'tab:olive',21:'tab:gray',22:'mediumseagreen',23:'brown',24:'tab:cyan',25:'darkmagenta',26:'beige',27:'coral',28:'lightgreen',29:'skyblue',30:'crimson' }

        fig, axs = plt.subplots(1,3,figsize=(30,10))

        #print(f"[DEBUG] {tsne.shape}, {labels_train.shape}")
        sns.scatterplot(x=tsne[:,0], y=tsne[:,1],hue=labels_train,ax=axs[0],legend="full",palette=cdict,alpha=0.7,linewidth=0.)

        #for g in np.unique(labels_train):
        #    ix = np.where(labels_train == g)[0]
        #    axs[0].scatter(tsne[:,0][ix], tsne[:,1][ix], c = cdict[g], label = g, s =20,alpha=0.7,linewidth=1.)
        #plt.scatter(tsne[:,0],tsne[:,1],label=labels,c= cdict[g])
        axs[0].legend()
        axs[0].set_title("Real labels",fontsize=24)
        axs[1].set_title("Cluster labels",fontsize=24)
        axs[2].set_title("Mapped labels",fontsize=24)


        palette=sns.color_palette("Paired",n_colors=self.n_basis)
        
        sns.scatterplot(x=tsne[:,0], y=tsne[:,1],hue=kl_train,ax=axs[1],legend="full",palette=palette,alpha=0.7,linewidth=0.,edgecolors='none')

        sns.scatterplot(x=tsne[:,0], y=tsne[:,1],hue=kl_mapped,ax=axs[2],legend="full",palette=cdict,alpha=0.7,linewidth=0.,edgecolors='none')


        
        self.plot_means_over_tsne(axs[1],tsne,h_sampled_train,cluster_labels,zoom=zoom,threedimensional=threedimensional)
        #axs[1].legend()
        #axs[1].set_title("tSNE represantation of the latent space - Training set with cluster labels")
        
        
        
        plt.show()
        return fig
    
    
    def plot_means_over_tsne(self,ax,tsne,h_sampled_train,cluster_labels,zoom=1.,threedimensional=False):
        device=self.device
        model=self
        recons=[]
        recons_label=[]
        for c in range(len(set(cluster_labels))):
            h_valid=h_sampled_train[cluster_labels==c]
            h_mean=h_valid.mean(dim=0)
            sim=model.compute_similarity(h_mean.unsqueeze(0))
            soft_sim=softmax(sim,dim=1)
            with torch.no_grad():
                x_recon=model.decoder(h_mean.to(device),soft_sim.to(device))
            recons.append(x_recon.cpu().numpy())
            recons_label.append(self.mappings[c])
        
        recons=torch.Tensor(np.array(recons))
    
        if threedimensional:  ##handle 3d images taking axial slice
            half=x_recon.shape[-1]//2
            recons=recons[:,:,:,:,half].squeeze()

        
        if len(recons.shape)>4:
            ##removing extra channel if exist
            recons=recons.squeeze(1)

        for g in np.unique(cluster_labels):
            ix = np.where(cluster_labels == g)[0]
            x0=np.mean(tsne[:,0][ix])
            y0=np.mean(tsne[:,1][ix])

            if threedimensional:
                img=recons[g].cpu()
            else:
                img=recons[g].permute(1,2,0).cpu()
                if img.shape[-1]==1:
                    img=img.squeeze()

            ##check also if image is grayscale


            off_img=OffsetImage(img,zoom=zoom)

            ab = AnnotationBbox(off_img, (x0, y0), frameon=False)
            ax.add_artist(ab)

    
    
    def plot_prototypes_over_tsne(self,ax,tsne,cl_train,zoom=1.,threedimensional=False,label="soft"):
        
        
        
        ##find coordinates
        device=self.device
        z=self.basis.T.to(device)
        sim=self.compute_similarity(z)
        soft_sim=softmax(sim,dim=1)
        
        
        with torch.no_grad():
            x_recon=self.decoder(z,soft_sim)

        if threedimensional:  ##handle 3d images taking axial slice
            half=x_recon.shape[-1]//2
            x_recon=x_recon[:,:,:,:,half].squeeze()
        
        
        for g in np.unique(cl_train):
            ix = np.where(cl_train == g)[0]
            x0=np.mean(tsne[:,0][ix])
            y0=np.mean(tsne[:,1][ix])
            
            if threedimensional:
                img=x_recon[g].cpu()
            else:
                img=x_recon[g].permute(1,2,0).cpu()
                if img.shape[-1]==1:
                    img=img.squeeze()
                
            ##check also if image is grayscale


            off_img=OffsetImage(img,zoom=zoom)

            ab = AnnotationBbox(off_img, (x0, y0), frameon=False)
            ax.add_artist(ab)

            
        
    
    def hiearchical_clustering(self,h):
        
        dendro=h.dendrogram_col.linkage
        double=dendro[dendro[:,-1]==2]
        to_reduce=double[double[:,2]<np.median(double[:,2])]
        
        #lavorare qua
        
        to_reduce[:,0:2]

        d={}
        for i in range(self.n_basis):
            if i not in np.unique(to_reduce[:,0:1]):
                d[i]=i


        d.update(dict(to_reduce[:,0:2].astype("int")))
        d=dict(sorted(d.items()))                          ###mapping of hiearchical clustering
        
        l=len(np.unique(list(d.values())))
        print(f"Merging the {len(to_reduce)} most similar cluster of two elements reduced the number of basis from {self.n_basis} to {l}")
        
        
        return d
        
        
    def compute_hierarchical_mappings(self,cl_train,labels_train,d):
        
        #debug
        
        
        cl_hiearchical_train=[d[i] for i in cl_train.numpy()]
        
        self.hierarchical_mappings,acc=self.mapping(cl_train,labels_train)
        
        return self.hierarchical_mappings
        

        
    def compute_distribution_for_sampling(self,train_dataloader):
        
        device=self.device
        model=self
        
        z_train=[]
        h_train=[]
        cl_train=[]
        y_train_labels=[]

        with torch.no_grad():
            for x_i,y in tqdm.tqdm(train_dataloader):
                x_i,y=x_i.to(device),y.to(device)
                x_i_recon,z_i,h_i,c_i = model(x_i)
                z_train+=z_i.cpu().numpy().tolist()
                h_train+=h_i.cpu().numpy().tolist()
                y_train_labels+=y.cpu().numpy().tolist()

        
        mu=torch.Tensor(np.array(h_train).mean(axis=0))
        std=torch.Tensor(np.array(h_train).std(axis=0))

        self.dist = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(std))

        if self.mappings is None:
            print(f"[INFO] Computing mappings between clusters labels and real labels by mode")
            self.mappings,acc=self.mapping(cl_train,y_train_labels)
            
        
    def sample_latent(self,n_of_samples):
        if self.dist is None:
            print(f"[WARNING] Please run compute_distribution_for_sampling before to get the distribution of your data inside the latent space")
        
        h_synth=self.dist.sample((n_of_samples,))
        sim=self.compute_similarity(h_synth)
        soft_sim=softmax(sim/self.temperature,dim=1)
        c=sim.argmax(-1)
        y_synth=[self.mappings[i.item()].item() for i in c]
        
        return h_synth,soft_sim,y_synth
        
    def sample_imgs(self,n_of_samples):
        
        h_synth,soft_sim,y_synth=self.sample_latent(n_of_samples)
        
        h_synth,soft_sim=h_synth.to(self.device),soft_sim.to(self.device)
        x_i=self.decoder(h_synth,soft_sim)
        
        return x_i,y_synth
        

    
    
    def evaluate(self,batch,train_dataloader,test_dataloader,wandb_log=True,n_semi=50,zoom=1.,threedimensional=False,label="soft",neighbours=5,lin_epochs=100):
        
        """This function evaluate a lot of things about the clustering proprierties.
        
        parameters:
        
        batch: a batch from training set
        train_dataset: Pytorch Dataset
        test_dataset: Pytorch Dataset
        n_semi: int, number of labelled used for semi-supervised learning
        threedimensional: bool, if True the model is used on 3D images -> so prototypes are "axial" reprentations of those 3D volumes
        
        
        
        ### EVALUATION
        
        1. correlation_matrix() -> ax,h
            ax is a heatmap figure of correlation between prototypes of basis matrix
            h is a GridCluster instance. It contains information about hiearchical clustering based on similarity matrix
            
        2. hiearhical
        
        3. label distribution of batch elements among different clusters.
        
        4. mappings, requires the training dataset and n_semi. It evaluate the cluster assignment taking the mode of the label distribution for each cluster
            it returns the mappings (dict), the frequency of the chosen class for each cluster (dict), z_sampled_train (np.array) an array of latent representation of part of training set, cl_train (np.array) cluster assignemnet for each sample, labels_train                   (np.array) the real labels for those sampls
        
        5. Report. Take the test dataset and use the mappings to evaluate the accuracy of this unsupervised (or semisupervised clustering)
            it returns class report and z_sampled, cluster assignemtn and labels for test set
        
        6. Clustering metrics, take z_sampled, cluster and labels (test set)
            it returns silhouette score, dunn index and normalized mutual information
            
        7. Compute tsne training and compute tsne -> compute the tSNE representations. 
        
        """
        
        
        
        
        model=self
        device=self.device
        
        x,y = batch
        x2,z,h,s=model(x.to(device))
        #x2,z,s=model(x.to(device))
        cl=s.argmax(dim=1)
        
        
        
        
        
        ### 1. CORRELATION MATRIX AND HIERARCHICAL
        
        ax,h=self.correlation_matrix()
        
        if wandb_log:
            wandb.log({"Correlation Matrix": wandb.Image(ax)})
            
            
        ##2. HIEARCHICAL CLUSTERING
        
        d=self.hiearchical_clustering(h)
        
        
        
        
        ##3. LABEL DISTRIBUTION
        fig,ax=plt.subplots()
        ax.hist(cl.detach().cpu().numpy())
        ax.set_title("Distribution of samples over cluster")
        
        if self.n_basis>5:
            nb=self.n_basis//5
        else:
            nb=1
            
        figs,axs=plt.subplots(5,nb,figsize=(30,30))

        for (i,ax) in enumerate(axs.ravel()):
            dist=y[cl==i].cpu().numpy()
            ax.hist(dist)
            ax.set_title(f"Labels over cluster {i}")
        
        if wandb_log:
            wandb.log({"Distribution over cluster": wandb.Image(fig), "Label distributions":wandb.Image(figs)})
            
        
        ## COMPUTE MAPPINGS ON PART OF TRAINING SET
        mappings,acc,z_sampled_train,h_sampled_train,cl_train,kl_train,labels_train=self.compute_mappings(train_dataloader,n_semi,label)
        
        self.mappings=mappings
        
            
        frequencies=[(k,v) for k,v in acc.items()]
        if wandb_log:
             wandb.log({"Freq": frequencies})
                     
        print(acc)
        print(mappings)
        
        
        
        hiearchical_mappings=self.compute_hierarchical_mappings(cl_train,labels_train,d)
        
        
       
        
        ### ALL THESE METRICS ARE COMPUTED ON TEST SET
        
        print("Computing classification report for hierarchical clustering")
        class_report_hier,z_sampled_hier,h_sampled_hier,labels_hier,clusters_hier,k_clusters_hier,accuracy_hier,df_hier=self.compute_report(test_dataloader,mode="hierarchical",label=label)
        print(class_report_hier)
        
        
        
        print("Computing classification report for standard clustering")
        class_report,z_sampled,h_sampled,labels,clusters,k_clusters,accuracy,df=self.compute_report(test_dataloader,label=label)
        print(class_report)
        
        ### CLUSTERING METRICS
        sil_score,di_score,nmi_score,sil_labels=self.cluster_metrics(z_sampled,clusters,labels)
        
        print(f"Silhouette score:\t{sil_score}\nDunn Index:\t{di_score}\nNMI score:\t{nmi_score}\nSilhouette Score Mapped:\t{sil_labels}")
        if wandb_log:
            wandb.log({"Silhouette Score": sil_score, "Dunn Index":di_score, "NMI":nmi_score,"Silhouette Score Mapped": sil_labels})
        
        
        ##COMPUTE kNN and linear approaches
        print(f"[INFO] Running evaluations from k Nearest Neighbour with k= {neighbours} on z space")
        
        neigh = KNeighborsClassifier(neighbours)
        neigh.fit(z_sampled_train,labels_train)
        y_pred_neigh=neigh.predict(z_sampled)
        
        
        class_report, df, neigh_accuracy_z=just_report(labels,y_pred_neigh)
        print(class_report)
        if wandb_log:
            wandb.log({"kNN (z) Report": df, "kNN (z) Accuracy":neigh_accuracy_z})
            
        
        print(f"[INFO] Running evaluations from k Nearest Neighbour with k= {neighbours} on h space")
        
        neigh = KNeighborsClassifier(neighbours)
        neigh.fit(h_sampled_train,labels_train)
        y_pred_neigh=neigh.predict(h_sampled)
        
        
        class_report, df, neigh_accuracy_h=just_report(labels,y_pred_neigh)
        print(class_report)
        if wandb_log:
            wandb.log({"kNN (h) Report": df, "kNN (h) Accuracy":neigh_accuracy_h})
        
        
        
        #### Compute linear
        
        
                ## LINEAR CLASSIFIER
        print(f"[INFO] Running evaluations from with linear classifier on z space")
        
        lin_z,z_loss,z_acc=evaluate_linear(z_sampled_train,labels_train,epochs=lin_epochs)
        y_z_pred,y_z_true=test_linear(lin_z,z_sampled,labels)

        fig,axs=plt.subplots(1,2,figsize=(6,6))
        axs[0].plot(z_loss,label="z loss")
        axs[0].set_title("Linear training loss")
        axs[1].plot(z_acc,label="z loss")
        axs[1].set_title("Linear training acc")
        
        class_report, df, lin_accuracy_z=just_report(y_z_true,y_z_pred)
        print(class_report)
        if wandb_log:
            wandb.log({"Linear Report (z)": df, "Linear Accuracy (z)":lin_accuracy_z,"Linear layer (z)": wandb.Image(fig)})
        
        
        
        lin_z,z_loss,z_acc=evaluate_linear(h_sampled_train,labels_train,epochs=lin_epochs)
        y_z_pred,y_z_true=test_linear(lin_z,h_sampled,labels)

        fig,axs=plt.subplots(1,2,figsize=(6,6))
        axs[0].plot(z_loss,label="z loss")
        axs[0].set_title("Linear training loss")
        axs[1].plot(z_acc,label="z loss")
        axs[1].set_title("Linear training acc")
        
        class_report, df, lin_accuracy_h=just_report(y_z_true,y_z_pred)
        print(class_report)
        if wandb_log:
            wandb.log({"Linear Report (h)": df, "Linear Accuracy (h)":lin_accuracy_h,"Linear layer (h)": wandb.Image(fig)})
        
        
        
        ## TSNE    

        fig=self.compute_tsne_training(z_sampled_train,h_sampled_train,cl_train,kl_train,labels_train,threedimensional=threedimensional,label=label)
       
        
        if wandb_log:
            wandb.log({"Training latent space": wandb.Image(fig)})
            
        
            

        
        ax=self.compute_tsne(z_sampled,labels)
        
        if wandb_log:
            wandb.log({"Report": df, "Accuracy":accuracy,"Latent space": wandb.Image(ax)})

        return accuracy,neigh_accuracy_z,neigh_accuracy_h,lin_accuracy_z,lin_accuracy_h

class encoder(nn.Module):
    
    def __init__(self,latent_dim=50,n_conv=3,n_init_filters=32,input_channels=1):
        super().__init__()
        layers=[]
        for i in range(n_conv):
            if i==0:
                layers.append(nn.Conv2d(input_channels,n_init_filters,kernel_size=4,stride=2,padding=1))
                layers.append(nn.GELU())
                layers.append(nn.BatchNorm2d(n_init_filters))
            else:
                layers.append(nn.Conv2d(n_init_filters*(2**(i-1)),n_init_filters*2**i,kernel_size=4,stride=2,padding=1))
                layers.append(nn.GELU())
                layers.append(nn.BatchNorm2d(n_init_filters*2**i))

        layers.append(nn.AvgPool2d(kernel_size=2))
        layers.append(nn.Flatten())
        layers.append(nn.LazyLinear(latent_dim))
        
        self.network=nn.Sequential(*layers)
        
    def forward(self,x):
        return self.network(x)
        
        

class Decoder(nn.Module):
    def __init__(self,latent_dim,conv_filters,target_dim=(1,32,32),kernel_size=4):
        
        super().__init__()
        

        start_dim=int(target_dim[-1]/2**(len(conv_filters)))

        
        l=len(conv_filters)
        self.layers=l
        self.start_dim=start_dim
        self.init_channels=target_dim[0]


        conv=[]
        conv_filters=[self.init_channels]+conv_filters
        
        #conv.append()
        #conv.append)
        
        self.predecoder=nn.Linear(latent_dim,start_dim**2)
        self.unflatten=nn.Unflatten(-1,(1,start_dim,start_dim))

        for i in range(l):
            conv.append(ConvTransposeBlock(conv_filters[i],conv_filters[i+1],kernel_size))
        
        
        
        self.features=nn.Sequential(*conv)
        self.decoder_output=nn.LazyConvTranspose2d(self.init_channels,kernel_size=3,padding=1)
        self.activation=nn.Sigmoid()
                        
    def forward(self,x):
        x=self.predecoder(x)
        x=self.unflatten(x)
        x=self.features(x)
        x=self.decoder_output(x)
        x=self.activation(x)
        return x
    


class cDecoder(nn.Module):
    def __init__(self,unconditional_decoder,condition_dim):
        super().__init__()
        self.decoder=unconditional_decoder
        self.layers=unconditional_decoder.layers
        self.start_dim=unconditional_decoder.start_dim

        self.condition_dim=condition_dim

        self.condition =  nn.Linear(self.condition_dim,self.start_dim*self.start_dim)
        self.condition2shape = nn.Unflatten(1, (1,self.start_dim , self.start_dim))

        self.conv_mix=nn.LazyConv2d(self.decoder.init_channels,kernel_size=3,padding=1)
        

    def forward(self,x,c):
        x=self.decoder.predecoder(x)
        x=self.decoder.unflatten(x)

        c= self.condition(c)
        c= self.condition2shape(c)
            
            #x= torch.concat((x,c),axis=1)
        x = x.view(x.shape[0], -1, self.start_dim, self.start_dim)
        x = self.conv_mix(x)

        x=self.decoder.features(x)
        x=self.decoder.decoder_output(x)
        x=self.decoder.activation(x)

        return x







class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=2,padding=1,separable=False):
        super().__init__()
        
        self.out_channels=out_channels
        
        if separable:
            self.conv=DepthSepConv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        else:
            self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.norm=nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
    
    
class ConvTransposeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=2,padding=1,separable=False):
        super().__init__()
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        if separable:
            self.conv=DepthSepConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        else:
            self.conv=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.norm=nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x