import numpy as np
from torch import nn
import torch.distributions as D

from classes.Architectures import VAEDecoder, VAEEncoder, Discriminator,cVAEDecoder
import tqdm
from utils.callbacks import *
from sklearn.manifold import TSNE
from torch.nn.functional import softmax

from sklearn.metrics import classification_report,accuracy_score,silhouette_score,davies_bouldin_score,normalized_mutual_info_score
import seaborn as sns
import pandas as pd
import torchextractor as tx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier


class Linear(torch.nn.Module):
    def __init__(self,n_out=10):
        super().__init__()
        self.model=torch.nn.Sequential(torch.nn.LazyLinear(n_out))
        
    def forward(self,x):
        return self.model(x)
        



class VAESIM(nn.Module):
    """
    Pytorch implementation of VAESIM
    
    The idea is to learn jointly reconstruction and clustering in the latent space of a VAE implicitly
    B is the prototypes matrix with Q,K where Q is the dimension of the latent space and K the number of cluster.
    Ideally this should be the matrix for basis, whose columns are the prototypes of the clusters
    
    z= e(x)
    sim=z@B
    soft_sim=softmax(sim/temperature)
    x2=d(x,soft_sim)
    
    loss=recon_loss+kl_loss+sim_loss
    -> update B columns in the following way
    
    cl=argmax(soft_sim) #get the cluster
    
    for each cluster the new column is computed as the mean of the values.
    
    
    """


    def __init__(self, input_dim, latent_dim,encoder,decoder,n_basis=10,weight=None,sim_weight=1.,similarity=False,kl=False,sample_cluster=False,temperature=0.01,reinit=0.,schedule=True,ema=None,show=True):
        """

        :param input_dim: dimension of images
        :param latent_dim: latent_dim

        Attributes
        ----------
        total_loss_tracker: mean of the sum of reconstruction_loss and kl_loss
        reconstruction_loss: mean metrics that are L2 norm between input and outputs
        encoder_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..]
        decoder_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..]
        kl_loss: regularizer loss
        similarity: bool, if True compute also the similarity loss
        kl: bool, if True it computes also the KL divergence -> regularize the latent space
        weight: value used to enforce latent regularization, if None it's computed as latent_dim/prod(input_dim)
        sim_weight: float, value used to weight the similarity loss when used
        sample_cluster: bool, if True the cluster assignment comes from sampling a categorical distribution
        temperature: float, temperature parameter
        reinit: float, probability of reinit the cluster centroids
        schedule: bool, if True apply annealing on temperature

        """
        super().__init__()

        self.block=not show #if show is true block is false and the other way around
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        
        self.encoder = encoder
        self.decoder = decoder
    
        if weight is None:
            self.weight=latent_dim/np.prod(input_dim)
        else:
            self.weight=weight
        
        self.recon_loss=nn.MSELoss()
        self.sim_loss=nn.MSELoss()
        self.loss=None

        self.patience=0 #for early stopping
        self.device=None

        self.n_basis=n_basis
        self.spacing=0.2
        self.basis_mean=torch.arange(0,n_basis)*self.spacing
        
        self.basis=None
        self.similarity=similarity
        self.kl=kl
        self.sample_cluster=sample_cluster
        self.T0=temperature
        self.temperature=temperature
        self.schedule=schedule
        self.ema=ema
        
        if reinit==0.:
            self.reinit=False
        else:
            self.reinit=reinit
        
        self.wandb_log=False
        
        print(f"[INFO] Running model with KL: {kl}, similarity loss: {similarity} and sampling: {sample_cluster} temperature: {temperature} reinitilize: {reinit} ema: {ema} ")
        
        
    def save_model(self,base_path):
        torch.save(self.state_dict(),os.path.join(base_path,"model.pt"))
        torch.save(self.basis,os.path.join(base_path,"vaesim_basis.pt"))
        
        
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
            tsne=TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(self.basis.T)
            ax.scatter(tsne[:,0],tsne[:,1])
            
            if self.wandb_log:
                wandb.log({"init": wandb.Image(ax)})
            
            plt.show(block=self.block)
        
            
        
         
    def forward(self,x):
        
        """forward pass implementing the idea of this VAESIM
        
        @param: x data images
        
        return x_recon, z, soft_sim: Tuple of (Tensor,Tensor,Tensor) where x_recon are the reconstructed imgs, z are the latent reprs and         soft_sim the softmax of the similarities
        
        
        """
        
        
        z_mean,z_log_var = self.encoder(x)
        
        self.q_z = D.normal.Normal(z_mean, torch.exp(0.5 * z_log_var))

        device=z_mean.get_device()

        # sample z from it
        z = self.q_z.rsample().to(device)
        
        self.z=z
        
        #
        
        # reference N(0,1)
        #ref_mu = torch.zeros(z.shape[0], z.shape[-1])
        #ref_sigma = torch.ones(z.shape[0], z.shape[-1])

        #self.p_z = D.normal.Normal(ref_mu.to(device), ref_sigma.to(device))
        

        sim=self.compute_similarity(z)
        soft_sim=softmax(sim,dim=1)
        
        x = self.decoder(z,soft_sim)
        
        
        ## generate the posterior
        
        
        self.p_z=self.generate_ref(z,soft_sim)
        
        self.soft_sim=soft_sim
        
        return x,z,soft_sim
                
        
    def clusterize(self,soft_sim):
        
        """
        clustering function
        
        @param: soft_sim: Tensor, softmax of the similarity
        
        if sample.cluster -> cluster are chosen sampling a categorical distribution
        otherwise -> take the argmax
        
        return cl: Tensor with shape (BS,)
        """
        
        
        
        if self.sample_cluster:
            ## sampling
            d=D.Categorical(soft_sim/self.temperature)
            cl=d.sample()

        else:
            #take the argmax
            cl=soft_sim.argmax(dim=1)
        return cl
    
    
    
    def hierarchical_clustering(self,soft_sim):
        """A hierarchical function to associate labels to most probable cluster in a hiearchical way.
        At start time it expects a distribution of label over cluster which is uniform (this is an assumption -> dataset are balanced?)
        If the number of samples is below the threshold -> BS/2*n_basis that cluster is ignored and samples are associated to the most similar one"""
        
        expected=len(soft_sim)//(2*self.n_basis)
        
        ##generate clustering
        cl=soft_sim.argmax(dim=1)
        
        ##count elements for each cluster
        ###LAVORARE QUA
        
    
        
        
    def generate_ref(self,z,soft_sim):
        
        
        """
        compute the reference distributions for KL divergence
        
        @param z: Tensor, latent reprs
        @param soft_sim: Tensor, softmax of similarity
        
        return p_z: Distribution
        """
        
        cl=soft_sim.argmax(dim=1)
        ref=torch.zeros((cl.shape[0],self.latent_dim))
        
        device=soft_sim.get_device()

        for i in range(cl.shape[0]):
            ref[i,:]=self.basis[:,cl[i]]
        
        ref_mu = torch.zeros(z.shape[0], z.shape[-1])
        ref_sigma = 0.8*torch.ones(z.shape[0], z.shape[-1])
        
        
        self.p_z=D.normal.Normal(ref_mu.to(device), ref_sigma.to(device))
        
        return self.p_z
        
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

    
    def compute_kl_loss(self):
        
        """Compute the KL-divergence
        
        return weight*kl_div: Tensor, weighted kl divergence
        """

        kl_div = torch.mean(D.kl_divergence(self.q_z, self.p_z))
        #kl_div=-0.5 * torch.sum(1 + self.z_log_var - self.z_mean.pow(2) - self.z_log_var.exp())
        if self.kl:
            return self.weight*kl_div
        else:
            return 0.*kl_div

    
    def compute_reconstruction_loss(self, y_pred, y):
        
        """Compute recon loss
        @param y_pred: Tensor, reconstructed imgs
        @param y: Tensor, imgs
        """
        
        return self.recon_loss(y_pred, y)
        
    def compute_loss(self, y_pred, y):
        
        """Compute the global loss"""
        
        recon_loss = self.compute_reconstruction_loss(y_pred, y)
        kl_loss = self.compute_kl_loss()
        sim_loss=self.compute_similarity_loss()
        loss=recon_loss+sim_loss+kl_loss
        self.loss = loss
        return loss,recon_loss,kl_loss,sim_loss
    
    
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
        #cl=soft_sim.argmax(-1)
        
        cl=self.clusterize(soft_sim)
        
        #update the basis matrix columns according to the mean of the vector that belongs to that cluster
        for i in range(self.n_basis):
            
             #Here I check there is no NaN in the mean vector
            if not torch.isnan(z[cl==i].mean(axis=0)).any():
                
                if self.reinit:
                    r=torch.rand(1)
                    if r<=self.reinit:
                        ## pick a random value
                        idx=torch.randint(0,len(z[cl==i]),(1,))[0]
                        new_mean=z[cl==i][idx]
                    else:
                        ## just compute the usual way
                        new_mean=z[cl==i].mean(axis=0)
                else:
                    new_mean=z[cl==i].mean(axis=0)
                
                ##manage EMA if is set, otherwise just use the new mean
                
                if self.ema is not None:
                    ## EMA update
                    new_mean=self.ema*new_mean.cpu()+(1-self.ema)*self.basis[:,i]
                
                #update
                self.basis[:,i]=new_mean
                    
                
                
        
        return cl
    
    def compute_similarity_loss(self):
        
        """Compute similarity loss
        It's just the matrix product between B and transpose of B
        
        """
        
        B=self.normalize_basis()
        
        autosim=(B.T@B)
        sim_loss=self.sim_loss(autosim,torch.eye(self.n_basis))
        
        if self.similarity:
            return sim_loss
        else:
            return 0*sim_loss
    
    
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
    
    def schedule_temperature(self,t0,epoch,init=5,end=15,v1=1.,v2=0.05):
        if epoch<init:
            return t0
        elif epoch>end:
            return v2
        elif epoch>=init and epoch<=end:
            m=(v2-v1)/(end-init)
            return (epoch-init)*m+v1
    
    

    def fit(self,train_dataloader,val_dataloader=None,epochs=10,optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None,scheduler=None,threedimensional=False):

        """Main train loop. The logic is all under model.train().
        After the training epoch there is validation loop and some "callbacks" to store weights, save outputs to W&B and visualize it


        """

        self.device=device #set in memory the device type
        model=self
        
        self.wandb_log=wandb_log

        loss_history = []
        recon_loss_history = []
        kl_loss_history = []
        sim_loss_history = []

        val_loss_history = []
        val_recon_loss_history = []
        val_sim_loss_history = []

        for epoch in range(epochs):
            
            
            ### TRAIN ###
            
            # just store loss and accuracy for this epoch
            
            if self.schedule:
                self.temperature=self.schedule_temperature(self.T0,epoch)

            
            loss, recon_loss, kl_loss, sim_loss=self.train_epoch(train_dataloader,device=device,optimizer=optimizer,epoch=epoch)
                    
            # store the metrics for epochs
            loss_history.append(loss)
            recon_loss_history.append(recon_loss)
            kl_loss_history.append(kl_loss)
            sim_loss_history.append(sim_loss)
            
            if scheduler is not None:
                scheduler.step()
            
            if wandb_log:

                    wandb.log({"loss": loss_history[-1], "recon_loss": recon_loss_history[-1],
                               "kl_loss": kl_loss_history[-1], "sim_loss": sim_loss_history[-1]})
                    
            
            if val_dataloader is not None:
                
                val_imgs,_=next(iter(val_dataloader))
                
                if threedimensional:
                    sample=False
                else:
                    sample=True
                WandbImagesSIMVAE(model,val_imgs,show=False,sample=sample,smooth=True,save_prototypes=True,threedimensional=threedimensional)
                
            
            if save_model is not None:
                self.save_model(save_model)
            
        return loss_history,recon_loss_history,kl_loss_history,sim_loss_history

    
    
    
    def train_epoch(self,train_dataloader,device,optimizer,epoch):
        
        
            loss_temp = []
            recon_loss_temp = []
            kl_loss_temp = []
            sim_loss_temp = []

            self.train()
            with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:

                tepoch.set_description(f"Epoch {epoch}")
                for batch in tepoch:
                    # get the data and pass them to device

                    if len(batch)==2:
                        x,y=batch
                    else:
                        x = batch



                    x = x.to(device)

                    # compute the loss
                    x_pred,z,soft_sim = self(x)
                    
                    
                    loss,recon_loss,kl_loss,sim_loss = self.compute_loss(x_pred, x)

                    
                    z_clone=z.clone().detach()
                    s_clone=soft_sim.clone().detach()
                    
                    # backpropagate
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    self.update_cluster_means(z.detach(),soft_sim.detach())
                    
                    loss_temp.append(loss.item())
                    recon_loss_temp.append(recon_loss.item())
                    kl_loss_temp.append(kl_loss.item())
                    sim_loss_temp.append(sim_loss.item())
                    
                    tepoch.set_postfix({"loss": loss.item() ,"sim_loss":sim_loss.item(),"recon_loss":recon_loss.item(),                                                           "kl_loss":kl_loss.item()})
                    
            loss=np.mean(loss_temp)
            recon_loss=np.mean(recon_loss_temp)
            kl_loss=np.mean(kl_loss_temp)
            sim_loss=np.mean(sim_loss_temp)
            
            return loss, recon_loss, kl_loss, sim_loss
        
        
    
    def mapping(self,cl,y):
        """map[k]=label so we map the cluster k-th to the label"""

        y=y.squeeze()
        map={}

        acc={}

        for k in range(self.n_basis):
            if len(y[cl==k]):
                v=y[cl==k].mode().values
                map[k]=v
                acc[k]=(y[cl==k]==v).sum()/len(y[cl==k]) 
    
    
        #fix missing keys
        
        val,counts=np.unique(list(map.values()),return_counts=True)
        mode=val[np.argmax(counts)]
        
        map=defaultdict(lambda:mode,map)
        
        return map,acc
        
        
    def compute_mappings(self,train_dataloader,n_batch=30):
        
        model=self
        device=self.device
        i=0
        with torch.no_grad():




            stack=[]
            labels=[]
            z_list=[]
            cl_list=[]
            
            for data in tqdm.tqdm(train_dataloader):
                x,y=data
                #stack.append(x)
                labels.append(y)
                
                z_mean,z_log_var = model.encoder(x.to(device))
                model.q_z = D.normal.Normal(z_mean, torch.exp(0.5 * z_log_var))
                device=z_mean.get_device()
                z = model.q_z.rsample()
                
                sim=model.compute_similarity(z)
                soft_sim=softmax(sim,dim=1)
                cl=model.clusterize(soft_sim)
                
                z_list.append(z)
                cl_list.append(cl)
                i+=1
                if i==(n_batch-1):
                    break

            

            

            
            z=torch.cat(z_list)
            cl=torch.cat(cl_list)
            labels=torch.cat(labels)
            
            
           
            
            




            ##map each vector into his cluster
            #cl=sim.argmax(-1)
        print(f"Computing mappings using {len(cl)} samples from the training set")    
        mappings,accuracy=self.mapping(cl,labels)
        return mappings,accuracy,z.cpu(),cl.cpu(),labels.cpu()
    
    
    def compute_report(self,test_dataloader,mode="standard"):
        
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

        clusters=[]
        
    
        for x,y in tqdm.tqdm(test_dataloader):

            x_rec,z,sim=model(x.to(device))
            cluster=sim.argmax(dim=1)
            #cluster=model.clusterize(soft_sim)
            
            mapped=[mappings[cluster[i].cpu().item()].item() for i in range(len(cluster))]
            y_pred.append(mapped)

            clusters.append(cluster.cpu())
            labels.append(y)
            z_sampled.append(z.detach().cpu())
                                                                                      
            
                                                                                          

        
        
        
        y_pred=[item for sublist in y_pred for item in sublist]
        
        
        #y_pred=torch.cat(y_pred)
        labels=torch.cat(labels).numpy()
        z_sampled=torch.cat(z_sampled).numpy()
        clusters=torch.cat(clusters).numpy()
        #z_sampled=np.array(z_sampled)
        
        
        
        print(f"Computing mappings using {len(clusters)} samples from the test set")    

        class_report=classification_report(labels, y_pred)
        class_report_dict=classification_report(labels, y_pred,output_dict=True)
        df = pd.DataFrame(class_report_dict).transpose()
        accuracy=accuracy_score(labels,y_pred)
        
        return class_report,z_sampled,labels,clusters,accuracy,df
        
        
        
    def compute_tsne(self,z_sampled,labels):
        tsne=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(np.squeeze(z_sampled))
         
        cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray',10:'tab:red', 11: 'tab:orange', 12: 'tab:green', 13: 'cyan', 14: 'lime', 15:'seagreen'}

        fig, ax = plt.subplots(figsize=(10,10))
        for g in np.unique(labels):
            ix = np.where(labels == g)[0]
            
            if len(ix):
                ax.scatter(tsne[:,0][ix], tsne[:,1][ix], c = cdict[g], label = g, s =20,alpha=0.7,linewidth=1.)
        #plt.scatter(tsne[:,0],tsne[:,1],label=labels,c= cdict[g])
        ax.legend()
        ax.set_title("tSNE represantation of the latent space")
        plt.show(block=self.block)
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
    
    
    
    def compute_tsne_training(self,z_sampled_train,cl_train,labels_train,zoom=1.,threedimensional=False):
        tsne=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(np.squeeze(z_sampled_train.squeeze()))
         
        cdict = {0: 'red', 1: 'blue', 2: 'green',3: 'yellow', 4: 'orange', 5: 'pink', 6:'tab:blue', 7:'purple',8:'brown',9:'gray',10:'tab:red', 11: 'tab:orange', 12: 'tab:green', 13: 'cyan', 14: 'lime', 15:'seagreen'}

        fig, axs = plt.subplots(1,2,figsize=(20,10))
        for g in np.unique(labels_train):
            ix = np.where(labels_train == g)[0]
            axs[0].scatter(tsne[:,0][ix], tsne[:,1][ix], c = cdict[g], label = g, s =20,alpha=0.7,linewidth=1.)
        #plt.scatter(tsne[:,0],tsne[:,1],label=labels,c= cdict[g])
        axs[0].legend()
        axs[0].set_title("tSNE represantation of the latent space - Training set with real labels")
        
        
       
        
        sns.scatterplot(x=tsne[:,0], y=tsne[:,1],hue=cl_train,ax=axs[1],legend="full",palette="muted")
        
        
        self.plot_prototypes_over_tsne(axs[1],tsne,cl_train,zoom=zoom,threedimensional=threedimensional)
        axs[1].legend()
        axs[1].set_title("tSNE represantation of the latent space - Training set with cluster labels")
        
        
        
        plt.show(block=self.block)
        return fig
    
    
    def plot_prototypes_over_tsne(self,ax,tsne,cl_train,zoom=1.,threedimensional=False):
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
        
    
    
    def just_report(self,y_true,y_pred):
        """just compute the report"""
        class_report=classification_report(y_true, y_pred)
        class_report_dict=classification_report(y_true, y_pred,output_dict=True)
        df = pd.DataFrame(class_report_dict).transpose()
        accuracy=accuracy_score(y_true,y_pred)
        
        return class_report,df,accuracy
    
    def evaluate_linear(self,z_sampled_train,labels_train,BS=512,lr=1e-3,epochs=10):
        
        lin_z=Linear()
        lin_z=lin_z.to(self.device)
        
        opt_z=torch.optim.Adam(lin_z.parameters(),lr=lr)
        criterion=torch.nn.CrossEntropyLoss()
        
        z_train_dataset=torch.utils.data.TensorDataset(z_sampled_train,labels_train)
        z_train_dataloader=torch.utils.data.DataLoader(z_train_dataset,shuffle=True,batch_size=BS)
        
        z_loss,z_acc=self.train_linear(model=lin_z,criterion=criterion,optim=opt_z,epochs=epochs,train_dataloader=z_train_dataloader)
        
        return lin_z,z_loss,z_acc

        
        
    
    def test_linear(self,model,z_sampled,labels,BS=512):
        device=self.device
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
    
    
    
    def train_linear(self,model,criterion,optim,epochs,train_dataloader):
        
        device=self.device
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
        

    
    def evaluate(self,batch,train_dataloader,test_dataloader,wandb_log=True,n_semi=50,zoom=1.,threedimensional=False,neighbours=5,lin_epochs=100,show=True):
        
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
        x2,z,s=model(x.to(device))
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
        
        try:
            figs,axs=plt.subplots(5,self.n_basis//5,figsize=(30,30))

            for (i,ax) in enumerate(axs.ravel()):
                dist=y[cl==i].cpu().numpy()
                ax.hist(dist)
                ax.set_title(f"Labels over cluster {i}")

            if wandb_log:
                wandb.log({"Distribution over cluster": wandb.Image(fig), "Label distributions":wandb.Image(figs)})
        except: 
            print("Difficult to check label distribution")
        
        ## COMPUTE MAPPINGS ON PART OF TRAINING SET
        mappings,acc,z_sampled_train,cl_train,labels_train=self.compute_mappings(train_dataloader,n_semi)
        
        self.mappings=mappings
        
            
        frequencies=[(k,v.numpy()) for k,v in acc.items()]
        if wandb_log:
             wandb.log({"Freq": frequencies})
                     
        print(acc)
        print(mappings)
        
        
        
        hiearchical_mappings=self.compute_hierarchical_mappings(cl_train,labels_train,d)
        
        
       
        
        ### ALL THESE METRICS ARE COMPUTED ON TEST SET
        
        print("Computing classification report for hierarchical clustering")
        class_report_hier,z_sampled_hier,labels_hier,clusters_hier,accuracy_hier,df_hier=self.compute_report(test_dataloader,mode="hierarchical")
        print(class_report_hier)
        
        
        
        print("Computing classification report for standard clustering")
        class_report,z_sampled,labels,clusters,accuracy,df=self.compute_report(test_dataloader)
        print(class_report)
        
        ### CLUSTERING METRICS
        sil_score,di_score,nmi_score,sil_labels=self.cluster_metrics(z_sampled,clusters,labels)
        
        print(f"Silhouette score:\t{sil_score}\nDunn Index:\t{di_score}\nNMI score:\t{nmi_score}\nSilhouette Score Mapped:\t{sil_labels}")
        if wandb_log:
            wandb.log({"Silhouette Score": sil_score, "Dunn Index":di_score, "NMI":nmi_score,"Silhouette Score Mapped": sil_labels})
            
            
        ### further evaluations
        
        print(f"[INFO] Running evaluations from k Nearest Neighbour with k= {neighbours}")
        
        neigh = KNeighborsClassifier(neighbours)
        neigh.fit(z_sampled_train,labels_train)
        y_pred_neigh=neigh.predict(z_sampled)
        
        
        class_report, df, neigh_accuracy=just_report(labels,y_pred_neigh)
        print(class_report)
        if wandb_log:
            wandb.log({"kNN Report": df, "kNN Accuracy":neigh_accuracy})
            
            
        ## LINEAR CLASSIFIER
        print(f"[INFO] Running evaluations from with linear classifier")
        
        lin_z,z_loss,z_acc=evaluate_linear(z_sampled_train,labels_train,epochs=lin_epochs)
        y_z_pred,y_z_true=test_linear(lin_z,z_sampled,labels)

        fig,axs=plt.subplots(1,2,figsize=(6,6))
        axs[0].plot(z_loss,label="z loss")
        axs[0].set_title("Linear training loss")
        axs[1].plot(z_acc,label="z loss")
        axs[1].set_title("Linear training acc")
        
        class_report, df, lin_accuracy=self.just_report(y_z_true,y_z_pred)
        print(class_report)
        if wandb_log:
            wandb.log({"Linear Report": df, "Linear Accuracy":lin_accuracy,"Linear layer": wandb.Image(fig)})
        
        
         
            
        fig=self.compute_tsne_training(z_sampled_train,cl_train,labels_train,threedimensional=threedimensional)
        if wandb_log:
            wandb.log({"Training latent space": wandb.Image(fig)})
            
        
            

        
        ax=self.compute_tsne(z_sampled,labels)
        
        if wandb_log:
            wandb.log({"Report": df, "Accuracy":accuracy,"Latent space": wandb.Image(ax)})
            
        return accuracy,neigh_accuracy,lin_accuracy
        
        


        
        
        
        
        
        
class SkipVAESIM(VAESIM):
    
    def __init__(self, input_dim, latent_dim,encoder,decoder,n_basis=10,weight=None,sim_weight=1.,similarity=False,kl=False,sample_cluster=False,temperature=0.01,reinit=0.,schedule=True):
        super().__init__(input_dim, 
                    latent_dim,
                    encoder,decoder,
                    n_basis,
                    weight,
                    sim_weight,
                    similarity,
                    kl,
                    sample_cluster,
                    temperature,
                    reinit,
                    schedule)
        
        
        #print(f"Check for skip connections")
        #module_list=tx.list_module_names(self.encoder)


        #conv_skip=[]
        #for i in module_list:
        #    if "conv" in i:
        #        print(i)
        #        conv_skip.append(i)
        #encoder = tx.Extractor(self.encoder, conv_skip)
        #dummy_input=torch.zeros(*input_dim)

        #print(dummy_input.unsqueeze(0).shape)
        #print(e(dummy_input.unsqueeze(0)))
        #out,features = encoder(dummy_input.unsqueeze(0))
        
        #feature_shapes = {name: f.shape for name, f in features.items()}
        #print(feature_shapes)

        #self.encoder=encoder
        
        
        
        
        
    def forward(self,x):
        
        """forward pass implementing the idea of this VAESIM
        
        @param: x data images
        
        return x_recon, z, soft_sim: Tuple of (Tensor,Tensor,Tensor) where x_recon are the reconstructed imgs, z are the latent reprs and         soft_sim the softmax of the similarities
        
        
        """
        
        
        out,skip =self.encoder(x)
        z_mean,z_log_var=out
        self.q_z = D.normal.Normal(z_mean, torch.exp(0.5 * z_log_var))

        device=z_mean.get_device()

        # sample z from it
        z = self.q_z.rsample().to(device)
        
        self.z=z
        
        #
        
        # reference N(0,1)
        #ref_mu = torch.zeros(z.shape[0], z.shape[-1])
        #ref_sigma = torch.ones(z.shape[0], z.shape[-1])

        #self.p_z = D.normal.Normal(ref_mu.to(device), ref_sigma.to(device))
        

        sim=self.compute_similarity(z)
        soft_sim=softmax(sim,dim=1)
        
        x = self.decoder(z,soft_sim,skip)
        
        
        ## generate the posterior
        
        
        self.p_z=self.generate_ref(z,soft_sim)
        
        self.soft_sim=soft_sim
        
        return x,z,soft_sim
    
    
    def fit(self,train_dataloader,val_dataloader=None,epochs=10,optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None):

        """Main train loop. The logic is all under model.train().
        After the training epoch there is validation loop and some "callbacks" to store weights, save outputs to W&B and visualize it


        """

        self.device=device #set in memory the device type
        model=self
        
        self.wandb_log=wandb_log

        loss_history = []
        recon_loss_history = []
        kl_loss_history = []
        sim_loss_history = []

        val_loss_history = []
        val_recon_loss_history = []
        val_sim_loss_history = []

        for epoch in range(epochs):
            
            
            ### TRAIN ###
            
            # just store loss and accuracy for this epoch
            
            if self.schedule:
                self.temperature=self.schedule_temperature(self.T0,epoch)

            
            loss, recon_loss, kl_loss, sim_loss=self.train_epoch(train_dataloader,device=device,optimizer=optimizer,epoch=epoch)
                    
            # store the metrics for epochs
            loss_history.append(loss)
            recon_loss_history.append(recon_loss)
            kl_loss_history.append(kl_loss)
            sim_loss_history.append(sim_loss)
            
            if wandb_log:

                    wandb.log({"loss": loss_history[-1], "recon_loss": recon_loss_history[-1],
                               "kl_loss": kl_loss_history[-1], "sim_loss": sim_loss_history[-1]})
                    
            
            if val_dataloader is not None:
                
                val_imgs,_=next(iter(val_dataloader))
                
                WandbImagesSIMVAE(model,val_imgs,show=False,sample=False,smooth=False,save_prototypes=False)
                
            
            if save_model is not None:
                self.save_model(save_model)
            
        return loss_history,recon_loss_history,kl_loss_history,sim_loss_history

    
    def compute_mappings(self,train_dataloader):
        
        model=self
        device=self.device
        with torch.no_grad():

            n_semi=2000



            stack=[]
            labels=[]
            for i,data in enumerate(train_dataset):
                x,y=data
                stack.append(x)
                labels.append(y)
                if i==n_semi-1:
                    break


            stack=torch.stack(stack).to(device)
            labels=torch.Tensor(labels)

            out,skip =self.encoder(stack)
            z_mean,z_log_var=out

            model.q_z = D.normal.Normal(z_mean, torch.exp(0.5 * z_log_var))

            device=z_mean.get_device()
            z = model.q_z.rsample()

            sim=model.compute_similarity(z)
            soft_sim=softmax(sim,dim=1)


            ##map each vector into his cluster
            #cl=sim.argmax(-1)
            cl=model.clusterize(soft_sim)
        mappings,accuracy=self.mapping(cl,labels)
        return mappings,accuracy
    
    