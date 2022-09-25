import torch
import tqdm
from torch import nn
from torch import distributions as D
from torch.nn import functional as F
from .Architectures import VAEDecoder, VAEEncoder, Discriminator
import numpy as np

import sys
sys.path.append('../utils')
from utils.callbacks import *



class miniVAE(nn.Module):
    def __init__(self,input_dim, latent_dim,encoder_architecture=[(0,128),(0,256)], decoder_architecture=[(0,128),(0,256)]):
        super().__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.encoder = VAEEncoder(latent_dim,encoder_architecture)
        self.decoder = VAEDecoder(input_dim,latent_dim,decoder_architecture)

    def reparametrize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z = mu + torch.exp(0.5 * logvar) * epsilon
        return z

    def loss_calc(self, x, x_hat, mu, logvar):
        BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)


        #z = self.reparametrize(mu, logvar)

        self.q_z = D.Normal(z_mean, torch.exp(z_log_var/2.))

        z = self.q_z.rsample()

        x_hat = self.decoder(z)
        loss = self.loss_calc(x, x_hat, z_mean, z_log_var)
        return x_hat, loss



    def fit(self,train_dataloader,val_dataloader=None,epochs=10,optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None):
        self.device=device #set in memory the device type
        model=self

        loss_history = []
        recon_loss_history = []
        kl_loss_history = []

        val_loss_history = []
        val_recon_loss_history = []
        val_kl_loss_history = []

        for epoch in range(epochs):

            # just store loss and accuracy for this epoch

            loss_temp = []
            recon_loss_temp = []
            kl_loss_temp = []

            model.train()
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
                    x_pred, loss= model(x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(loss=loss.item())




class VAE(nn.Module):

    """Class for Variational Autoencoder extending nn.Module

        more in-depth explaination could be found
        at :         https://arxiv.org/abs/1606.05908
        or :         https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
        very useful: https://agustinus.kristia.de/techblog/2016/12/10/variational-autoencoder/


        Theory:
        --------

        The idea of a VAE is that x=d(e(x)) where e is an encoder and d a decoder.
        The encoder map the input x in a latent space distribution p(z|x) and we want that similar inputs are close in this latent space


        We want p(z|x) gaussian and we can use bayes theorem

        p(z|x)= p(x|z) p(z)/ p(x)

        p(x|z) is a gaussian N(f(z),cI).
        Let's assume a N(0,1) for the prior p(z) so ideally we could compute p(z|x).
        Unfortunately p(x) is a sort for normalizazion that could be expressed p(x) = integrate p(x|u)p(u)du over all possible u.
        This is usally untractable so we need a function that approximate p(z|x) because we can't compute it directly.

        In Variation Inference (VI) usually we look for best approximation of a target distribution from parametrized distributions in
        a family like gaussians. We try to minimize a measure of distance between target and approximating distributions.

        So we can approximate p(z|x) with a q_x(z) which is a Gaussian N(g(x),h(x))

        We can search in the space of g and h their best values, the values that minimize the KL divergence

        g*,h*=argmin KL(q_x(z),p(z|x))

        using KL definition and bayes theorem

        g*,h*=argmin (E[log(q_x(z))] - E[log(p(x|z)] - E[log(p(z))] + E[log(p(x))])

        rearranging the terms and discarding E[log(p(z))] which is a constant

        g*,h*=argmin( E(log(p(x|z)) - KL (q_x(z),p(x))).

        E(log(p(x|z)) is just (1/2c) *||x-f(z)||^2 becuase p(x|z) is a guassian and when we found the best values to compute q
        we an use them to approximate p(x) which was the problematic quantity.

        Basically encoder is Q(z|x) or q_x(z) and the decoder is p(x|z).



        """

    def __init__(self, input_dim, latent_dim,weight=None, encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]],**kwargs):
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



        """
        super().__init__(**kwargs)


        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture

        if weight is None:
            self.weight=latent_dim/np.prod(input_dim)
        else:
            self.weight=weight

        self.encoder = VAEEncoder( latent_dim=latent_dim,  conv_layer_list=encoder_architecture)
        self.decoder = VAEDecoder(self.input_dim, latent_dim=latent_dim,  conv_layer_list=decoder_architecture)

        self.kl_loss=0
        self.recon_loss=nn.MSELoss()
        self.loss=None

        self.patience=0 #for early stopping
        self.device=None

    def sample(self,z_mean,z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(z_log_var/2.) * epsilon


    def forward(self, x):


        z_mean,z_log_var = self.encoder(x)


        self.q_z = D.normal.Normal(z_mean, torch.exp(0.5 * z_log_var))

        device=z_mean.get_device()

        # sample z from it
        z = self.q_z.rsample()

        #z=self.sample(z_mean,z_log_var)

        self.z=z
        self.z_mean,self.z_log_var=z_mean,z_log_var

        # reference N(0,1)
        ref_mu = torch.zeros(z.shape[0], z.shape[-1])
        ref_sigma = torch.ones(z.shape[0], z.shape[-1])

        self.p_z = D.normal.Normal(ref_mu.to(device), ref_sigma.to(device))

        x = self.decoder(z)

        return x

    def compute_kl_loss(self):

        #log_qzx = self.q_z.log_prob(self.z)
        #log_pz =  self.p_z.log_prob(self.z)

        # kl
        #kl = -(log_qzx - log_pz)
        #kl = kl.sum(-1)

        kl_div = torch.mean(D.kl_divergence(self.q_z, self.p_z))
        #kl_div=-0.5 * torch.sum(1 + self.z_log_var - self.z_mean.pow(2) - self.z_log_var.exp())
        return self.weight*kl_div

    def compute_reconstruction_loss(self, y_pred, y):
        return self.recon_loss(y_pred, y)

    def compute_loss(self, y_pred, y):
        recon_loss = self.compute_reconstruction_loss(y_pred, y)
        kl_loss = self.compute_kl_loss()

        self.loss = recon_loss + kl_loss

        return self.loss, recon_loss, kl_loss


    def train_step(self,x,optimizer):
        x_pred = self(x)

        loss, recon_loss, kl_loss = self.compute_loss(x_pred, x)

        # backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss,recon_loss,kl_loss


    def fit(self,train_dataloader,val_dataloader=None,epochs=10,optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None):

        """Main train loop. The logic is all under model.train().
        After the training epoch there is validation loop and some "callbacks" to store weights, save outputs to W&B and visualize it


        """

        self.device=device #set in memory the device type
        model=self

        loss_history = []
        recon_loss_history = []
        kl_loss_history = []

        val_loss_history = []
        val_recon_loss_history = []
        val_kl_loss_history = []

        for epoch in range(epochs):

            # just store loss and accuracy for this epoch

            loss_temp = []
            recon_loss_temp = []
            kl_loss_temp = []

            model.train()
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
                    x_pred = model(x)

                    loss, recon_loss, kl_loss = model.compute_loss(x_pred, x)

                    # backpropagate
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_temp.append(loss.item())
                    recon_loss_temp.append(recon_loss.item())
                    kl_loss_temp.append(kl_loss.item())

                    tepoch.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), kl_loss=kl_loss.item())

            # store the metrics for epochs
            loss_history.append(np.mean(loss_temp))
            recon_loss_history.append(np.mean(recon_loss_temp))
            kl_loss_history.append(np.mean(kl_loss_temp))

            if wandb_log:
                wandb.log({"loss": loss_history[-1], "recon_loss": recon_loss_history[-1], "kl_loss": kl_loss_history[-1]})

            if val_dataloader is not None:
                val_loss_temp = []
                val_recon_loss_temp = []
                val_kl_loss_temp = []

                model.eval()
                with tqdm.tqdm(val_dataloader, unit="batch") as tepoch:

                    tepoch.set_description(f"Val {epoch}")
                    for batch in tepoch:
                        # get the data and pass them to device

                        if len(batch) == 2:
                            x, y = batch
                        else:
                            x = batch
                        x = x.to(device)

                        # compute the loss
                        x_pred = model(x)

                        val_loss, val_recon_loss, val_kl_loss = model.compute_loss(x_pred, x)

                        # store info
                        val_loss_temp.append(val_loss.item())
                        val_recon_loss_temp.append(val_recon_loss.item())
                        val_kl_loss_temp.append(val_kl_loss.item())

                        tepoch.set_postfix(val_loss=val_loss.item(), val_recon_loss=recon_loss.item(),
                                           val_kl_loss=kl_loss.item())
                    ##handle callbacks



                # store the metrics for validation epochs
                val_loss_history.append(np.mean(val_loss_temp))
                val_recon_loss_history.append(np.mean(val_recon_loss_temp))
                val_kl_loss_history.append(np.mean(val_kl_loss_temp))

                if early_stop is not None:
                    if len(val_loss_history)>2:
                        if val_loss_history[-1]>val_loss_history[-2]:
                            self.patience+=1
                    if self.patience==early_stop:
                        print("[INFO] Early Stopping")
                        break

                if wandb_log:

                    wandb.log({"val_loss": val_loss_history[-1], "val_recon_loss": val_recon_loss_history[-1],
                               "val_kl_loss": val_kl_loss_history[-1]})

                    WandbImagesVAE(self, x,show=True)



                if save_model is not None:
                    SaveModel(self,save_model)

    def encode(self,x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)


class VAEGAN(nn.Module):
    """A class to train a VAE-GAN model

    This implementation follows the basic idea of a GAN, so we will have

    VAE:
    Encoder
    Decoder

    GAN
    Decoder/Generator
    Discriminator

    the models share the Decoder weights and are trained in a loop:

    Discriminator trained on real/sampled images with right labels

    VAE trained to reconstruct images

    Sampled images passed to discriminator as real to train the generator to foolish it


    """
    def __init__(self,input_dim,latent_dim,output_channels=1,encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]],discriminator_architecture=[(0,128),[(0,256)]],discriminator_dense=None):
        """

        :param input_dim: tuple, input dimension (for example (28,28,1)
        :param latent_dim: int, dimension of the latent space
        :param output_channels: number of output channels. Usally should be the same of input_dim[-1]

        """


        super().__init__()

        self.input_dim=input_dim
        self.latent_dim=latent_dim


        self.output_channels=input_dim[-1]


        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture
        self.discriminator_architecture=discriminator_architecture

        self.discrminator_dense = discriminator_dense
        self.vae = VAE(input_dim, latent_dim,encoder_architecture=encoder_architecture, decoder_architecture=decoder_architecture,)
        self.discriminator=Discriminator(n_classes=1,conv_layer_list=discriminator_architecture,dense=discriminator_dense,activation="sigmoid")

        self.d_criterion=nn.BCELoss()
        self.recon_criterion=nn.MSELoss()

        self.patience=0

    def forward(self,x):
        return self.vae(x)



    def fit(self,train_dataloader,val_dataloader=None,epochs=10,v_optimizer=None,d_optimizer=None,g_optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None):
        """
        Method used to override the behaviour of .fit method

        This training step involve three different steps:

        step 1: Discriminator training

        Real images and fake ones (generated by noise from the vae's decoder) are passed to the discriminator
        with 0 as labels for real ones and 1 for the fake ones.
        The parameters of the discriminator are updated from the binarycrossentropy computations

        Step 2: VAE training

        Real images x are reconstructed by the VAE as x'=D(E(x)) where E is the encoder network and D the decoder
        Using the KL loss and the reconstruction loss (MSE) weights of the E and D are updated

        Step 3: VAE-GAN training

        Sampled images are passed to the discriminator labelled as real ones with misleading labels
        The Decoder weights are updated using the BinaryCrossentropy gradients.
        This will incourage the decoder to learn how to foolish the discriminator.




        :param data: images passed
        :return: dict with metrics
        """
        self.device=device #set in memory the device type
        self.vae.device=device

        model=self

        loss_history = []
        d_loss_history=[]
        kl_loss_history=[]
        g_loss_history=[]

        val_loss_history = []
        val_d_loss_history = []
        val_d_acc_history = []

        for epoch in range(epochs):

                # just store loss and accuracy for this epoch

                loss_temp = []
                d_loss_temp = []
                kl_loss_temp = []
                g_loss_temp = []

                model.train()
                with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:

                    tepoch.set_description(f"Epoch {epoch}")
                    for batch in tepoch:
                        # get the data and pass them to device

                        if len(batch)==2:
                            x,y=batch
                        else:
                            x = batch

                        bs=len(x)

                        x = x.to(device)

                        #step 1: Train discriminator
                        d_loss=self.train_discriminator(x,d_optimizer=d_optimizer)

                        #step 2: Train the vae itself

                        loss, recon_loss, kl_loss = self.vae.train_step(x,v_optimizer)

                        #step 3: train the vaegan (namely the generator) in adversarial way

                        g_loss=self.train_vaegan(bs,g_optimzer=g_optimizer)

                        tepoch.set_postfix(loss=loss.item(), d_loss=d_loss.item(), g_loss=g_loss.item(),kl_loss=kl_loss.item())

                        loss_temp.append(loss.item())
                        d_loss_temp.append(d_loss.item())
                        kl_loss_temp.append(kl_loss.item())
                        g_loss_temp.append(g_loss.item())

                loss_history.append(np.mean(loss_temp))
                d_loss_history.append(np.mean(d_loss_temp))
                g_loss_history.append(np.mean(g_loss_temp))
                kl_loss_history.append(np.mean(kl_loss_temp))


                ## VALIDATION
                if wandb_log:
                    wandb.log({"loss": loss_history[-1], "d_loss": d_loss_history[-1],
                               "kl_loss": kl_loss_history[-1],"g_loss":g_loss_history[-1]})

                if val_dataloader is not None:
                    val_loss_temp = []
                    val_d_loss_temp = []
                    val_acc_temp = []

                    model.eval()
                    with torch.no_grad():
                        with tqdm.tqdm(val_dataloader, unit="batch") as tepoch:

                            tepoch.set_description(f"Val {epoch}")
                            for batch in tepoch:
                                # get the data and pass them to device

                                if len(batch) == 2:
                                    x, y = batch
                                else:
                                    x = batch
                                x = x.to(device)

                                d_loss,d_accuracy=self.test_discriminator(x)
                                x_recon,loss=self.test_vae(x)

                                tepoch.set_postfix(loss=loss.item(), d_loss=d_loss.item(), d_accuracy=d_accuracy.cpu().numpy())
                                val_acc_temp.append(d_accuracy.cpu())
                                val_d_loss_temp.append(d_loss.item())
                                val_loss_temp.append(loss.item())

                # store the metrics for validation epochs
                val_loss_history.append(np.mean(val_loss_temp))
                val_d_loss_history.append(np.mean(val_d_loss_temp))
                val_d_acc_history.append(np.mean(val_acc_temp))

                if early_stop is not None:
                    if len(val_loss_history)>2:
                        if val_loss_history[-1]>val_loss_history[-2]:
                            self.patience+=1
                    if self.patience==early_stop:
                        print("[INFO] Early Stopping")
                        break

                if wandb_log:

                    wandb.log({"val_loss": val_loss_history[-1], "val_d_loss": val_d_loss_history[-1],
                               "val_d_acc": val_d_acc_history[-1]})

                    WandbImagesVAE(self.vae, x,show=True)



                if save_model is not None:
                    SaveModel(self,save_model)


    def train_vaegan(self,bs,g_optimzer):
        random_latent_vectors = torch.randn(bs,self.latent_dim).to(self.device)
        misleading_labels = torch.zeros((bs, 1)).to(self.device)

        y_pred = self.discriminator(self.vae.decoder(random_latent_vectors))
        g_loss = self.d_criterion(y_pred,misleading_labels)

        g_optimzer.zero_grad()
        g_loss.backward()
        g_optimzer.step()

        return g_loss


    def train_discriminator(self,x,d_optimizer,smooth=False):

        random_latent_vectors = torch.randn(len(x),self.latent_dim).type_as(x)

        # Decode them to fake images
        generated_images = self.vae.decode(random_latent_vectors)

        # Combine them with real images
        combined_images = torch.cat([generated_images, x], axis=0)

        # Assemble labels discriminating real from fake images
        labels = torch.cat([torch.ones((len(x), 1)), torch.zeros((len(x), 1))], axis=0)

        if smooth:
            # Add random noise to the labels - important trick!
            labels += 0.05 * torch.rand(*labels.shape)

        labels=labels.to(self.device)
        y_pred=self.discriminator(combined_images)

        d_loss=self.d_criterion(y_pred,labels)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        return d_loss


    def test_discriminator(self,x):

        random_latent_vectors = torch.randn(len(x), self.latent_dim).type_as(x)

        # Decode them to fake images
        generated_images = self.vae.decode(random_latent_vectors)

        # Combine them with real images
        combined_images = torch.cat([generated_images, x], axis=0)

        # Assemble labels discriminating real from fake images
        labels = torch.cat([torch.ones((len(x), 1)), torch.zeros((len(x), 1))], axis=0)

        labels = labels.to(self.device)
        y_pred = self.discriminator(combined_images)

        d_loss = self.d_criterion(y_pred, labels)
        d_accuracy=torch.sum(y_pred==labels)

        return d_loss,d_accuracy

    def test_vae(self,x):
        y_pred=self.vae(x)
        return y_pred,self.recon_criterion(y_pred,x)