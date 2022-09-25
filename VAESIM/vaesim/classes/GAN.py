from collections import OrderedDict

import torch
import tqdm
import wandb
from torch import nn
from torch import distributions as D
import torch.nn.functional as F
import torchvision
import numpy as np
from classes.Architectures import Decoder,Discriminator
from utils.callbacks import SaveModel, WandbImagesGAN


class GAN(nn.Module):
    """
    base class for adversarial network learning that extends keras.Model

    """

    def __init__(self,target_shape,latent_dim,discriminator_architecture=[(0,128),[(0,256)]], generator_architecture=[(0,128),[(0,256)]],discriminator_dense=None):
        """

        Attributes
        ----------

        :param latent_dim: dimension of the latent space (i.e the number of random numbers required to generate an image)
        :param target_shape: tuple, shape of the image
        :param discriminator: model
        :param generator : model
        :param latent_dim: dimension of the latent space (i.e the number of random numbers required to generate an image)
        :param discriminator_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..] for discriminator
        :param generator_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..] for generator


        Methods
        ---------
        build_discriminator : build a sequential Keras model to discriminate between real and fake images
        build_generator: build a sequential Keras model to generate images from noise though Conv2DTranspose layers.

        """
        super().__init__()

        self.target_shape = target_shape
        self.latent_dim = latent_dim
        self.decoder_architecture=discriminator_architecture
        self.generator_architecture=generator_architecture


        self.discriminator=Discriminator(1,discriminator_architecture,dense=discriminator_dense,activation="sigmoid")
        self.generator=Decoder(target_shape,latent_dim,generator_architecture)

        self.gan=nn.Sequential(self.generator,self.discriminator)

        self.lr=1e-4
        self.b1=0.99
        self.b2=0.9

        self.validation_z = torch.randn(100, self.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)




    def fit(self,train_dataloader,val_dataloader=None,epochs=10,g_optimizer=None,d_optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None):


        self.device=device #set in memory the device type
        model=self


        d_loss_history = []
        g_loss_history = []
        d_accuracy_history = []


        val_d_loss_history = []
        val_g_loss_history = []
        val_d_accuracy_history = []


        for epoch in range(epochs):

            # just store loss and accuracy for this epoch

            d_loss_temp = []
            g_loss_temp = []
            d_accuracy_temp = []

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

                    # sample noise
                    z = torch.randn(x.shape[0], self.latent_dim)
                    z = z.type_as(x)

                    ## step 1 train the generator
                    self.generated_imgs = self(z)
                    valid = torch.ones(x.size(0), 1)
                    valid = valid.type_as(x)

                    g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)

                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()


                    g_loss_temp.append(g_loss.item())



                    ################
                    #
                    # step 2 train the discriminator
                    ############



                    valid = torch.ones(x.size(0), 1)
                    valid = valid.type_as(x)

                    y_real_pred=self.discriminator(x)
                    real_loss = self.adversarial_loss(y_real_pred, valid)

                    # how well can it label as fake?
                    fake = torch.zeros(x.size(0), 1)
                    fake = fake.type_as(x)

                    y_fake_pred=self.discriminator(self.generator(z).detach())

                    fake_loss = self.adversarial_loss(y_fake_pred, fake)

                    d_loss = (real_loss + fake_loss) / 2

                    d_optimizer.zero_grad()
                    d_loss.backward()
                    d_optimizer.step()

                    d_loss_temp.append(d_loss.item())

                    d_acc=(torch.sum(y_real_pred == valid)+torch.sum(y_fake_pred==fake))
                    d_accuracy_temp.append(d_acc.cpu())

                g_loss_history.append(np.mean(g_loss_temp))
                d_loss_history.append(np.mean(d_loss_temp))
                d_accuracy_history.append(np.mean(d_accuracy_temp))

            ## callbacks

            if wandb_log:
                wandb.log({"g_loss": g_loss_history[-1], "d_loss": d_loss_history[-1], "d_acc": d_accuracy_history[-1]})
                WandbImagesGAN(self, show=True)
            ## Save model
            if save_model is not None:
                SaveModel(self, save_model)

            ## Log to




class improvedGAN(nn.Module):
    """
    base class for adversarial network learning that extends nn.Module
    implements some of the suggestion form "Improved Techniques for GANs" paper https://arxiv.org/pdf/1606.03498v1.pdf

    """

    def __init__(self,target_shape,latent_dim,discriminator_architecture=[(0,128),[(0,256)]], generator_architecture=[(0,128),[(0,256)]],discriminator_dense=None):
        """

        Attributes
        ----------

        :param latent_dim: dimension of the latent space (i.e the number of random numbers required to generate an image)
        :param target_shape: tuple, shape of the image
        :param discriminator: model
        :param generator : model
        :param latent_dim: dimension of the latent space (i.e the number of random numbers required to generate an image)
        :param discriminator_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..] for discriminator
        :param generator_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..] for generator


        Methods
        ---------
        build_discriminator : build a sequential Keras model to discriminate between real and fake images
        build_generator: build a sequential Keras model to generate images from noise though Conv2DTranspose layers.

        """
        super().__init__()

        self.target_shape = target_shape
        self.latent_dim = latent_dim
        self.decoder_architecture=discriminator_architecture
        self.generator_architecture=generator_architecture


        self.discriminator=Discriminator(1,discriminator_architecture,dense=discriminator_dense,activation="sigmoid",matching=True) #matching True is the main difference between standard gan
        self.generator=Decoder(target_shape,latent_dim,generator_architecture)

        self.gan=nn.Sequential(self.generator,self.discriminator)

        self.lr=1e-4
        self.b1=0.99
        self.b2=0.9

        self.validation_z = torch.randn(100, self.latent_dim)
        self.feature_loss= nn.MSELoss()

    def forward(self, z):
        return self.generator(z)


    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)


    def fit(self,train_dataloader,val_dataloader=None,epochs=10,g_optimizer=None,d_optimizer=None,device="cuda",wandb_log=True,save_model=None,early_stop=None):


        self.device=device #set in memory the device type
        model=self


        d_loss_history = []
        g_loss_history = []
        d_accuracy_history = []


        val_d_loss_history = []
        val_g_loss_history = []
        val_d_accuracy_history = []


        for epoch in range(epochs):

            # just store loss and accuracy for this epoch

            d_loss_temp = []
            g_loss_temp = []
            d_accuracy_temp = []

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

                    # sample noise
                    z = torch.randn(x.shape[0], self.latent_dim)
                    z = z.type_as(x)

                    ## step 1 train the generator
                    self.generated_imgs = self(z)
                    valid = torch.ones(x.size(0), 1)
                    valid = valid.type_as(x)

                    d_outputs_fake,d_features_fake=self.discriminator(self.generator(z))
                    d_outputs_real,d_features_real=self.discriminator(x)

                    g_loss = self.feature_loss(torch.mean(d_features_fake), torch.mean(d_features_real))

                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()


                    g_loss_temp.append(g_loss.item())



                    ################
                    #
                    # step 2 train the discriminator
                    ############



                    valid = torch.ones(x.size(0), 1)
                    valid = valid.type_as(x)

                    y_real_pred,_=self.discriminator(x)
                    real_loss = self.adversarial_loss(y_real_pred, valid)

                    # how well can it label as fake?
                    fake = torch.zeros(x.size(0), 1)
                    fake = fake.type_as(x)

                    y_fake_pred,_=self.discriminator(self.generator(z).detach())

                    fake_loss = self.adversarial_loss(y_fake_pred, fake)

                    d_loss = (real_loss + fake_loss) / 2

                    d_optimizer.zero_grad()
                    d_loss.backward()
                    d_optimizer.step()

                    d_loss_temp.append(d_loss.item())

                    d_acc=(torch.sum(y_real_pred == valid)+torch.sum(y_fake_pred==fake))
                    d_accuracy_temp.append(d_acc.cpu())

                g_loss_history.append(np.mean(g_loss_temp))
                d_loss_history.append(np.mean(d_loss_temp))
                d_accuracy_history.append(np.mean(d_accuracy_temp))

            ## callbacks

            if wandb_log:
                wandb.log({"g_loss": g_loss_history[-1], "d_loss": d_loss_history[-1], "d_acc": d_accuracy_history[-1]})
                WandbImagesGAN(self, show=True)
            ## Save model
            if save_model is not None:
                SaveModel(self, save_model)

            ## Log to
