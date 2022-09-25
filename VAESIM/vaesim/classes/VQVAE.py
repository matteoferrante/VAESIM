import torch
import tqdm
from torch import nn
from torch import distributions as D
from .Architectures import VQVAE2Encoder, VQVAE2Decoder, VectorQuantizer, VQVAEEncoder, VQVAEDecoder, Quantize,VectorQuantizerEMA
import numpy as np
from torch.nn import functional
import sys
sys.path.append('../utils')
from utils.callbacks import *

class VQVAE(nn.Module):
    """Class to implement vector quantized variational autoencoder
     from article: https://arxiv.org/abs/1711.00937

    The idea is to use a discrete space as latent space. This is called the codebook and it's learnable.
    This space consist in num_embeddings of latent_dim (for example: num_embeddings=100 with latent dim 32 means
    that the codebook is composed by 100 unique vectors with 32 components).

    The prior is learnable instead of a static gaussian (like p(z) in VAEs) and the latent space in quantized)

    loss function is basiacally the sum of three terms:

    reconstruction_loss+commitment_loss+codebook_loss.

    Decoder optimizes the first term,
    Encoder the first and the last term
    Embedding are optimized by the middle loss term

     """

    def __init__(self, input_dim, latent_dim=32, num_embeddings=128,encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]], decay=0.99,commitment_cost=0.25,data_variance=4,wandb_log=True,save_model=None,early_stop=None):
        """

        :param input_dim: input image dimension
        :param latent_dim: latent dimension in the embedding space
        :param num_embeddings: number of discrete vectors in the codebook
        :param train_variance: hyper-parameters to weight the reconstruction loss
        :param decay: float, parameter if > 0 the model will use Exponential Moving Average EMA
        :param commitment_cost: float, beta parameters of the paper
        """

        super(VQVAE, self).__init__()
        self.input_dim=input_dim
        self.latent_dim = latent_dim
        self.decay=decay
        self.commitment_cost=commitment_cost
        self.num_embeddings = num_embeddings

        self.data_variance=data_variance

        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture

        self.encoder=VQVAEEncoder(self.latent_dim,conv_layer_list=encoder_architecture)
        self.decoder=VQVAEDecoder(self.input_dim,self.latent_dim,conv_layer_list=decoder_architecture)

        self.patience=0 #for early stopping


        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, latent_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, latent_dim,
                                           commitment_cost)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity

    def fit(self, train_dataloader, val_dataloader=None, epochs=10, optimizer=None, device="cuda", wandb_log=True,
            save_model=None, early_stop=None):



        """Main train loop. The logic is all under model.train().
        After the training epoch there is validation loop and some "callbacks" to store weights, save outputs to W&B and visualize it


        """

        self.device=device #set in memory the device type
        model=self

        train_res_recon_error = []
        train_res_perplexity = []
        loss_history=[]

        val_train_res_recon_error = []
        val_train_res_perplexity = []
        val_loss_history=[]

        for epoch in range(epochs):

            # just store loss and accuracy for this epoch

            train_res_recon_error_temp = []
            train_res_perplexity_temp = []
            loss_temp=[]

            model.train()
            with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                for batch in tepoch:

                    if len(batch)==2:
                        x,y=batch
                    else:
                        x = batch

                    x = x.to(device)
                    optimizer.zero_grad()

                    vq_loss, data_recon, perplexity = model(x)
                    recon_error = functional.mse_loss(data_recon, x) / self.data_variance
                    loss = recon_error + vq_loss
                    loss.backward()

                    optimizer.step()

                    train_res_recon_error_temp.append(recon_error.item())
                    train_res_perplexity_temp.append(perplexity.item())
                    loss_temp.append(loss.item())
                    tepoch.set_postfix(loss=loss.item(), recon_loss=recon_error.item(), perplexity=perplexity.item())

            # store the metrics for epochs
            loss_history.append(np.mean(loss_temp))
            train_res_recon_error.append(np.mean(train_res_recon_error_temp))
            train_res_perplexity.append(np.mean(train_res_perplexity_temp))



            ## VALIDATION

            if wandb_log:
                wandb.log({"loss": loss_history[-1], "recon_loss": train_res_recon_error[-1], "perplexity": train_res_perplexity[-1]})

            if val_dataloader is not None:

                with torch.no_grad():
                    val_train_res_recon_error_temp = []
                    val_train_res_perplexity_temp = []
                    val_loss_temp = []

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
                            val_vq_loss, data_recon, val_perplexity = model(x)
                            val_recon_error = functional.mse_loss(data_recon, x) / self.data_variance

                            val_loss = val_recon_error + val_vq_loss

                            val_train_res_recon_error_temp.append(val_recon_error.item())
                            val_train_res_perplexity_temp.append(val_perplexity.item())
                            val_loss_temp.append(val_loss.item())
                            tepoch.set_postfix(val_loss=val_loss.item(), val_recon_loss=val_recon_error.item(), val_perplexity=val_perplexity.item())

                        # store the metrics for epochs
                        val_loss_history.append(np.mean(val_loss_temp))
                        val_train_res_recon_error.append(np.mean(val_train_res_recon_error_temp))
                        val_train_res_perplexity.append(np.mean(val_train_res_perplexity_temp))

            ## CALLBACKS
            if early_stop is not None:
                if len(val_loss_history) > 2:
                    if val_loss_history[-1] > val_loss_history[-2]:
                        self.patience += 1
                if self.patience == early_stop:
                    print("[INFO] Early Stopping")
                    break

            if wandb_log:
                wandb.log({"val_loss": val_loss_history[-1], "val_recon_loss": val_train_res_recon_error[-1], "val_perplexity": val_train_res_perplexity[-1]})


                WandbImagesVQVAE(self, x, show=True)

            if save_model is not None:
                SaveModel(self, save_model)










class VQVAE2(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = VQVAE2Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = VQVAE2Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = VQVAE2Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = VQVAE2Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        self.criterion=nn.MSELoss()
        self.latent_loss_weight=0.25

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


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
                    x = batch
                    x=x.to(model.device)

                    BS=len(x)

                    out, latent_loss = model(x)
                    recon_loss = self.criterion(out, x)
                    latent_loss = latent_loss.mean()
                    loss = recon_loss + self.latent_loss_weight * latent_loss
                    loss.backward()

                    part_mse_sum = recon_loss.item() * BS
                    comm = {"mse_sum": part_mse_sum, "mse_n": BS}

                    ## store the losses

                ## log the losses sum

                ## VALIDATION


                ##CALLBACKS
