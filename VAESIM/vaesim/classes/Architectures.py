import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

##TODO: implement conditional decoder VQVAE, conditional Discriminator


#### SEPARABLE CONVOLUTIONS 2d


class DepthSepConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)



class DepthSepConvTranspose2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)




#### SEPARABLE CONVOLUTIONS 3D

class DepthSepConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)

        

class DepthSepConvTranspose3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)










def activation_func(activation):
    return nn.ModuleDict({
        'relu':nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'selu': nn.SELU(inplace=True),
        'none': nn.Identity(),
        'softmax':nn.Softmax(),
        'sigmoid':nn.Sigmoid(),
    })[activation]

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, in_channels, out_channels, ksize=(3, 3)):
        """
        Attributes:
        -------------
        :param in_channels: int, number of in_chanels features for the first conv2d block
        :param out_channels: int, number of out_channels in the conv2d blocks.
        :param ksize: tuple, default (3,3) -> kernel with shape of (3,3)
        Description:
        ------------
        A residual layer with two block of convolutions, batch normalization and relu activation
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, padding="same"),
            nn.BatchNorm2d(num_features=out_channels))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=ksize,
                               padding="same")  # in_channels=out_channels because it cames from previous computations
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation2 = nn.LeakyReLU(negative_slope=0.2)

    def should_apply_shortcut(self):
        """check if dimension of input and output are equal"""
        return self.in_channels != self.expanded_channels

    def forward(self, x):
        """

        :param inputs: Tensor, input data
        :return: Tensor, output of the residual connection
        """

        residual = x

        if self.should_apply_shortcut: residual = self.shortcut(x)  # just to match output dimensions

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = residual + x  # residual connection

        x = self.activation2(x)
        return x



class ConvResBlock(nn.Module):
    """
    Convolutional block followed by residual ones to make the network deeper
    """

    def __init__(self,n_conv_filters,n_res_block,n_res_filters):
        """

        :param n_conv_filters: int, number of convolutional filters (out_channels)
        :param n_res_block: int, number of residual layers
        :param n_res_filters: int, number of filters for convlutional layers inside the residual blocks
        """
        super().__init__()

        self.n_conv_filters=n_conv_filters
        self.n_res_block=n_res_block
        self.n_res_filters=n_res_filters

        self.conv=nn.LazyConv2d(out_channels=n_conv_filters,stride=2,kernel_size=4,padding=1)
        self.res_blocks=[]
        for i in range(n_res_block):
            if i==0:
                #first block takes n_conv_filters as input
                self.res_blocks.append(ResidualBlock(in_channels=n_conv_filters,out_channels=n_res_filters))
            else:
                #the others takes the output of precedent layers so n_res_filters
                self.res_blocks.append(ResidualBlock(in_channels=n_res_filters, out_channels=n_res_filters))

        self.residual_blocks=nn.Sequential(*self.res_blocks)

    def forward(self,x):
        """pass input trough convolutional and residual layers"""

        x=self.conv(x)
        x=self.residual_blocks(x)
        return x




class ConvTransposeResBlock(nn.Module):
    """
     Convolutional Transpose block followed by residual ones to make the network deeper
     """

    def __init__(self,n_conv_filters, n_res_block, n_res_filters,ksize=(4,4)):
        """

        :param n_conv_filters: int, number of convolutional filters
        :param n_res_block: int, number of residual layers
        :param n_res_filters: int, number of filters for conovlutional layers inside the residual blocks, should be the same as conv_layers
        """
        super().__init__()
        self.n_conv_filters = n_conv_filters
        self.n_res_block = n_res_block
        self.n_res_filters = n_res_filters


        self.conv=nn.LazyConvTranspose2d(out_channels=n_conv_filters,kernel_size=ksize,stride=2,padding=1)
        self.activation=nn.PReLU()
        #self.upsample=nn.Upsample(scale_factor=2)

        self.res_blocks = []
        for i in range(n_res_block):
            if i == 0:
                # first block takes n_conv_filters as input
                self.res_blocks.append(ResidualBlock(in_channels=n_conv_filters, out_channels=n_res_filters))
            else:
                # the others takes the output of precedent layers so n_res_filters
                self.res_blocks.append(ResidualBlock(in_channels=n_res_filters, out_channels=n_res_filters))

        self.residual_blocks = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        """pass input trough convolutional and residual layers"""

        x = self.conv(x)
        x = self.activation(x)
        #x = self.upsample(x)
        x = self.residual_blocks(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings



class Decoder(nn.Module):
    """Decoder generic class for generators
    """

    def __init__(self, target_shape ,latent_dim, conv_layer_list,):
        super().__init__()
        """
        Return a decoder/generator network
        :param latent_dim: dimension of the latent space
    
        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
        :param encoder_output_shape: tuple, required if version is "vqvae"
        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: decoder model
        """

        # infer the starting dimension.
        target_shape_side = target_shape[-1]
        self.startDim = target_shape_side // 2 ** len(conv_layer_list)
        self.first_channels=conv_layer_list[0][-1]

        self.predecoder=nn.Linear(latent_dim,self.first_channels*self.startDim*self.startDim)


        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvTransposeResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.decoder_output=nn.LazyConvTranspose2d(target_shape[0],3,padding=1)
        self.activation=nn.Sigmoid()

    def forward(self,x):
        x = self.predecoder(x)
        x = x.view(x.shape[0], self.first_channels, self.startDim, self.startDim)
        x = self.features(x)
        x = self.decoder_output(x)
        x = self.activation(x)

        return x



class VAEEncoder(nn.Module):

    """Encoder for VAE"""


    def __init__(self, latent_dim, conv_layer_list):
        """
        Return an encoder network
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)

        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: encoder model
        """
        super().__init__()

        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)


        self.flatten = nn.Flatten()
        self.mean = nn.LazyLinear(latent_dim)
        self.logvariance = nn.LazyLinear(latent_dim)


    def forward(self, x):

        x = self.features(x)


        x = self.flatten(x)
        mean = self.mean(x)
        logvariance = self.logvariance(x)

        return mean, logvariance


class cVAEEncoder(nn.Module):
    """Encoder for Conditional VAE"""

    def __init__(self, input_dim,latent_dim, conv_layer_list):
        """
        Return an encoder network
        :param input_dim: tuple, used to match the dimension of the condition
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)


        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: encoder model
        """
        super().__init__()


        self.condition=nn.Flatten()
        self.condition_dense=nn.LazyLinear(out_features=np.prod(input_dim))
        self.reshape=Reshape(-1,*input_dim)

        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.flatten = nn.Flatten()
        self.mean = nn.LazyLinear(latent_dim)
        self.logvariance = nn.LazyLinear(latent_dim)

    def forward(self, x,c):

        #condition
        c=self.condition(c)
        c=self.condition_dense(c)
        c=self.reshape(c)


        #concatenate
        x=torch.cat([x,c],dim=1)

        #encode
        x = self.features(x)

        x = self.flatten(x)
        mean = self.mean(x)
        logvariance = self.logvariance(x)

        return mean, logvariance





class VQVAEEncoder(nn.Module):

    """Encoder for VQVAE"""

    def __init__(self, latent_dim, conv_layer_list):
        """
        Return an encoder network
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)

        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: encoder model
        """
        super().__init__()

        feature_layers=[]
        for conv_filters in conv_layer_list:
            n_res_block,n_filters=conv_filters
            feature_layers.append(ConvResBlock(n_conv_filters=n_filters,n_res_block=n_res_block,n_res_filters=n_filters))

        self.features=nn.Sequential(*feature_layers)

        self.encoder_outputs=nn.LazyConv2d(out_channels=latent_dim,kernel_size=1,padding="same")


    def forward(self,x):

        x=self.features(x)
        x=self.encoder_outputs(x)

        return x




class cVQVAEEncoder(nn.Module):

    """Encoder for Conditional VQVAE"""

    def __init__(self, input_dim, latent_dim, conv_layer_list):
        """
        Return an encoder network
        :param input_dim: tuple, dimension of the input
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)

        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: encoder model
        """
        super().__init__()

        self.condition=nn.Flatten()
        self.condition_dense=nn.LazyLinear(out_features=np.prod(input_dim))
        self.reshape=Reshape(-1,*input_dim)

        feature_layers=[]
        for conv_filters in conv_layer_list:
            n_res_block,n_filters=conv_filters
            feature_layers.append(ConvResBlock(n_conv_filters=n_filters,n_res_block=n_res_block,n_res_filters=n_filters))

        self.features=nn.Sequential(*feature_layers)

        self.encoder_outputs=nn.LazyConv2d(out_channels=latent_dim,kernel_size=1,padding="same")


    def forward(self,x,c):

        #condition
        c=self.condition(c)
        c=self.condition_dense(c)
        c=self.reshape(c)


        #concatenate
        x=torch.cat([x,c],dim=1)

        x=self.features(x)
        x=self.encoder_outputs(x)

        return x




class VAEDecoder(nn.Module):
    """Decoder class for VAE"""

    def __init__(self, target_shape ,latent_dim, conv_layer_list,):
        super().__init__()
        """
        Return a decoder/generator network
        :param input_shape: tuple, the shape of the input
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
        :param encoder_output_shape: tuple, required if version is "vqvae"
        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: decoder model
        """

        # infer the starting dimension.
        target_shape_side = target_shape[-1]
        self.startDim = target_shape_side // 2 ** len(conv_layer_list)

        self.first_channels=conv_layer_list[0][-1]

        #self.predecoder=nn.Unflatten(self.first_channels*self.startDim*self.startDim)
        self.predecoder=nn.Linear(latent_dim,self.first_channels*self.startDim*self.startDim)
        self.unflatten=nn.Unflatten(1,(self.first_channels,self.startDim,self.startDim))
        #self.predecoder=Reshape(-1,self.startDim,self.startDim)

        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvTransposeResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.decoder_output=nn.LazyConvTranspose2d(target_shape[0],3,padding=1)
        self.activation=nn.Sigmoid()

    def forward(self,x):
        x = self.predecoder(x)
        x= self.unflatten(x)
        #x = x.view(x.shape[0], -1, self.startDim, self.startDim)
        x = self.features(x)
        x = self.decoder_output(x)
        x = self.activation(x)

        return x


class cVAEDecoder(nn.Module):
    """Decoder class for conditional VAE"""

    def __init__(self, target_shape ,latent_dim, conv_layer_list, condition_dim):
        super().__init__()
        """
        Return a decoder/generator network
        :param input_shape: tuple, the shape of the input
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
        :param encoder_output_shape: tuple, required if version is "vqvae"
        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: decoder model
        """


        self.condition_dim=condition_dim



        # infer the starting dimension.
        target_shape_side = target_shape[-1]



        self.startDim = target_shape_side // 2 ** len(conv_layer_list)

        self.first_channels=conv_layer_list[0][-1]

        #self.predecoder=nn.Unflatten(self.first_channels*self.startDim*self.startDim)
        self.predecoder=nn.Linear(latent_dim,self.first_channels*self.startDim*self.startDim)
        self.unflatten=nn.Unflatten(1,(self.first_channels,self.startDim,self.startDim))

        self.condition =  nn.Linear(self.condition_dim,self.startDim*self.startDim)
        self.condition2shape = nn.Unflatten(1, (1,self.startDim , self.startDim))
        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvTransposeResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.decoder_output=nn.LazyConvTranspose2d(target_shape[0],3,padding=1)
        self.activation=nn.Sigmoid()

    def forward(self,x,c):
        x = self.predecoder(x)
        x= self.unflatten(x)

        c= self.condition(c)
        c= self.condition2shape(c)

        x= torch.concat((x,c),axis=1)
        #x = x.view(x.shape[0], -1, self.startDim, self.startDim)
        x = self.features(x)
        x = self.decoder_output(x)
        x = self.activation(x)

        return x








class VQVAEDecoder(nn.Module):
    """Decoder class for VQ VAE"""

    def __init__(self, target_shape ,latent_dim, conv_layer_list,):
        super().__init__()
        """
        Return a decoder/generator network
        :param input_shape: tuple, the shape of the input
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
        :param encoder_output_shape: tuple, required if version is "vqvae"
        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: decoder model
        """

        # infer the starting dimension.
        #target_shape_side = target_shape[-1]
        #self.startDim = target_shape_side // 2 ** len(conv_layer_list)
        #self.first_channels=conv_layer_list[0][-1]

        #self.predecoder=nn.LazyLinear(self.first_channels*self.startDim*self.startDim)
        #self.reshape=Reshape(self.first_channels,self.startDim,self.startDim)
        #self.preactivation=nn.LeakyReLU(negative_slope=0.2)

        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvTransposeResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.decoder_output=nn.LazyConvTranspose2d(target_shape[0],3,padding=1)
        self.activation=nn.Sigmoid()

    def forward(self,x):

        #decode
        x = self.features(x)
        x = self.decoder_output(x)
        x = self.activation(x)

        return x



class VQVAEDecoder(nn.Module):
    """Decoder class for VQ VAE"""

    def __init__(self, target_shape ,latent_dim, conv_layer_list,):
        super().__init__()
        """
        Return a decoder/generator network
        :param input_shape: tuple, the shape of the input
        :param latent_dim: dimension of the latent space
        :param version: string, could be "vae" of "vqvae".
                        if "vae" it outputs a sampled vector and the parameter of the gaussian
                        if "vq vae" it outputs a Conv2D layer with latent_dim filters

        :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
        :param encoder_output_shape: tuple, required if version is "vqvae"
        for example
        conv_layer_list= [(2,64),(3,128),(1,256)]


        will result in a three block network
        block 1: Conv (64) -> Res(50) -> Res(50)
        block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
        block 3: conv (256) -> Res(200)

        :return: decoder model
        """

        # infer the starting dimension.
        #target_shape_side = target_shape[-1]
        #self.startDim = target_shape_side // 2 ** len(conv_layer_list)
        #self.first_channels=conv_layer_list[0][-1]

        #self.predecoder=nn.LazyLinear(self.first_channels*self.startDim*self.startDim)
        #self.reshape=Reshape(self.first_channels,self.startDim,self.startDim)
        #self.preactivation=nn.LeakyReLU(negative_slope=0.2)

        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvTransposeResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.decoder_output=nn.LazyConvTranspose2d(target_shape[0],3,padding=1)
        self.activation=nn.Sigmoid()

    def forward(self,x):

        #decode
        x = self.features(x)
        x = self.decoder_output(x)
        x = self.activation(x)

        return x





class Discriminator(nn.Module):


    def __init__(self, n_classes, conv_layer_list, dense=None, dropout=0.3, activation=None,matching=False):

        """
            Return a discriminator network
            :param n_classes: int, number of output classes
            :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
            :param dropout: float, percentage of dropout
            :param dense: int, number of units before the last dense layer
            :param activation: string, could be "relu" or "softmax" or "tanh" for example
            :param matching: bool, if true the forward return both the features and the output
            for example
            conv_layer_list= [(2,64),(3,128),(1,256)]


            will result in a three block network
            block 1: Conv (64) -> Res(50) -> Res(50)
            block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
            block 3: conv (256) -> Res(200)

            :return: discriminator model
            """

        super().__init__()




        self.matching=matching
        self.activation=activation




        feature_layers = []
        for conv_filters in conv_layer_list:
            n_res_block, n_filters = conv_filters
            feature_layers.append(
                ConvResBlock(n_conv_filters=n_filters, n_res_block=n_res_block, n_res_filters=n_filters))

        self.features = nn.Sequential(*feature_layers)

        self.flatten = nn.Flatten()


        if dense is not None:
            self.dense=nn.LazyLinear(dense)
            self.dropout=nn.Dropout(dropout)
            self.act_dense=nn.LeakyReLU(negative_slope=0.2)
            self.output=nn.LazyLinear(n_classes)
            self.classifier=nn.Sequential(self.flatten,self.dense,self.dropout,self.act_dense,self.output)
        else:
            self.output = nn.LazyLinear(n_classes)
            self.classifier = nn.Sequential(self.flatten,self.output)

        if activation is not None:
            self.activation=activation_func(activation)




    def forward(self,x):
        feat= self.features(x)
        x= self.classifier(feat)
        if self.activation is not None:
            x=self.activation(x)

        if self.matching:
            return x,self.flatten(feat)
        else:
            return x


######### VQ VAE 2 #################


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class VQVAE2Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)





class VQVAE2Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            embed_onehot_sum=torch.sum(embed_onehot_sum)            #ATTENZIONE forse mean
            embed_sum=torch.sum(embed_sum)                          #ATTENZIONE forse mean

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    
class DepthSepConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,                                                        groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)
    
    
    
class DepthSepConvTranspose2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,                                                        groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)
    
    
    

    
    
    
class DepthSepConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,                                                        groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)
    
    
class DepthSepConvTranspose3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,                                                        groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)
    
    
    
    
class DSConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=2,padding=1):
        super().__init__()
        
        self.out_channels=out_channels
        self.conv=DepthSepConv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.norm=nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
    
    
class DSConvTransposeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=2,padding=1):
        super().__init__()
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        
        self.conv=DepthSepConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.norm=nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x