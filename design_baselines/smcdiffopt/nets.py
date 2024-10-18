"""Implements the neural networks used in the diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedWithTime(nn.Module):
    """
    A simple model with multiple fully connected layers and some Fourier features for the time variable.
    Adapted from the jax code in smcdiffopt.
    
    Attributes:
        in_size: The size of the input tensor.
        time_embed_size: The size of the time embedding.
        max_t: The maximum time value.
    """
    
    def __init__(self, in_size: int, time_embed_size: int = 4, max_t: int = 999):
        super(FullyConnectedWithTime, self).__init__()
        out_size = in_size
        self.time_embed_size = time_embed_size
        self.max_t = max_t
        
        self.layers = nn.ModuleList([
            nn.Linear(in_size + self.time_embed_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, out_size),
        ])
        
    def _get_time_embedding(self, t):
        t = t / self.max_t
        device = t.device
        half_dim = self.time_embed_size // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        time_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return time_emb
        
    def forward(self, x, t):
        t_fourier = self._get_time_embedding(t)
        # rershape t_fourier to match the batch size
        t_fourier = t_fourier.expand(x.shape[0], -1).to(x.device)
        x = torch.cat([x, t_fourier], dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        x = self.layers[-1](x)
        
        return x
    

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal positional embeddings for time steps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        position = torch.arange(0, 1000).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(1000, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, timesteps):
        timesteps = timesteps.long()
        pe = self.pe[timesteps]
        return pe
    

class TinyConvNet(nn.Module):
    """
    A tiny ConvNet model for MNIST.
    """
    def __init__(self, time_dim=32):
        super().__init__()
        self.time_dim = time_dim

        # time embedding 
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU()
        )

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

        # batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

        self.time_emb_layers = nn.ModuleDict({
            'layer1': nn.Linear(time_dim * 4, 32),
            'layer2': nn.Linear(time_dim * 4, 64),
            'layer3': nn.Linear(time_dim * 4, 64),
            'layer4': nn.Linear(time_dim * 4, 32),
        })

    def forward(self, x, t):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Noisy input images of shape [batch_size, 1, 28, 28].
            t (torch.Tensor): Time steps of shape [batch_size], with values in [0, 999].

        Returns:
            torch.Tensor: Output images of shape [batch_size, 1, 28, 28].
        """
        # Compute time embeddings
        t_emb = self.time_mlp(t)  # Shape: [batch_size, time_dim * 4]

        
        h = self.conv1(x)
        h = self.bn1(h)
        t_emb_layer1 = self.time_emb_layers['layer1'](t_emb)
        h += t_emb_layer1[:, :, None, None]
        h = F.relu(h)

        
        h = self.conv2(h)
        h = self.bn2(h)
        t_emb_layer2 = self.time_emb_layers['layer2'](t_emb)
        h += t_emb_layer2[:, :, None, None]
        h = F.relu(h)

        
        h = self.conv3(h)
        h = self.bn3(h)
        t_emb_layer3 = self.time_emb_layers['layer3'](t_emb)
        h += t_emb_layer3[:, :, None, None]
        h = F.relu(h)

        
        h = self.conv4(h)
        h = self.bn4(h)
        t_emb_layer4 = self.time_emb_layers['layer4'](t_emb)
        h += t_emb_layer4[:, :, None, None]
        h = F.relu(h)

        
        out = self.conv_out(h)
        return out


# code from tutorial https://dataflowr.github.io/website/    
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)
    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])
    return embedding

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        x2 = self.up_scale(x2) # (bs,out_ch,2*w2,2*h2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # (bs,out_ch,w1,h1)
        x = torch.cat([x2, x1], dim=1) # (bs,2*out_ch,w1,h1)
        return x

class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch): # !! 2*out_ch = in_ch !!
        super(up_layer, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        a = self.up(x1, x2) # (bs,2*out_ch,w1,h1)
        x = self.conv(a) # (bs,out_ch,w1,h1) because 2*out_ch = in_ch
        return x
    
class SmallUNet(nn.Module):
    """
    Small UNet that can handle CIFAR10 images.
    """
    def __init__(self, in_channels=1, n_steps=1000, time_emb_dim=100):
        super(SmallUNet, self).__init__()
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)
        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, in_channels, 1)
        
        # time embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.te2 = self._make_te(time_emb_dim, 64)
        self.te3 = self._make_te(time_emb_dim, 128)
        self.te4 = self._make_te(time_emb_dim, 256)
        self.te5 = self._make_te(time_emb_dim, 512)
        self.te1_up = self._make_te(time_emb_dim, 1024)
        self.te2_up = self._make_te(time_emb_dim, 512)
        self.te3_up = self._make_te(time_emb_dim, 256)
        self.te4_up = self._make_te(time_emb_dim, 128)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
    def forward(self, x , t): # x (bs,in_channels,w,d)
        bs = x.shape[0]
        # if len(t) < bs:
        #     t = t.repeat(x.shape[0])
        if not t.dtype == torch.long:
            t = t.long()
        t = self.time_embed(t)
        
        x1 = self.conv1(x+self.te1(t).reshape(bs, -1, 1, 1)) # (bs,64,w,d)
        x2 = self.down1(x1+self.te2(t).reshape(bs, -1, 1, 1)) # (bs,128,w/2,d/2)
        x3 = self.down2(x2+self.te3(t).reshape(bs, -1, 1, 1)) # (bs,256,w/4,d/4)
        x4 = self.down3(x3+self.te4(t).reshape(bs, -1, 1, 1)) # (bs,512,w/8,h/8)
        x5 = self.down4(x4+self.te5(t).reshape(bs, -1, 1, 1)) # (bs,1024,w/16,h/16)
        x1_up = self.up1(x4, x5+self.te1_up(t).reshape(bs, -1, 1, 1)) # (bs,512,w/8,h/8)
        x2_up = self.up2(x3, x1_up+self.te2_up(t).reshape(bs, -1, 1, 1)) # (bs,256,w/4,h/4)
        x3_up = self.up3(x2, x2_up+self.te3_up(t).reshape(bs, -1, 1, 1)) # (bs,128,w/2,h/2)
        x4_up = self.up4(x1, x3_up+self.te4_up(t).reshape(bs, -1, 1, 1)) # (bs,64,w,h)
        output = self.last_conv(x4_up) # (bs,in_channels,w,h)
        return output