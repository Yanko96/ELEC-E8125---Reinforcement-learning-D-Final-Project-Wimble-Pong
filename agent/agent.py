import torch
import numpy as np 
import gym 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from statistics import mean,stdev
from torch.distributions import Categorical
import threading 

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    

class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n
    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2*ch)  # 32
        self.conv3 = ResDown(2*ch, 4*ch)  # 16
        self.conv4 = ResDown(4*ch, 8*ch)  # 8
        self.conv5 = ResDown(8*ch, 8*ch)  # 4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8*ch, z, 2, 2)  # 2

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.training:
            mu = self.conv_mu(x)
            log_var = self.conv_log_var(x)
            x = self.sample(mu, log_var)
        else:
            mu = self.conv_mu(x)
            x = mu
            log_var = None

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = ResUp(z, ch*8)
        self.conv2 = ResUp(ch*8, ch*8)
        self.conv3 = ResUp(ch*8, ch*4)
        self.conv4 = ResUp(ch*4, ch*2)
        self.conv5 = ResUp(ch*2, ch)
        self.conv6 = ResUp(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 


class RESNET_VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in, ch=64, z=512):
        super(RESNET_VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, z=z)
        self.decoder = Decoder(channel_in, ch=ch, z=z)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon = self.decoder(encoding)
        return recon, mu, log_var


class PolicyNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNet, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return Categorical(torch.softmax(self.policy(x.reshape(x.shape[0], -1)), dim=1))

class CriticNet(nn.Module):
    def __init__(self, input_shape):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.critic(x.reshape(x.shape[0], -1))

class Agent(nn.Module):
    def __init__(self, image_channels=3, ch=32, z_dim=128, num_action=2, vae_checkpoint_path=None, checkpoint_path=None):
        super(Agent, self).__init__()
        self.model_return = []
        self.cnn_vae = RESNET_VAE(image_channels, ch, z_dim)
        self.policy = PolicyNet(z_dim*3*3, num_action)
        self.critic = CriticNet(z_dim*3*3)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        if vae_checkpoint_path:
            self.cnn_vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=torch.device('cpu')))
    
    def choose_action(self,frame):
        # latent_codes = self.cnn_vae.forward(frame)
        x, mu, log_var = self.cnn_vae.encoder(frame)
        policy_prob = self.policy(mu)
        action = policy_prob.sample()
        # policy_log_prob = policy_prob.log_prob(action)
        # policy_log_probs.append(policy_prob.log_prob(action))
        value = self.critic(mu)
        # self.model_return.append((-categorical.log_prob(action),value))

        return action, value, policy_prob

    def load_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

class SimpleAgent(nn.Module):
    def __init__(self, input_size=6, num_action=2, checkpoint_path=None):
        super(SimpleAgent, self).__init__()
        self.model_return = []
        self.policy = PolicyNet(input_size, num_action)
        self.critic = CriticNet(input_size)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def choose_action(self, observation):
        policy_prob = self.policy(observation)
        action = policy_prob.sample()
        # policy_log_prob = policy_prob.log_prob(action)
        # policy_log_probs.append(policy_prob.log_prob(action))
        value = self.critic(observation)
        # self.model_return.append((-categorical.log_prob(action),value))

        return action, value, policy_prob

    def load_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)




