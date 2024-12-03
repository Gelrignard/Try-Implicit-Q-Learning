import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3, log_std_min=-10, log_std_max=2, num_heads=4):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # First layer matches hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # Attention with matching embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,  # Must match fc1 output
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Keep hidden_size consistent
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.fc1.weight.device)
        
        # Handle different input shapes
        if state.ndim == 1:
            # If 1D tensor [state_size], reshape to [1, state_size]
            state = state.unsqueeze(0)
        elif state.ndim == 2 and state.shape[1] == 1:
            # If shape is [N,1], transpose to [1,N]
            state = state.transpose(0,1)
        
        # Now state should be [batch_size, state_size]
        x = F.relu(self.fc1(state))
        
        # Reshape for attention [batch_size, seq_len=1, hidden_size] 
        x = x.unsqueeze(1)
        
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.layer_norm(x)
        x = x.squeeze(1)
        
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()

        return action, dist
        
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, _ = self.forward(state)
        return mu.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, hidden_size=32):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)



    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)