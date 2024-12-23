import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

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
        self.state_size = state_size
        self.action_size = action_size
        
        # Attention with matching embed_dim
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=hidden_size,  # Must match fc1 output
        #     num_heads=num_heads,
        #     batch_first=True
        # )
        
        # Keep hidden_size consistent
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        
        # Load GPT2 model with custom configurations
        config = GPT2Config(
            vocab_size=1,  # No vocabulary needed
            n_embd=hidden_size,  # Embedding size (matches hidden_size)
            n_layer=4,  # Number of transformer layers
            n_head=4,  # Number of attention heads
            resid_pdrop=0.1,  # Dropout for residual connections
            attn_pdrop=0.1,  # Dropout for attention
        )
        self.gpt2 = GPT2Model(config)

        # Linear layers for input and output processing
        self.fc_input = nn.Linear(state_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, action_size)

    def forward(self, state, attention_mask=None):
        '''
        # Convert to tensor if neededs
        x = F.relu(self.fc1(state))
        # x = x.unsqueeze(0)
        # attn_output, _ = self.attention(x, x, x)
        # x = attn_output.squeeze(0)
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        '''
        # Pass the input through the first fully connected layer
        
        # if state.dim() == 1:
        #     state = state.unsqueeze(0)  # Convert [state_size] -> [1, state_size]
        # elif state.dim() == 2:
        #     if state.size(1) != self.state_size:
        #         # Convert [state_size, seq_len] -> [seq_len, state_size]
        #         state = state.transpose(0, 1)  

        if state.dim() == 2:
            if state.size(1) == 1:
                # convert [3,1] into [3,]
                state = state.reshape(-1)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size, seq_length = state.shape[0], state.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        stacked_attention_mask = attention_mask
        # to make the attention mask fit the stacked inputs, have to stack it as well
        # In DT, it has action, state, return, so we have to stack it 3 times; but here we only have state
        # stacked_attention_mask = torch.stack(
        #     (attention_mask, attention_mask, attention_mask), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        
                
        # print("State size: ")
        # print(state.size())
        # print(state)

        embeddings = self.fc_input(state)

        # GPT2 expects input in sequence form, so unsqueeze batch dimension
        embeddings = embeddings.unsqueeze(1)  # [batch_size, seq_len=1, hidden_size]

        # Pass through GPT2 (you can add attention_mask if using sequences)
        gpt_output = self.gpt2(inputs_embeds=embeddings, attention_mask=stacked_attention_mask)

        # Take the hidden state from GPT2's output
        gpt_hidden = gpt_output.last_hidden_state.squeeze(1)  # [batch_size, hidden_size]

        # Map hidden state to action space
        mu = torch.tanh(self.fc_output(gpt_hidden))
        log_std = self.log_std_linear(gpt_hidden).clamp(self.log_std_min, self.log_std_max)
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
        # '''
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        # print("Action size: ")
        # print(action.size())
        # print(action)
        return action.detach().cpu()
        # '''
        # action = self.forward(state)
        # print(action)
        # return action.detach().cpu()
    
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
        
        if state.dim() == 2:
            if state.size(1) == 1:
                # convert [3,1] into [3,]
                state = state.reshape(-1)
        # print("State size: ")
        # print(state.size())
        # print('action size: ')
        # print(action.size())
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