---
slug: chapter-9-robot-learning-ai-integration
title: Chapter 9 - Robot Learning & AI Integration
description: Comprehensive guide to robot learning and AI integration for robotics
tags: [robot-learning, ai-integration, robotics, ai]
---

# 📚 Chapter 9: Robot Learning & AI Integration 📚

## 🎯 Learning Objectives 🎯

By the end of this chapter, students will be able to:
- Implement machine learning techniques for robot perception and control
- Design deep learning architectures for robotics applications
- Apply reinforcement learning algorithms to robotic tasks
- Integrate multiple AI systems for cohesive robot behavior
- Evaluate and validate AI models in simulation and real-world environments

## 👋 9.1 Introduction to Robot Learning 👋

Robot learning involves applying machine learning techniques to enable robots to acquire skills, adapt to new situations, and improve performance over time. This encompasses several approaches including supervised learning for perception tasks, reinforcement learning for control problems, and imitation learning for skill acquisition.

### 🎯 9.1.1 Types of Robot Learning 🎯

- **Supervised Learning**: Learning from labeled examples (e.g., object recognition from images)
- **Reinforcement Learning**: Learning through interaction with the environment to maximize rewards
- **Imitation Learning**: Learning by observing and mimicking human demonstrations
- **Unsupervised Learning**: Discovering patterns in data without labeled examples
- **Self-supervised Learning**: Learning from the structure of data itself

### 🎯 9.1.2 Challenges in Robot Learning 🎯

- Safety during learning (especially for physical robots)
- Sample efficiency (real-world data collection is expensive)
- Sim-to-real transfer (models trained in simulation must work on real robots)
- Real-time constraints (many robotic tasks require fast responses)
- Multi-modal integration (combining various sensor modalities)

## 🎯 9.2 Perception Systems with Deep Learning 🎯

Robots require accurate perception of their environment to make intelligent decisions. Deep learning has revolutionized computer vision and sensor processing for robotics.

### 👁️ 9.2.1 Convolutional Neural Networks for Vision 👁️

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotVisionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RobotVisionCNN, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 6 * 6)  # Adjust based on input size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

### ℹ️ 9.2.2 Semantic Segmentation for Scene Understanding ℹ️

Semantic segmentation helps robots understand the spatial layout of their environment by labeling each pixel in an image with its corresponding object class.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticSegmentationNet(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationNet, self).__init__()
        
        # Encoder (feature extraction)
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Decoder (upsampling to full resolution)
        self.dec_conv3 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_conv1 = nn.Conv2d(64, num_classes, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv1(x))
        x = self.pool(x1)
        
        x2 = F.relu(self.enc_conv2(x))
        x = self.pool(x2)
        
        x = F.relu(self.enc_conv3(x))
        
        # Decoder
        x = self.upsample(x)
        x = F.relu(self.dec_conv3(x + x2))  # Skip connection
        
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x + x1))  # Skip connection
        
        x = self.upsample(x)
        x = self.dec_conv1(x)
        
        return x
```

### ℹ️ 9.2.3 Depth Estimation ℹ️

Depth estimation is crucial for navigation and manipulation tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEstimationNet(nn.Module):
    def __init__(self):
        super(DepthEstimationNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        
        # Output layer for depth map
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        depth_map = self.output(x)
        return depth_map
```

## 🎯 9.3 Robot Control with Supervised Learning 🎯

Supervised learning can be used to learn robot control policies from demonstrations or to map sensor inputs to appropriate motor commands.

### ⚡ 9.3.1 Sensor-to-Action Mapping ⚡

```python
import torch
import torch.nn as nn
import numpy as np

class SensorActionNet(nn.Module):
    def __init__(self, sensor_dim, action_dim):
        super(SensorActionNet, self).__init__()
        
        self.fc1 = nn.Linear(sensor_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # Output actions (no activation for continuous control)
        return x

# 🤖 Example usage in a robot controller 🤖
class SupervisedRobotController:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SensorActionNet(sensor_dim=64, action_dim=12).to(self.device)
        
        # Load pre-trained model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def get_action(self, sensor_input):
        # Convert sensor input to tensor
        sensor_tensor = torch.FloatTensor(sensor_input).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            action = self.model(sensor_tensor)
        
        # Convert to numpy array for robot control
        return action.cpu().numpy().flatten()
```

### 🎛️ 9.3.2 PID Controllers Enhanced with ML 🎛️

Machine learning can be used to tune PID parameters or enhance traditional controllers:

```python
import numpy as np
from scipy import signal

class MLPIDController:
    def __init__(self, initial_kp=1.0, initial_ki=0.0, initial_kd=0.0):
        # Basic PID parameters
        self.kp = initial_kp
        self.ki = initial_ki
        self.kd = initial_kd
        
        # Error tracking
        self.previous_error = 0
        self.integral_error = 0
        
        # ML-enhanced parameter adjustment
        self.param_adjustment_model = self._create_param_adjustment_model()
        
    def _create_param_adjustment_model(self):
        # Simple model to adjust PID parameters based on system state
        # In practice, this could be a neural network trained on system data
        return None

    def update(self, error, dt):
        # Calculate PID terms
        proportional = self.kp * error
        self.integral_error += error * dt
        integral = self.ki * self.integral_error
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Update previous error
        self.previous_error = error
        
        # Calculate control output
        control_output = proportional + integral + derivative
        
        return control_output

    def adjust_parameters(self, system_state):
        # ML model adjusts the PID parameters based on current system state
        # This is a simplified placeholder implementation
        pass
```

## 🎯 9.4 Deep Learning Architectures for Robotics 🎯

### ℹ️ 9.4.1 Recurrent Neural Networks for Sequential Decision Making ℹ️

RNNs are valuable for tasks that require memory of past states, such as path planning and navigation.

```python
import torch
import torch.nn as nn

class RobotRNNController(nn.Module):
    def __init__(self, sensor_dim, action_dim, hidden_dim=128, num_layers=2):
        super(RobotRNNController, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN layer (using LSTM for better gradient flow)
        self.lstm = nn.LSTM(sensor_dim, hidden_dim, num_layers, batch_first=True)
        
        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize hidden state
        self.hidden = None

    def forward(self, sensor_input, hidden_state=None):
        # sensor_input shape: (batch_size, sequence_length, sensor_dim)
        lstm_out, self.hidden = self.lstm(sensor_input, hidden_state)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Predict action
        action = self.action_head(last_output)
        
        return action, self.hidden

    def reset_hidden_state(self):
        self.hidden = None
```

### 🧠 9.4.2 Attention Mechanisms for Multi-Modal Processing 🧠

Attention mechanisms can help robots focus on relevant information from multiple sensors.

```python
import torch
import torch.nn as nn

class MultiModalAttention(nn.Module):
    def __init__(self, modalities_dims, output_dim):
        super(MultiModalAttention, self).__init__()
        
        self.modalities_dims = modalities_dims
        self.num_modalities = len(modalities_dims)
        
        # Linear layers for each modality
        self.modality_transforms = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in modalities_dims
        ])
        
        # Attention computation
        self.attention_query = nn.Linear(output_dim, output_dim)
        self.attention_key = nn.Linear(output_dim, output_dim)
        self.attention_value = nn.Linear(output_dim, output_dim)
        
        # Output layer
        self.output_layer = nn.Linear(output_dim, output_dim)

    def forward(self, modalities):
        # modalities: list of tensors, each with shape (batch_size, dim)
        transformed_modalities = []
        
        # Transform each modality to common space
        for i, modality in enumerate(modalities):
            transformed = torch.relu(self.modality_transforms[i](modality))
            transformed_modalities.append(transformed)
        
        # Stack modalities for attention computation
        stacked = torch.stack(transformed_modalities, dim=1)  # (batch_size, num_modalities, output_dim)
        
        # Compute attention
        Q = self.attention_query(stacked)
        K = self.attention_key(stacked)
        V = self.attention_value(stacked)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Sum across modalities
        output = torch.sum(attended, dim=1)
        output = self.output_layer(output)
        
        return output, attention_weights
```

### 🧠 9.4.3 Convolutional LSTM for Spatiotemporal Processing 🧠

For tasks that involve both spatial and temporal information, ConvLSTM can be effective:

```python
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))
        
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(input_tensor.size(1)):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
        
        return layer_output_list[-1], hidden_state

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
```

## 🎯 9.5 Reinforcement Learning for Robotics 🎯

Reinforcement learning is particularly powerful for robotics as it allows robots to learn complex behaviors through interaction with the environment.

### ⚡ 9.5.1 Deep Q-Network (DQN) for Discrete Actions ⚡

For environments with discrete action spaces, DQN can be effective:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Update target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_dim))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### ⚡ 9.5.2 Policy Gradient Methods (PPO) for Continuous Actions ⚡

For robotic tasks with continuous action spaces, policy gradient methods like PPO are more appropriate:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic (value) network
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_extractor(state)
        
        # Actor
        action_mean = torch.tanh(self.actor_mean(features))  # Bound actions to [-1, 1]
        action_log_std = self.actor_std(features)
        action_std = torch.exp(action_log_std)
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.old_actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, value = self.old_actor_critic(state)
        
        # Sample action from normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def update(self, states, actions, rewards, dones, log_probs, values):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # Calculate discounted rewards (returns)
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy multiple times
        for _ in range(self.epochs):
            action_means, action_stds, new_values = self.actor_critic(states)
            
            # Calculate new log probabilities
            dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate objectives
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate critic loss (MSE between predicted values and returns)
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Update old policy
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())
```

### 🎯 9.5.3 Soft Actor-Critic (SAC) for Sample Efficient Learning 🎯

Soft Actor-Critic is known for its sample efficiency and stability in continuous control tasks:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(SACActor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        mean = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamping log_std
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()
        
        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = SACActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update parameter
        self.alpha = alpha  # Temperature parameter
        self.action_dim = action_dim
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        # Sample batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        not_done = torch.BoolTensor(not_done).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + not_done * self.gamma * target_q

        # Critic loss
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        pi, log_pi = self.actor.sample(state)
        q1, q2 = self.critic(state, pi)
        min_q = torch.min(q1, q2)
        
        actor_loss = ((self.alpha * log_pi) - min_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## 🤖 9.6 Sensor Fusion with AI 🤖

Sensor fusion combines data from multiple sensors to provide more accurate, reliable, and robust information than a single sensor could provide.

### 🔨 9.6.1 Kalman Filter Implementation 🔨

```python
import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State vector (e.g., [x, y, vx, vy] for 2D position and velocity)
        self.x = np.zeros((state_dim, 1))
        
        # State covariance matrix
        self.P = np.eye(state_dim)
        
        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 1.0
        
        # Measurement matrix (maps state to measurement space)
        self.H = np.zeros((measurement_dim, state_dim))
        
        # Control matrix (maps control input to state change)
        self.B = np.zeros((state_dim, 1)) if state_dim > 0 else None
        
        # State transition matrix
        self.F = np.eye(state_dim)

    def predict(self, u=None):
        """Prediction step - predict state and uncertainty"""
        # State prediction: x = F*x + B*u
        if u is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
        
        # Covariance prediction: P = F*P*F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x

    def update(self, z):
        """Update step - incorporate measurement"""
        # Innovation: y = z - H*x
        y = z - np.dot(self.H, self.x)
        
        # Innovation covariance: S = H*P*H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman gain: K = P*H^T*S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # State update: x = x + K*y
        self.x = self.x + np.dot(K, y)
        
        # Covariance update: P = (I - K*H)*P
        I = np.eye(len(self.x))
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x

# 🤖 Example for robot position tracking 🤖
class RobotKalmanFilter(KalmanFilter):
    def __init__(self):
        # State: [x, y, vx, vy] (position and velocity in 2D)
        super().__init__(state_dim=4, measurement_dim=2)
        
        # For a constant velocity model
        dt = 0.1  # Time step
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * 0.1
        
        # Measurement noise
        self.R = np.eye(2) * 0.5  # Measurement uncertainty

    def predict_position(self):
        return self.x[:2].flatten()  # Return [x, y] position
```

### 🔨 9.6.2 Particle Filter Implementation 🔨

Particle filters are useful for non-linear, non-Gaussian systems:

```python
import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, state_dim, measurement_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initialize particles randomly
        self.particles = np.random.rand(num_particles, state_dim) * 10  # Random initial states
        self.weights = np.ones(num_particles) / num_particles  # Uniform weights initially
        
    def predict(self, control_input, process_noise_std):
        """Predict step: move particles based on motion model"""
        # Add noise to each particle based on motion model and process noise
        noise = np.random.normal(0, process_noise_std, self.particles.shape)
        self.particles += noise
        
        # Or apply a more complex motion model based on control input
        # self.particles = motion_model(self.particles, control_input)
    
    def update(self, measurement, measurement_noise_std):
        """Update step: compute weights based on measurement likelihood"""
        # Calculate likelihood of each particle given the measurement
        # This is a simplified example for 2D position measurement
        for i in range(self.num_particles):
            # Calculate difference between particle's predicted measurement and actual measurement
            predicted_measurement = self.particles[i, :self.measurement_dim]  # Simplified
            diff = measurement - predicted_measurement
            
            # Calculate likelihood using Gaussian probability
            likelihood = np.exp(-0.5 * np.sum((diff)**2) / (measurement_noise_std**2))
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        """Resample particles based on their weights"""
        # Systematic resampling
        indices = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        u = np.random.uniform(0, 1/self.num_particles)
        
        i, j = 0, 0
        while i < self.num_particles:
            while u < cumulative_sum[j]:
                indices[i] = j
                u += 1/self.num_particles
                i += 1
            j += 1
        
        # Resample particles
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights after resampling
    
    def estimate(self):
        """Get state estimate as weighted average of particles"""
        return np.average(self.particles, weights=self.weights, axis=0)

# ℹ️ Example usage ℹ️
pf = ParticleFilter(num_particles=1000, state_dim=4, measurement_dim=2)  # 4D state, 2D measurement
```

### 🔗 9.6.3 Neural Network-Based Sensor Fusion 🔗

For more advanced fusion approaches using neural networks:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSensorFusion(nn.Module):
    def __init__(self, sensor_dims, output_dim):
        super(NeuralSensorFusion, self).__init__()
        
        self.sensor_dims = sensor_dims
        self.num_sensors = len(sensor_dims)
        
        # Process each sensor input separately
        self.sensor_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            ) for dim in sensor_dims
        ])
        
        # Fusion layer combining processed sensor data
        fusion_input_dim = 32 * self.num_sensors  # 32 features from each sensor processor
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Attention mechanism to weight sensor importance
        self.attention = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_sensors),
            nn.Softmax(dim=1)
        )

    def forward(self, sensor_inputs):
        # sensor_inputs: list of tensors, each tensor has shape [batch_size, sensor_dim]
        
        processed_sensors = []
        for i, sensor_input in enumerate(sensor_inputs):
            processed = self.sensor_processors[i](sensor_input)
            processed_sensors.append(processed)
        
        # Concatenate all processed sensor data
        concatenated = torch.cat(processed_sensors, dim=1)
        
        # Apply attention to weight sensor importance
        attention_weights = self.attention(concatenated)
        weighted_inputs = []
        
        for i, processed_sensor in enumerate(processed_sensors):
            weight = attention_weights[:, i].unsqueeze(1)  # Shape: [batch, 1]
            weighted_inputs.append(processed_sensor * weight)
        
        # Concatenate weighted inputs
        weighted_concatenated = torch.cat(weighted_inputs, dim=1)
        
        # Final fusion
        output = self.fusion_layer(weighted_concatenated)
        
        return output, attention_weights

# ℹ️ Example usage: ℹ️
# 📡 sensor_inputs = [lidar_data, camera_features, imu_data] 📡
# 🔗 fusion_model = NeuralSensorFusion(sensor_dims=[360, 512, 6], output_dim=128) 🔗
# 🔗 fused_output, attention_weights = fusion_model(sensor_inputs) 🔗
```

## 🎯 9.7 Learning from Demonstrations 🎯

Imitation learning allows robots to learn complex behaviors by observing human demonstrations.

### ℹ️ 9.7.1 Behavior Cloning ℹ️

Behavior cloning learns a direct mapping from states to actions using supervised learning:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BehaviorCloningNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(BehaviorCloningNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class BehaviorCloningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = BehaviorCloningNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, states, actions, epochs=10, batch_size=64):
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(states_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.network.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predicted_actions = self.network(batch_states)
                
                # Compute loss
                loss = self.criterion(predicted_actions, batch_actions)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(self, state):
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.network(state_tensor)
            return action.cpu().numpy().squeeze()
    
    def save_model(self, filepath):
        torch.save(self.network.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
```

### 📊 9.7.2 DAgger (Dataset Aggregation) 📊

DAgger addresses the distribution shift problem in behavior cloning:

```python
import numpy as np
from collections import deque

class DAggerAgent:
    def __init__(self, state_dim, action_dim, expert_policy, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expert_policy = expert_policy  # Function that takes state and returns expert action
        self.learning_rate = learning_rate
        
        self.network = BehaviorCloningNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Buffer to store aggregated dataset
        self.state_buffer = deque(maxlen=10000)
        self.action_buffer = deque(maxlen=10000)
        
    def aggregate_data(self, states, actions_from_expert=True):
        """Add state-action pairs to training dataset"""
        if actions_from_expert:
            # Add state-expert_action pairs
            for state in states:
                expert_action = self.expert_policy(state)
                self.state_buffer.append(state)
                self.action_buffer.append(expert_action)
        else:
            # Add state-expert_action pairs (actions from current policy were corrected by expert)
            for state, action in zip(states, self.predict_batch(states)):
                expert_action = self.expert_policy(state)
                self.state_buffer.append(state)
                self.action_buffer.append(expert_action)
    
    def predict_batch(self, states):
        """Predict actions for a batch of states"""
        self.network.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            actions = self.network(states_tensor)
            return actions.numpy()
    
    def predict(self, state):
        """Predict action for a single state"""
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.network(state_tensor)
            return action.numpy().squeeze()
    
    def train(self, epochs=10):
        if len(self.state_buffer) < 100:  # Need minimum amount of data
            return
            
        # Convert buffers to tensors
        states_tensor = torch.FloatTensor(list(self.state_buffer))
        actions_tensor = torch.FloatTensor(list(self.action_buffer))
        
        # Create dataset and dataloader
        dataset = TensorDataset(states_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.network.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predicted_actions = self.network(batch_states)
                
                # Compute loss
                loss = self.criterion(predicted_actions, batch_actions)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"DAgger Training - Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def get_expert_actions(self, states):
        """Get expert actions for given states"""
        return [self.expert_policy(state) for state in states]
```

### 🎯 9.7.3 Generative Adversarial Imitation Learning (GAIL) 🎯

GAIL learns policies by matching the state-action distribution of expert demonstrations:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class GAILAgent:
    def __init__(self, policy_network, state_dim, action_dim, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_network = policy_network
        self.discriminator = Discriminator(state_dim, action_dim)
        
        self.policy_optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        self.criterion = nn.BCELoss()
        
    def compute_reward(self, state, action):
        """Compute reward as -log(D(s,a)) from discriminator"""
        with torch.no_grad():
            prob_expert = self.discriminator(state, action)
            # Return log(D(s,a)/(1-D(s,a))) which is equivalent to log(D) - log(1-D)
            # But more commonly used is just -log(1-D(s,a)) which encourages D to go to 1
            reward = -torch.log(1 - prob_expert + 1e-8)  # Add small epsilon to avoid log(0)
            return reward.squeeze()
    
    def discriminator_loss(self, expert_states, expert_actions, policy_states, policy_actions):
        # Expert data should be classified as expert (1)
        expert_labels = torch.ones(expert_states.size(0), 1)
        expert_loss = self.criterion(
            self.discriminator(expert_states, expert_actions), 
            expert_labels
        )
        
        # Policy data should be classified as not expert (0)
        policy_labels = torch.zeros(policy_states.size(0), 1)
        policy_loss = self.criterion(
            self.discriminator(policy_states, policy_actions), 
            policy_labels
        )
        
        return expert_loss + policy_loss
    
    def update_discriminator(self, expert_states, expert_actions, policy_states, policy_actions):
        """Update discriminator to better distinguish expert vs policy data"""
        self.discriminator_optimizer.zero_grad()
        
        loss = self.discriminator_loss(
            expert_states, expert_actions, 
            policy_states, policy_actions
        )
        
        loss.backward()
        self.discriminator_optimizer.step()
        
        return loss.item()
    
    def update_policy(self, states, actions, rewards):
        """Update policy to maximize discriminator confusion (minimize discriminator output)"""
        self.policy_optimizer.zero_grad()
        
        # Get new actions from updated policy
        new_actions = self.policy_network(states)
        
        # Discriminator should output low value for policy actions
        disc_output = self.discriminator(states, new_actions)
        policy_loss = -torch.log(disc_output + 1e-8).mean()  # Maximize log(1-D) equivalent
        
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()

# ℹ️ Example usage: ℹ️
# ⚡ policy_network = SomePolicyNetwork(state_dim, action_dim) ⚡
# 🤖 gail_agent = GAILAgent(policy_network, state_dim, action_dim) 🤖
```

## 🤖 9.8 Navigation and Path Planning with AI 🤖

Robot navigation requires planning efficient, collision-free paths through environments.

### ℹ️ 9.8.1 A* Algorithm with Neural Heuristics ℹ️

Traditional A* can be enhanced with learned heuristics:

```python
import heapq
import numpy as np

class NeuralHeuristicAStar:
    def __init__(self, grid_map, neural_heuristic_model):
        self.grid_map = grid_map  # 2D grid where 0=free, 1=obstacle
        self.neural_heuristic = neural_heuristic_model  # Pre-trained neural network
        self.rows, self.cols = grid_map.shape
        
    def heuristic(self, pos, goal):
        """Neural network-based heuristic function"""
        # Convert position and goal to feature vector for the neural network
        features = np.array([pos[0], pos[1], goal[0], goal[1]]).astype(np.float32)
        features = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            heuristic_value = self.neural_heuristic(features)
        
        return heuristic_value.item()
    
    def get_neighbors(self, pos):
        """Get valid neighbors for a position"""
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid_map[nr][nc] == 0:
                # Add cost based on grid value (for weighted grids)
                move_cost = 1.0 if abs(dr) + abs(dc) == 1 else 1.414  # Diagonal moves cost more
                neighbors.append(((nr, nc), move_cost))
        return neighbors
    
    def plan_path(self, start, goal):
        """Plan a path from start to goal using A* with neural heuristic"""
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        
        # Costs: g_score (actual cost from start) and f_score (g + heuristic)
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # Track path parents
        came_from = {}
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get path from start to goal
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = current_g + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        return None  # No path found
```

### 🎯 9.8.2 Learning-based Path Planning 🎯

Using neural networks to directly learn navigation policies:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NavigationPolicy(nn.Module):
    def __init__(self, map_channels=1, goal_dim=2, hidden_dim=128, action_dim=4):
        super(NavigationPolicy, self).__init__()
        
        # CNN to process map information
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # Reduce spatial dimensions
        )
        
        # Process goal information
        self.goal_fc = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU()
        )
        
        # Combine map and goal features
        cnn_output_size = 64 * 8 * 8  # After CNN and pooling
        combined_input_size = cnn_output_size + 64  # + goal features
        
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers for action and value
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, map_tensor, goal_pos):
        # Process map with CNN
        map_features = self.map_cnn(map_tensor)
        map_features = map_features.view(map_features.size(0), -1)  # Flatten
        
        # Process goal
        goal_features = self.goal_fc(goal_pos)
        
        # Combine features
        combined = torch.cat([map_features, goal_features], dim=1)
        
        # Process combined features
        hidden = self.combined_layers(combined)
        
        # Output action logits and value
        action_logits = self.action_head(hidden)
        value = self.value_head(hidden)
        
        return action_logits, value

class LearningBasedNavigator:
    def __init__(self, policy_network, device='cpu'):
        self.policy_network = policy_network
        self.device = device
        
    def get_action(self, map_tensor, goal_pos, available_actions=None):
        self.policy_network.eval()
        
        with torch.no_grad():
            map_tensor = map_tensor.to(self.device).unsqueeze(0)  # Add batch dimension
            goal_pos = goal_pos.to(self.device).unsqueeze(0)  # Add batch dimension
            
            action_logits, _ = self.policy_network(map_tensor, goal_pos)
            
            if available_actions is not None:
                # Mask out unavailable actions
                masked_logits = torch.full_like(action_logits, float('-inf'))
                masked_logits[0, available_actions] = action_logits[0, available_actions]
                action_probs = F.softmax(masked_logits, dim=-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample action from distribution or take argmax
            action = torch.multinomial(action_probs, 1).item()
        
        return action
```

### 🎯 9.8.3 RRT* with Learning Enhancement 🎯

Rapidly-exploring Random Trees (RRT*) can be enhanced with learning to guide exploration:

```python
import numpy as np
import random
from scipy.spatial import KDTree

class LearningEnhancedRRTStar:
    def __init__(self, bounds, start, goal, obstacle_list, goal_bias=0.05):
        self.bounds = bounds  # [(min_x, max_x), (min_y, max_y), ...]
        self.start = start
        self.goal = goal
        self.obstacles = obstacle_list
        self.goal_bias = goal_bias  # Probability of sampling goal
        
        # For RRT*: radius for choosing parents and rewiring
        self.r = 1.5
        
        # Graph storage
        self.vertices = [start]
        self.edges = {}  # vertex -> list of (neighbor, cost)
        self.costs = {tuple(start): 0.0}  # vertex -> cost from start
        self.parents = {tuple(start): None}
        
        # Learning component: bias sampling toward promising areas
        self.visit_counts = {tuple(start): 0}
        
    def is_collision_free(self, point1, point2):
        """Check if path between two points is collision-free"""
        # Simple implementation; in practice, you'd want more sophisticated collision checking
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            point = point1 * (1 - t) + point2 * t
            
            for obs in self.obstacles:
                if np.linalg.norm(point - obs[:2]) <= obs[2]:  # Assuming circular obstacles
                    return False
        return True
    
    def sample_free(self):
        """Sample a free configuration"""
        # With some probability, sample the goal
        if random.random() < self.goal_bias:
            return self.goal
        
        # Otherwise, sample based on learned preferences
        # For now, uniform random sampling with bounds
        point = np.array([random.uniform(bound[0], bound[1]) for bound in self.bounds])
        
        # Check if point is in free space
        for obs in self.obstacles:
            if np.linalg.norm(point - obs[:2]) <= obs[2]:
                # If in obstacle, try again
                return self.sample_free()
        
        return point
    
    def nearest_vertex(self, point):
        """Find nearest vertex in tree to given point"""
        points = np.array(self.vertices)
        tree = KDTree(points)
        dist, idx = tree.query(point)
        return self.vertices[idx]
    
    def extend_tree(self, new_point):
        """Extend the tree toward a new point"""
        nearest = self.nearest_vertex(new_point)
        
        # Try to connect to all vertices within radius that are collision-free
        valid_parents = []
        for vertex in self.vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(new_point))
            if dist <= self.r and self.is_collision_free(vertex, new_point):
                total_cost = self.costs[tuple(vertex)] + dist
                valid_parents.append((vertex, total_cost))
        
        if not valid_parents:
            return False  # Cannot connect
        
        # Choose parent with minimum cost
        parent, min_cost = min(valid_parents, key=lambda x: x[1])
        
        # Add new vertex and edge
        self.vertices.append(new_point)
        parent_key = tuple(parent)
        new_point_key = tuple(new_point)
        
        if parent_key not in self.edges:
            self.edges[parent_key] = []
        self.edges[parent_key].append((new_point, np.linalg.norm(np.array(parent) - np.array(new_point))))
        
        self.costs[new_point_key] = min_cost
        self.parents[new_point_key] = parent_key
        
        # Rewire: Check if we can improve costs of nearby vertices through new point
        for vertex in self.vertices[:-1]:  # Exclude the new point
            vertex_key = tuple(vertex)
            dist_to_new = np.linalg.norm(np.array(vertex) - np.array(new_point))
            if dist_to_new <= self.r and self.is_collision_free(new_point, vertex):
                new_cost = min_cost + dist_to_new
                if new_cost < self.costs[vertex_key]:
                    # Update parent of vertex
                    old_parent = self.parents[vertex_key]
                    self.parents[vertex_key] = new_point_key
                    
                    # Update edges: remove old edge, add new one
                    if old_parent in self.edges:
                        self.edges[old_parent] = [(v, c) for v, c in self.edges[old_parent] if not np.array_equal(v, vertex)]
                    
                    if new_point_key not in self.edges:
                        self.edges[new_point_key] = []
                    self.edges[new_point_key].append((vertex, dist_to_new))
                    
                    # Update cost
                    self.costs[vertex_key] = new_cost
        
        return True
    
    def plan(self, max_iterations=1000):
        """Plan a path using RRT*"""
        for i in range(max_iterations):
            new_point = self.sample_free()
            self.extend_tree(new_point)
            
            # Check if we've reached the goal region
            for vertex in self.vertices:
                if np.linalg.norm(np.array(vertex) - np.array(self.goal)) < 0.5:  # Goal region threshold
                    return self.reconstruct_path(tuple(self.goal))
        
        return None  # No path found
    
    def reconstruct_path(self, goal_point_key):
        """Reconstruct path from goal to start"""
        path = []
        current = goal_point_key
        
        while current is not None:
            path.append(np.array(current))
            current = self.parents[current]
        
        return path[::-1]  # Reverse to get path from start to goal
```

## 🎯 9.9 Human-Robot Interaction Learning 🎯

### 🎯 9.9.1 Learning from Human Feedback (LfHF) 🎯

Learning from human feedback allows robots to align their behavior with human preferences:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PreferenceModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PreferenceModel, self).__init__()
        
        self.state_action_dim = state_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(self.state_action_dim * 2, hidden_dim),  # Two state-action pairs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output preference probability
            nn.Sigmoid()
        )
    
    def forward(self, state1, action1, state2, action2):
        # Concatenate both state-action pairs
        sa1 = torch.cat([state1, action1], dim=-1)
        sa2 = torch.cat([state2, action2], dim=-1)
        combined = torch.cat([sa1, sa2], dim=-1)
        
        return self.network(combined)

class LearningFromHumanFeedback:
    def __init__(self, state_dim, action_dim, policy_network, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.preference_model = PreferenceModel(state_dim, action_dim)
        self.policy_network = policy_network  # The policy being trained
        
        self.optimizer = optim.Adam(self.preference_model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def get_preference_prediction(self, state1, action1, state2, action2):
        """Get the model's prediction of which state-action pair is preferred"""
        with torch.no_grad():
            prob = self.preference_model(state1, action1, state2, action2)
            return prob.item()
    
    def update_preference_model(self, preferred_pairs, unpreferred_pairs):
        """Update the preference model based on human feedback"""
        # preferred_pairs and unpreferred_pairs are lists of (state, action) tuples
        # where preferred_pairs[0] is preferred to unpreferred_pairs[0], etc.
        
        # Create batch tensors
        pref_states = torch.stack([pair[0] for pair in preferred_pairs])
        pref_actions = torch.stack([pair[1] for pair in preferred_pairs])
        unprefer_states = torch.stack([pair[0] for pair in unpreferred_pairs])
        unprefer_actions = torch.stack([pair[1] for pair in unpreferred_pairs])
        
        self.optimizer.zero_grad()
        
        # Predict preference: probability that first option is preferred
        prob_first_preferred = self.preference_model(
            pref_states, pref_actions, 
            unprefer_states, unprefer_actions
        )
        
        # Since first is preferred, target should be 1
        target = torch.ones_like(prob_first_preferred)
        
        loss = self.criterion(prob_first_preferred, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_action_rankings(self, state, action_candidates):
        """Rank a set of action candidates for a given state"""
        rankings = []
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        for action in action_candidates:
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            # Compare each action to a baseline (could be random or policy action)
            baseline_action = torch.randn_like(action_tensor)  # Random baseline
            
            preference = self.get_preference_prediction(
                state_tensor, action_tensor,
                state_tensor, baseline_action
            )
            
            rankings.append((action, preference))
        
        # Sort by preference (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
```

### 🎯 9.9.2 Interactive Robot Learning 🎯

Interactive learning allows robots to ask questions to clarify human intentions:

```python
class InteractiveLearningAgent:
    def __init__(self, robot_policy, question_generation_model):
        self.policy = robot_policy
        self.question_model = question_generation_model
        self.uncertainty_threshold = 0.2  # Threshold for asking questions
        
        # Track learned preferences/parameters
        self.human_preferences = {}
        self.task_parameters = {}
        
    def should_ask_question(self, state, action_distribution):
        """Determine if the robot should ask a question based on uncertainty"""
        # Calculate entropy of action distribution as uncertainty measure
        entropy = -sum(p * np.log(p + 1e-10) for p in action_distribution)
        max_entropy = np.log(len(action_distribution))  # Max possible entropy
        
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy > self.uncertainty_threshold
    
    def generate_question(self, state, current_task):
        """Generate an appropriate question based on state and task"""
        # This is a simplified example; in practice, this could be a learned model
        # that generates questions based on the state and current task uncertainty
        
        # Based on state and task, determine what information is most needed
        if current_task == "navigation":
            # Generate navigation-specific questions
            return {
                "type": "preference_query",
                "question": "Which path would you prefer: the shortest or the safest?",
                "options": ["shortest", "safest"]
            }
        elif current_task == "manipulation":
            # Generate manipulation-specific questions
            return {
                "type": "parameter_query", 
                "question": "How gently should I grasp the object?",
                "range": [0.1, 1.0]  # Force range from gentle to firm
            }
        else:
            # Default question
            return {
                "type": "clarification",
                "question": f"What would you like me to do now?",
                "options": ["continue", "stop", "repeat"]
            }
    
    def incorporate_feedback(self, question, answer):
        """Incorporate human feedback into the model"""
        if question["type"] == "preference_query":
            self.human_preferences[question["question"]] = answer
        elif question["type"] == "parameter_query":
            self.task_parameters[question["question"]] = answer
        elif question["type"] == "clarification":
            self.task_parameters["next_action"] = answer
    
    def get_adapted_action(self, state, current_task):
        """Get action adapted based on learned preferences"""
        # First, check if we should ask a question
        action_probs = self.policy.get_action_probabilities(state)
        
        if self.should_ask_question(state, action_probs):
            question = self.generate_question(state, current_task)
            # In a real implementation, you'd present this to the human
            # For now, we'll simulate getting an answer
            simulated_answer = self.simulate_human_answer(question)
            self.incorporate_feedback(question, simulated_answer)
            
            # Recompute action based on incorporated feedback
            adapted_action_probs = self.policy.get_adapted_action_probabilities(
                state, self.human_preferences, self.task_parameters
            )
            return np.random.choice(len(adapted_action_probs), p=adapted_action_probs)
        else:
            # Just take the most probable action
            return np.argmax(action_probs)
    
    def simulate_human_answer(self, question):
        """Simulate human response (in real implementation, this would be actual human input)"""
        if question["type"] == "preference_query":
            return question["options"][0]  # Simulate always preferring the first option
        elif question["type"] == "parameter_query":
            return question["range"][1] * 0.7  # Simulate preferring 70% of max value
        else:  # clarification
            return question["options"][0]
```

## 📊 9.10 Model Evaluation and Validation 📊

### 🎯 9.10.1 Metrics for Robot Learning 🎯

Evaluating robot learning systems requires specialized metrics:

```python
import numpy as np
import matplotlib.pyplot as plt

class RobotLearningEvaluator:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.collision_rates = []
        
    def add_episode_data(self, reward, length, success, collision):
        """Add data from a completed episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(success)
        self.collision_rates.append(collision)
    
    def compute_episode_metrics(self):
        """Compute metrics based on collected episode data"""
        if not self.episode_rewards:
            return {}
        
        metrics = {
            'average_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'average_length': np.mean(self.episode_lengths),
            'success_rate': np.mean(self.success_rates),
            'collision_rate': np.mean(self.collision_rates),
            'total_episodes': len(self.episode_rewards)
        }
        
        return metrics
    
    def compute_safety_metrics(self):
        """Compute safety-related metrics"""
        if not self.collision_rates:
            return {}
        
        return {
            'collision_rate': np.mean(self.collision_rates),
            'min_distance_to_obstacles': np.min(self.episode_rewards) if self.episode_rewards else float('inf'),
            'average_velocity_magnitude': np.mean([np.linalg.norm(r) for r in self.episode_rewards]) if self.episode_rewards else 0
        }
    
    def compute_efficiency_metrics(self):
        """Compute efficiency metrics"""
        if not self.episode_lengths or not self.success_rates:
            return {}
        
        successful_episodes = [l for l, s in zip(self.episode_lengths, self.success_rates) if s]
        
        return {
            'average_completion_time': np.mean(successful_episodes) if successful_episodes else float('inf'),
            'task_completion_rate': np.sum(self.success_rates) / len(self.success_rates) if self.success_rates else 0,
            'average_path_efficiency': self.compute_path_efficiency()  # Placeholder implementation
        }
    
    def compute_path_efficiency(self):
        """Compute how direct paths are compared to optimal"""
        # Placeholder: In practice, you'd compare actual path length to optimal path length
        return 0.85  # Example efficiency value
    
    def plot_learning_curves(self):
        """Plot learning curves"""
        if not self.episode_rewards:
            return
            
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot cumulative rewards
        cumulative_rewards = np.cumsum(self.episode_rewards)
        axes[0, 0].plot(cumulative_rewards)
        axes[0, 0].set_title('Cumulative Rewards Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward')
        
        # Plot rolling average of rewards
        if len(self.episode_rewards) > 10:
            rolling_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(rolling_avg)
            axes[0, 1].set_title('Rolling Average of Rewards (window=10)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Plot success rate over time
        if len(self.success_rates) > 10:
            rolling_success = np.convolve(self.success_rates, np.ones(10)/10, mode='valid')
            axes[1, 0].plot(rolling_success)
            axes[1, 0].set_title('Rolling Average Success Rate (window=10)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_ylim(0, 1)
        
        # Plot collision rate over time
        if len(self.collision_rates) > 10:
            rolling_collision = np.convolve(self.collision_rates, np.ones(10)/10, mode='valid')
            axes[1, 1].plot(rolling_collision, color='red')
            axes[1, 1].set_title('Rolling Average Collision Rate (window=10)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Collision Rate')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return fig
```

### 🎮 9.10.2 Simulation-to-Real Transfer Validation 🎮

Validating that models trained in simulation work in the real world:

```python
class SimToRealValidator:
    def __init__(self, sim_model, real_robot_interface, domain_randomization_params=None):
        self.sim_model = sim_model
        self.real_robot = real_robot_interface
        self.domain_params = domain_randomization_params or {}
        
        # Performance tracking
        self.sim_performance = {}
        self.real_performance = {}
        
    def validate_perception(self, test_data_sim, test_data_real):
        """Validate perception models between sim and real"""
        # Run perception on simulation data
        sim_predictions = self.sim_model.predict(test_data_sim)
        
        # Run on real data (requires real perception model)
        real_predictions = self.real_robot.run_perception(test_data_real)
        
        # Compare results
        perception_accuracy = self.compute_similarity(sim_predictions, real_predictions)
        
        return {
            'sim_accuracy': sim_predictions.accuracy if hasattr(sim_predictions, 'accuracy') else None,
            'real_accuracy': real_predictions.accuracy if hasattr(real_predictions, 'accuracy') else None,
            'sim_real_similarity': perception_accuracy
        }
    
    def validate_control_policy(self, num_trials=20):
        """Validate control policy between sim and real"""
        sim_rewards = []
        real_rewards = []
        
        # Test in simulation
        for _ in range(num_trials):
            episode_reward = self.evaluate_policy_in_sim(self.sim_model)
            sim_rewards.append(episode_reward)
        
        # Test on real robot
        for _ in range(num_trials):
            episode_reward = self.evaluate_policy_on_real(self.sim_model)
            real_rewards.append(episode_reward)
        
        return {
            'sim_mean_reward': np.mean(sim_rewards),
            'sim_std_reward': np.std(sim_rewards),
            'real_mean_reward': np.mean(real_rewards), 
            'real_std_reward': np.std(real_rewards),
            'sim_real_correlation': np.corrcoef(sim_rewards, real_rewards)[0, 1] if len(sim_rewards) > 1 else 0
        }
    
    def compute_similarity(self, sim_output, real_output):
        """Compute similarity between sim and real outputs"""
        # This is a simplified approach; in practice, you'd use domain-specific similarity measures
        if hasattr(sim_output, 'features') and hasattr(real_output, 'features'):
            # Compare feature vectors using cosine similarity
            sim_features = np.array(sim_output.features)
            real_features = np.array(real_output.features)
            
            cosine_sim = np.dot(sim_features, real_features) / (
                np.linalg.norm(sim_features) * np.linalg.norm(real_features)
            )
            return cosine_sim
        else:
            # Fallback to comparing raw outputs if feature extraction isn't available
            return np.mean(np.abs(sim_output - real_output))  # Mean absolute difference
            
    def evaluate_policy_in_sim(self, policy_model):
        """Evaluate policy in simulation environment"""
        # This would involve running the policy in your simulation environment
        # For this example, we'll use a placeholder implementation
        return np.random.normal(100, 10)  # Placeholder return value
    
    def evaluate_policy_on_real(self, policy_model):
        """Evaluate policy on real robot"""
        # This would involve deploying the policy to the real robot
        # and measuring task performance
        # For this example, we'll use a placeholder implementation
        return np.random.normal(85, 15)  # Placeholder return value (typically lower than sim)
    
    def apply_domain_randomization(self):
        """Apply domain randomization to improve sim-to-real transfer"""
        # Randomize physics parameters in simulation
        physics_params = {
            'gravity': np.random.uniform(9.0, 10.0),
            'friction': np.random.uniform(0.1, 0.9),
            'mass_variance': np.random.uniform(0.9, 1.1),
            'sensor_noise': np.random.uniform(0.01, 0.05)
        }
        
        # Apply parameters to simulation
        self.sim_model.set_physics_parameters(physics_params)
        
        # Randomize visual appearance
        visual_params = {
            'lighting': np.random.uniform(0.5, 1.5),
            'texture_randomization': np.random.choice([True, False]),
            'color_variance': np.random.uniform(0.8, 1.2)
        }
        
        self.sim_model.set_visual_parameters(visual_params)
        
        return physics_params, visual_params
```

## 🚚 9.11 Deployment Considerations 🚚

### 📈 9.11.1 Real-time Performance Optimization 📈

```python
import time
import threading
from queue import Queue

class RealTimeRobotController:
    def __init__(self, perception_model, policy_model, control_frequency=50):
        self.perception_model = perception_model
        self.policy_model = policy_model
        self.control_frequency = control_frequency  # Hz
        self.control_period = 1.0 / control_frequency
        
        # Threading for parallel processing
        self.sensor_data_queue = Queue(maxsize=10)
        self.action_queue = Queue(maxsize=1)
        
        # Performance monitoring
        self.last_perception_time = 0
        self.last_policy_time = 0
        self.control_loop_time = 0
        
        # Threading control
        self.running = False
        self.perception_thread = None
        self.policy_thread = None
        
    def preprocess_sensor_data(self, raw_sensors):
        """Optimized preprocessing of sensor data"""
        # Convert sensor data to appropriate format
        processed_data = {}
        
        if 'camera' in raw_sensors:
            # Resize and normalize image data
            img = raw_sensors['camera']
            processed_data['camera'] = self.resize_and_normalize_image(img)
        
        if 'lidar' in raw_sensors:
            # Process LiDAR data efficiently
            lidar = raw_sensors['lidar']
            processed_data['lidar'] = self.process_lidar_data(lidar)
            
        if 'imu' in raw_sensors:
            # Process IMU data
            imu = raw_sensors['imu']
            processed_data['imu'] = self.process_imu_data(imu)
        
        return processed_data
    
    def resize_and_normalize_image(self, img):
        """Efficient image preprocessing"""
        # Resize using efficient method
        resized = cv2.resize(img, (224, 224))  # Example size
        # Normalize to [0, 1] and transpose to channel-first
        normalized = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return normalized
    
    def process_lidar_data(self, lidar):
        """Efficient LiDAR preprocessing"""
        # Convert to tensor and normalize if needed
        return torch.FloatTensor(lidar).unsqueeze(0)
    
    def process_imu_data(self, imu):
        """Efficient IMU preprocessing"""
        return torch.FloatTensor(imu).unsqueeze(0)
    
    def perception_worker(self):
        """Worker thread for perception processing"""
        while self.running:
            try:
                # Get raw sensor data
                raw_data = self.sensor_data_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process perception
                processed_data = self.preprocess_sensor_data(raw_data)
                perception_output = self.perception_model(processed_data)
                
                self.last_perception_time = time.time() - start_time
                
                # Add to next processing queue
                self.policy_queue.put(perception_output)
                
            except Exception as e:
                print(f"Perception worker error: {e}")
                
    def policy_worker(self):
        """Worker thread for policy decision"""
        while self.running:
            try:
                # Get perception output
                perception_data = self.policy_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Get action from policy
                action = self.policy_model(perception_data)
                
                self.last_policy_time = time.time() - start_time
                
                # Add to action queue
                if not self.action_queue.full():
                    self.action_queue.put(action)
                
            except Exception as e:
                print(f"Policy worker error: {e}")
    
    def run_control_loop(self):
        """Main real-time control loop"""
        self.running = True
        
        # Start worker threads
        self.perception_thread = threading.Thread(target=self.perception_worker)
        self.policy_thread = threading.Thread(target=self.policy_worker)
        
        self.perception_thread.start()
        self.policy_thread.start()
        
        try:
            while self.running:
                start_time = time.time()
                
                # Get latest action
                if not self.action_queue.empty():
                    action = self.action_queue.get()
                    
                    # Send action to robot
                    self.send_action_to_robot(action)
                
                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.control_period - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.control_loop_time = time.time() - start_time
                
        except KeyboardInterrupt:
            print("Control loop interrupted")
        finally:
            self.stop()
    
    def send_action_to_robot(self, action):
        """Send action to robot hardware"""
        # Implementation depends on your robot interface
        # This is a placeholder
        pass
    
    def stop(self):
        """Stop the controller"""
        self.running = False
        if self.perception_thread:
            self.perception_thread.join()
        if self.policy_thread:
            self.policy_thread.join()

# ⚙️ Additional optimization techniques ⚙️

def optimize_model_for_inference(model, input_shape, quantize=False):
    """Optimize a PyTorch model for inference"""
    model.eval()
    
    # Trace the model
    example_input = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, example_input)
    
    if quantize:
        # Quantize the model for faster inference (reduces precision but increases speed)
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        return quantized_model
    
    return traced_model

def model_pruning(model, pruning_ratio=0.2):
    """Prune a neural network model to reduce size and improve speed"""
    import torch.nn.utils.prune as prune
    
    # Define parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))
    
    # Apply pruning
    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
    
    # Remove pruning reparameterization so it's permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, "weight")
    
    return model
```

## 📝 9.12 Summary 📝

This chapter covered key aspects of robot learning and AI integration in physical AI systems. We explored:

1. **Perception Systems**: Deep learning approaches for vision, segmentation, and depth estimation
2. **Robot Control**: Supervised learning, PID controllers enhanced with ML, and sequential decision making
3. **Deep Learning Architectures**: RNNs, attention mechanisms, and ConvLSTMs for robotics
4. **Reinforcement Learning**: DQN, PPO, and SAC algorithms for robot control
5. **Sensor Fusion**: Kalman filters, particle filters, and neural fusion methods
6. **Learning from Demonstrations**: Behavior cloning, DAgger, and GAIL approaches
7. **Navigation & Path Planning**: Learning-enhanced A* and RRT* algorithms
8. **Human-Robot Interaction Learning**: Learning from human feedback and interactive learning
9. **Evaluation and Validation**: Metrics for robot learning and sim-to-real transfer
10. **Deployment Considerations**: Real-time performance optimization

The implementations in this chapter provide foundational tools for developing learning-enabled robots. These techniques enable robots to adapt to new situations, learn complex behaviors, and improve performance over time.

### ℹ️ Key Takeaways: ℹ️
- Robot learning requires special consideration for safety, sample efficiency, and real-time performance
- Deep learning architectures can be adapted for robotics-specific tasks like perception and control
- Reinforcement learning offers powerful methods for learning complex behaviors through environment interaction
- Proper evaluation and validation are crucial for deploying learning systems on physical robots
- Simulation-to-real transfer remains challenging but can be improved with domain randomization and careful validation

## 🤔 Knowledge Check 🤔

1. Explain the differences between behavior cloning, DAgger, and GAIL in imitation learning.
2. Compare the advantages and disadvantages of DQN vs. PPO vs. SAC for robotic control.
3. Describe how sensor fusion can improve robot perception and decision-making.
4. What are the key challenges in deploying machine learning models on physical robots?
5. How can domain randomization improve sim-to-real transfer?

---
*Continue to [Chapter 10: Vision-Language-Action Integration](./chapter-10-vision-language-action.md)*