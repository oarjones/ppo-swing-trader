"""
Multi-Timeframe Neural Network Architecture for ElegantRL

This module implements the unified multi-timeframe architecture with regime detection
as custom neural network models for use with ElegantRL's PPO implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Try to import ElegantRL components
try:
    
    
    from external.elegantrl.agents.AgentPPO import ActorPPO, CriticPPO
    ELEGANTRL_AVAILABLE = True
except ImportError:
    # Define basic placeholder classes if ElegantRL is not available
    class ActorPPO(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class CriticPPO(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    
                
    class AgentPPO:
        def __init__(self, *args, **kwargs):
            pass
    
    ELEGANTRL_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class TimeframeEncoder(nn.Module):
    """
    Encoder module for processing a single timeframe's features.
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int = 128, 
                rnn_layers: int = 1,
                use_attention: bool = True,
                dropout: float = 0.1):
        """
        Initialize the timeframe encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            rnn_layers: Number of RNN layers
            use_attention: Whether to use attention mechanism
            dropout: Dropout rate
        """
        super(TimeframeEncoder, self).__init__()
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU for sequential data
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence.
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]
            
        Returns:
            Encoded representation [batch_size, hidden_dim]
        """
        batch_size, seq_len, features = x.size()
        
        # Extract features for each time step
        x_reshaped = x.reshape(-1, features)
        x_features = self.feature_net(x_reshaped)
        x_features = x_features.reshape(batch_size, seq_len, -1)
        
        # Process with RNN
        rnn_out, hidden = self.rnn(x_features)
        
        if self.use_attention:
            # Apply attention mechanism
            attn_weights = self.attention(rnn_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Weighted sum
            context = torch.sum(rnn_out * attn_weights, dim=1)
            return context
        else:
            # Use last hidden state
            return hidden[-1]


class RegimeDetector(nn.Module):
    """
    Module for detecting market regime.
    Used as an auxiliary task for better learning.
    """
    
    def __init__(self, input_dim: int, num_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            input_dim: Input dimension
            num_regimes: Number of market regimes to detect
        """
        super(RegimeDetector, self).__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_regimes)
        )
        
        self.num_regimes = num_regimes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect market regime.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits for each regime
        """
        return self.detector(x)


class MultiTimeframeActorPPO(ActorPPO):
    """
    Multi-timeframe actor network for PPO.
    Extends ElegantRL's ActorPPO.
    """
    
    def __init__(self,
                state_dim: int,
                action_dim: int,
                features_1h: int,
                seq_len_1h: int,
                features_4h: int,
                seq_len_4h: int,
                hidden_dim: int = 128,
                use_attention: bool = True,
                regime_detection: bool = True,
                num_regimes: int = 3):
        """
        Initialize the multi-timeframe actor.
        
        Args:
            state_dim: Total state dimension
            action_dim: Action dimension
            features_1h: Number of features per step in 1h data
            seq_len_1h: Sequence length for 1h data
            features_4h: Number of features per step in 4h data
            seq_len_4h: Sequence length for 4h data
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention
            regime_detection: Whether to include regime detection
            num_regimes: Number of market regimes to detect
        """
        # Initialize parent class but we'll override its network
        super(MultiTimeframeActorPPO, self).__init__(state_dim=state_dim, action_dim=action_dim)
        
        # Store dimensions for reshaping
        self.features_1h = features_1h
        self.seq_len_1h = seq_len_1h
        self.features_4h = features_4h
        self.seq_len_4h = seq_len_4h
        self.portfolio_features = 3  # balance, position, position_price
        
        # Encoders for each timeframe
        self.encoder_1h = TimeframeEncoder(
            input_dim=features_1h,
            hidden_dim=hidden_dim,
            use_attention=use_attention
        )
        
        self.encoder_4h = TimeframeEncoder(
            input_dim=features_4h,
            hidden_dim=hidden_dim,
            use_attention=use_attention
        )
        
        # Regime detection
        self.regime_detection = regime_detection
        if regime_detection:
            self.regime_detector = RegimeDetector(
                input_dim=hidden_dim * 2,  # Combining both timeframes
                num_regimes=num_regimes
            )
        
        # Fusion layer
        fusion_input_dim = hidden_dim * 2 + self.portfolio_features  # 1h + 4h + portfolio
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor) - override the default
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        # Standard deviation
        # Override from parent class
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for actor.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_mean, action_std, regime_logits)
        """
        # Reshape flat state into timeframe components
        x_1h, x_4h, x_portfolio = self._reshape_input(state)
        
        # Encode each timeframe
        encoded_1h = self.encoder_1h(x_1h)
        encoded_4h = self.encoder_4h(x_4h)
        
        # Detect regime if enabled
        regime_logits = None
        if self.regime_detection:
            combined_encodings = torch.cat([encoded_1h, encoded_4h], dim=1)
            regime_logits = self.regime_detector(combined_encodings)
        
        # Fusion layer
        fusion_input = torch.cat([encoded_1h, encoded_4h, x_portfolio], dim=1)
        fused = self.fusion_layer(fusion_input)
        
        # Policy head
        action_mean = self.net(fused)
        action_std = self.action_std_log.exp()
        
        return action_mean, action_std, regime_logits
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from state (with exploration).
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob)
        """
        a_mean, a_std, _ = self.forward(state)
        dist = torch.distributions.Normal(a_mean, a_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        return action, log_prob
    
    def get_action_logprob(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probability of an action given a state.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (new_action, log_prob)
        """
        a_mean, a_std, _ = self.forward(state)
        dist = torch.distributions.Normal(a_mean, a_std)
        log_prob = dist.log_prob(action).sum(1)
        return action, log_prob
    
    def get_action_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get entropy of the action distribution for a state.
        
        Args:
            state: State tensor
            
        Returns:
            Entropy tensor
        """
        a_mean, a_std, _ = self.forward(state)
        dist = torch.distributions.Normal(a_mean, a_std)
        return dist.entropy().sum(1)
    
    def _reshape_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape flat input state into timeframe components.
        
        Args:
            x: Flat state tensor
            
        Returns:
            Tuple of (1h_data, 4h_data, portfolio_data)
        """
        batch_size = x.size(0)
        
        # Calculate sizes
        size_1h = self.features_1h * self.seq_len_1h
        size_4h = self.features_4h * self.seq_len_4h
        
        # Split state
        x_1h_flat = x[:, :size_1h]
        x_4h_flat = x[:, size_1h:size_1h + size_4h]
        x_portfolio = x[:, size_1h + size_4h:]
        
        # Reshape for sequences
        x_1h = x_1h_flat.reshape(batch_size, self.seq_len_1h, self.features_1h)
        x_4h = x_4h_flat.reshape(batch_size, self.seq_len_4h, self.features_4h)
        
        return x_1h, x_4h, x_portfolio


class MultiTimeframeCriticPPO(CriticPPO):
    """
    Multi-timeframe critic network for PPO.
    Extends ElegantRL's CriticPPO.
    """
    
    def __init__(self,
                state_dim: int,
                features_1h: int,
                seq_len_1h: int,
                features_4h: int,
                seq_len_4h: int,
                hidden_dim: int = 128,
                use_attention: bool = True):
        """
        Initialize the multi-timeframe critic.
        
        Args:
            state_dim: Total state dimension
            features_1h: Number of features per step in 1h data
            seq_len_1h: Sequence length for 1h data
            features_4h: Number of features per step in 4h data
            seq_len_4h: Sequence length for 4h data
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention
        """
        # Initialize parent class but we'll override its network
        super(MultiTimeframeCriticPPO, self).__init__(state_dim=state_dim)
        
        # Store dimensions for reshaping
        self.features_1h = features_1h
        self.seq_len_1h = seq_len_1h
        self.features_4h = features_4h
        self.seq_len_4h = seq_len_4h
        self.portfolio_features = 3  # balance, position, position_price
        
        # Encoders for each timeframe
        self.encoder_1h = TimeframeEncoder(
            input_dim=features_1h,
            hidden_dim=hidden_dim,
            use_attention=use_attention
        )
        
        self.encoder_4h = TimeframeEncoder(
            input_dim=features_4h,
            hidden_dim=hidden_dim,
            use_attention=use_attention
        )
        
        # Fusion layer
        fusion_input_dim = hidden_dim * 2 + self.portfolio_features  # 1h + 4h + portfolio
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value head (critic) - override the default
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for critic.
        
        Args:
            state: State tensor
            
        Returns:
            Value estimate
        """
        # Reshape flat state into timeframe components
        x_1h, x_4h, x_portfolio = self._reshape_input(state)
        
        # Encode each timeframe
        encoded_1h = self.encoder_1h(x_1h)
        encoded_4h = self.encoder_4h(x_4h)
        
        # Fusion layer
        fusion_input = torch.cat([encoded_1h, encoded_4h, x_portfolio], dim=1)
        fused = self.fusion_layer(fusion_input)
        
        # Value head
        value = self.net(fused)
        return value
    
    def _reshape_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape flat input state into timeframe components.
        
        Args:
            x: Flat state tensor
            
        Returns:
            Tuple of (1h_data, 4h_data, portfolio_data)
        """
        batch_size = x.size(0)
        
        # Calculate sizes
        size_1h = self.features_1h * self.seq_len_1h
        size_4h = self.features_4h * self.seq_len_4h
        
        # Split state
        x_1h_flat = x[:, :size_1h]
        x_4h_flat = x[:, size_1h:size_1h + size_4h]
        x_portfolio = x[:, size_1h + size_4h:]
        
        # Reshape for sequences
        x_1h = x_1h_flat.reshape(batch_size, self.seq_len_1h, self.features_1h)
        x_4h = x_4h_flat.reshape(batch_size, self.seq_len_4h, self.features_4h)
        
        return x_1h, x_4h, x_portfolio


class MultiTimeframePPOPolicy():
    """
    Multi-timeframe policy for PPO.
    Extends ElegantRL's PPOPolicy.
    """
    
    def __init__(self,
                state_dim: int,
                action_dim: int,
                features_1h: int,
                seq_len_1h: int,
                features_4h: int,
                seq_len_4h: int,
                hidden_dim: int = 128,
                use_attention: bool = True,
                regime_detection: bool = True,
                num_regimes: int = 3,
                learning_rate: float = 3e-4,
                entropy_coef: float = 0.01,
                clip_ratio: float = 0.2,
                **kwargs):
        """
        Initialize the multi-timeframe PPO policy.
        
        Args:
            state_dim: Total state dimension
            action_dim: Action dimension
            features_1h: Number of features per step in 1h data
            seq_len_1h: Sequence length for 1h data
            features_4h: Number of features per step in 4h data
            seq_len_4h: Sequence length for 4h data
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention
            regime_detection: Whether to include regime detection
            num_regimes: Number of market regimes to detect
            learning_rate: Learning rate
            entropy_coef: Entropy coefficient
            clip_ratio: PPO clip ratio
            **kwargs: Additional arguments for the parent class
        """
        if not ELEGANTRL_AVAILABLE:
            logger.warning("ElegantRL not available. Using placeholder implementation.")
            return
            
        # Initialize parent class
        super(MultiTimeframePPOPolicy, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
        
        # Override with our custom networks
        self.act = MultiTimeframeActorPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            features_1h=features_1h,
            seq_len_1h=seq_len_1h,
            features_4h=features_4h,
            seq_len_4h=seq_len_4h,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
            regime_detection=regime_detection,
            num_regimes=num_regimes
        )
        
        self.cri = MultiTimeframeCriticPPO(
            state_dim=state_dim,
            features_1h=features_1h,
            seq_len_1h=seq_len_1h,
            features_4h=features_4h,
            seq_len_4h=seq_len_4h,
            hidden_dim=hidden_dim,
            use_attention=use_attention
        )
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.features_1h = features_1h
        self.seq_len_1h = seq_len_1h
        self.features_4h = features_4h
        self.seq_len_4h = seq_len_4h
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.regime_detection = regime_detection
        self.num_regimes = num_regimes
        
        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        
        # Additional losses
        self.regime_loss_coef = 0.1  # Weight for regime detection loss
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            params=list(self.act.parameters()) + list(self.cri.parameters()),
            lr=learning_rate
        )
        
        # Share some memory for efficiency (if networks have similar structure)
        if use_attention:
            # Share feature networks if identical
            self.cri.encoder_1h.feature_net = self.act.encoder_1h.feature_net
            self.cri.encoder_4h.feature_net = self.act.encoder_4h.feature_net
    
    def update_policy(self, buffer, batch_size, repeat_times, h_term=False):
        """
        Update the policy with PPO algorithm.
        Overrides ElegantRL's update_policy to include regime detection.
        
        Args:
            buffer: Experience buffer
            batch_size: Batch size
            repeat_times: Number of training iterations
            h_term: Whether to use H-term (ElegantRL's stability enhancement)
            
        Returns:
            Tuple of (actor_loss, critic_loss, entropy_loss)
        """
        if not ELEGANTRL_AVAILABLE:
            logger.warning("ElegantRL not available. Cannot update policy.")
            return 0, 0, 0
        
        # Get all experience from buffer
        buf_state, buf_action, buf_reward, buf_mask, buf_noise = buffer.get_all_data()
        
        # Calculate advantages and returns
        buf_value = self.cri(buf_state)
        
        # Calculate GAE
        buf_advantage, buf_return = self.compute_gae(
            buf_value, buf_reward, buf_mask, 
            self.gamma, self.lambda_gae
        )
        
        # Normalize advantages for stability
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        
        # Update with mini-batches
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        regime_losses = []
        
        for _ in range(repeat_times):
            indices = torch.randperm(buf_state.size(0))
            for start in range(0, buf_state.size(0), batch_size):
                end = min(start + batch_size, buf_state.size(0))
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_state = buf_state[batch_indices]
                batch_action = buf_action[batch_indices]
                batch_advantage = buf_advantage[batch_indices].unsqueeze(1)
                batch_return = buf_return[batch_indices]
                batch_noise = buf_noise[batch_indices]
                
                # Get old log probability
                old_logprob = self.act.get_old_logprob(batch_action, batch_noise)
                
                # Calculate new log probability and entropy
                action_mean, action_std, regime_logits = self.act.forward(batch_state)
                new_logprob = self.act.get_logprob(batch_action, action_mean, action_std)
                entropy = self.act.get_entropy(action_mean, action_std)
                
                # Calculate ratio for PPO
                ratio = (new_logprob - old_logprob).exp()
                
                # PPO objectives
                obj_surrogate1 = ratio * batch_advantage
                obj_surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantage
                
                # Actor loss
                actor_loss = -torch.min(obj_surrogate1, obj_surrogate2).mean()
                
                # Critic loss
                batch_value = self.cri(batch_state)
                if h_term:
                    # Use H-term for better stability
                    h_value = self.h_term(batch_state, batch_action, batch_advantage, batch_return)
                    critic_loss = torch.mean(torch.pow(h_value - batch_return, 2))
                else:
                    # Standard MSE
                    critic_loss = torch.mean(torch.pow(batch_value - batch_return, 2))
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Regime detection loss (if enabled)
                regime_loss = torch.zeros_like(actor_loss)
                if self.regime_detection and regime_logits is not None:
                    # For supervised regime detection, we would need actual regime labels
                    # For unsupervised detection, we can use a consistency or diversity loss
                    # Here we use a simple diversity loss to encourage using all regimes
                    regime_probs = F.softmax(regime_logits, dim=1)
                    avg_probs = regime_probs.mean(dim=0)
                    # Encourage uniform distribution of regimes
                    uniform_target = torch.ones_like(avg_probs) / self.num_regimes
                    regime_loss = F.kl_div(avg_probs.log(), uniform_target, reduction='batchmean')
                
                # Total loss
                total_loss = actor_loss + critic_loss + self.entropy_coef * entropy_loss
                if self.regime_detection:
                    total_loss += self.regime_loss_coef * regime_loss
                
                # Update networks
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                if self.regime_detection:
                    regime_losses.append(regime_loss.item())
        
        return (
            np.mean(actor_losses), 
            np.mean(critic_losses), 
            np.mean(entropy_losses)
        )
    
    def h_term(self, states, action, advantage, returns):
        """
        ElegantRL's H-term for better stability.
        
        Args:
            states: Batch of states
            action: Batch of actions
            advantage: Batch of advantages
            returns: Batch of returns
            
        Returns:
            H-value term
        """
        actor_mean, actor_std, _ = self.act.forward(states)
        value = self.cri(states)
        
        # Calculate new log probability and entropy
        logprob = self.act.get_logprob(action, actor_mean, actor_std)
        ratio = logprob.exp()
        
        # H-term calculation
        h_value = value + advantage / torch.clamp(ratio, 0.05, 2.0)
        return h_value
    
    @staticmethod
    def compute_gae(values, rewards, masks, gamma, lambda_gae):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            values: Value estimates
            rewards: Rewards
            masks: Episode masks (1 - done)
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Add terminal value
        values_extended = torch.cat([values, torch.zeros_like(values[:1])], dim=0)
        
        for t in reversed(range(rewards.size(0))):
            delta = rewards[t] + gamma * values_extended[t + 1] * masks[t] - values[t]
            advantages[t] = last_advantage = delta + gamma * lambda_gae * masks[t] * last_advantage
        
        returns = advantages + values
        return advantages, returns


class MultiTimeframePPOAgent(AgentPPO):
    """
    Multi-timeframe PPO agent.
    Extends ElegantRL's AgentPPO.
    """
    
    def __init__(self,
                env,
                features_1h: int,
                seq_len_1h: int,
                features_4h: int,
                seq_len_4h: int,
                hidden_dim: int = 128,
                use_attention: bool = True,
                regime_detection: bool = True,
                num_regimes: int = 3,
                learning_rate: float = 3e-4,
                *args, **kwargs):
        """
        Initialize the multi-timeframe PPO agent.
        
        Args:
            env: Environment
            features_1h: Number of features per step in 1h data
            seq_len_1h: Sequence length for 1h data
            features_4h: Number of features per step in 4h data
            seq_len_4h: Sequence length for 4h data
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention
            regime_detection: Whether to include regime detection
            num_regimes: Number of market regimes to detect
            learning_rate: Learning rate
            *args, **kwargs: Additional arguments for the parent class
        """
        if not ELEGANTRL_AVAILABLE:
            logger.warning("ElegantRL not available. Using placeholder implementation.")
            return
            
        # Store parameters
        self.features_1h = features_1h
        self.seq_len_1h = seq_len_1h
        self.features_4h = features_4h
        self.seq_len_4h = seq_len_4h
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.regime_detection = regime_detection
        self.num_regimes = num_regimes
        
        # Initialize environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Initialize parent class
        super(MultiTimeframePPOAgent, self).__init__(
            net_dim=hidden_dim,  # Not really used as we override it
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            *args, **kwargs
        )
        
        # Use our custom policy
        self.cri = self.act = MultiTimeframePPOPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            features_1h=features_1h,
            seq_len_1h=seq_len_1h,
            features_4h=features_4h,
            seq_len_4h=seq_len_4h,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
            regime_detection=regime_detection,
            num_regimes=num_regimes,
            learning_rate=learning_rate
        )