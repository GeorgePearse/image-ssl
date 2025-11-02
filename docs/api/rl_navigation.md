# API Reference: RL Navigation

*Detailed API documentation for the RL navigation module.*

## Core Components

### SemanticEncoder

**File**: `techniques/rl_navigation/encoder.py`

DINOv2-based semantic feature extractor.

```python
from techniques.rl_navigation import SemanticEncoder

encoder = SemanticEncoder(
    model_name="dinov2_vits14",  # Model variant
    patch_size=14,                # Patch size (DINOv2 default)
    freeze=True,                  # Freeze parameters initially
    device="cuda"                 # Device
)
```

#### Methods

##### `freeze()`
Freeze all encoder parameters (Phase 1 training).

```python
encoder.freeze()
```

##### `unfreeze_top_layers(n_layers=2)`
Unfreeze top transformer layers (Phase 2 training).

**Parameters**:
- `n_layers` (int): Number of top layers to unfreeze

```python
encoder.unfreeze_top_layers(n_layers=2)
```

##### `forward(image)`
Extract patch features from image.

**Parameters**:
- `image` (torch.Tensor): RGB image (B, 3, H, W)

**Returns**:
- `features` (torch.Tensor): Patch features (B, num_patches_h, num_patches_w, feature_dim)

##### `get_patch_features_at_position(image, position, patch_radius=1)`
Extract local patch features around a pixel position.

**Parameters**:
- `image` (torch.Tensor): RGB image
- `position` (tuple): (row, col) pixel coordinates
- `patch_radius` (int): Radius of patch neighborhood

**Returns**:
- `features` (torch.Tensor): Averaged local features (feature_dim,)

---

### NavigationPolicy

**File**: `techniques/rl_navigation/policy.py`

PPO-based actor-critic policy.

```python
from techniques.rl_navigation import NavigationPolicy

policy = NavigationPolicy(
    feature_dim=384,    # DINOv2 vits14 dimension
    action_dim=8,       # 8-connected movement
    hidden_dim=256      # Hidden layer size
)
```

#### Methods

##### `forward(features)`
Forward pass through policy.

**Parameters**:
- `features` (torch.Tensor): Semantic features (batch_size, feature_dim)

**Returns**:
- `action_logits` (torch.Tensor): Action distribution logits (batch_size, action_dim)
- `value` (torch.Tensor): Value estimate (batch_size, 1)

##### `act(features, deterministic=False)`
Sample action from policy.

**Parameters**:
- `features` (torch.Tensor): Semantic features
- `deterministic` (bool): If True, take argmax; else sample

**Returns**:
- `action` (torch.Tensor): Sampled action index
- `log_prob` (torch.Tensor): Log probability of action
- `value` (torch.Tensor): Value estimate

```python
action, log_prob, value = policy.act(features, deterministic=True)
```

##### `evaluate_actions(features, actions)`
Evaluate actions for PPO update.

**Parameters**:
- `features` (torch.Tensor): Semantic features (batch_size, feature_dim)
- `actions` (torch.Tensor): Actions taken (batch_size,)

**Returns**:
- `log_probs` (torch.Tensor): Log probabilities
- `values` (torch.Tensor): Value estimates
- `entropy` (torch.Tensor): Action distribution entropy

---

### ForwardDynamicsModel

**File**: `techniques/rl_navigation/forward_dynamics.py`

Predicts next semantic features given current features and action.

```python
from techniques.rl_navigation import ForwardDynamicsModel

predictor = ForwardDynamicsModel(
    feature_dim=384,
    action_dim=8,
    hidden_dim=512
)
```

#### Methods

##### `forward(features_t, action)`
Predict next features.

**Parameters**:
- `features_t` (torch.Tensor): Current features (batch_size, feature_dim)
- `action` (torch.Tensor): Action indices (batch_size,)

**Returns**:
- `predicted_features` (torch.Tensor): Predicted next features (batch_size, feature_dim)

##### `compute_intrinsic_reward(features_t, action, features_t1)`
Compute prediction error as intrinsic reward.

**Parameters**:
- `features_t` (torch.Tensor): Current features
- `action` (torch.Tensor): Actions taken
- `features_t1` (torch.Tensor): Actual next features

**Returns**:
- `reward` (torch.Tensor): Prediction error (batch_size,)

```python
reward = predictor.compute_intrinsic_reward(feat_t, action, feat_t1)
```

---

### RNDIntrinsicMotivation

**File**: `techniques/rl_navigation/forward_dynamics.py`

Random Network Distillation for intrinsic motivation.

```python
from techniques.rl_navigation import RNDIntrinsicMotivation

rnd = RNDIntrinsicMotivation(
    feature_dim=384,
    hidden_dim=512
)
```

#### Methods

##### `compute_intrinsic_reward(features)`
Compute RND prediction error.

**Parameters**:
- `features` (torch.Tensor): State features

**Returns**:
- `reward` (torch.Tensor): RND prediction error

---

### ImageNavigationEnv

**File**: `techniques/rl_navigation/environment.py`

MDP environment for image navigation.

```python
from techniques.rl_navigation import ImageNavigationEnv

env = ImageNavigationEnv(
    image=rgb_image,              # numpy array (H, W, 3)
    encoder=encoder,              # SemanticEncoder
    predictor=predictor,          # ForwardDynamicsModel or RND
    max_steps=500,                # Episode length
    reward_horizon=10,            # Lookahead steps
    reward_lambda=0.1,            # Exponential decay
    coverage_bonus_weight=0.1,    # Coverage bonus
    device="cuda"
)
```

#### Methods

##### `reset()`
Reset environment to random starting position.

**Returns**:
- `state` (dict): Initial state with keys:
  - `"position"`: (row, col) tuple
  - `"features"`: Semantic features tensor
  - `"visited"`: Visited mask

```python
state = env.reset()
position = state["position"]
features = state["features"]
```

##### `step(action)`
Execute action and return next state.

**Parameters**:
- `action` (int): Action index (0-7 for base actions)

**Returns**:
- `state` (dict): Next state
- `reward` (float): Reward for transition
- `done` (bool): Episode termination flag
- `info` (dict): Additional information

```python
next_state, reward, done, info = env.step(action)
```

##### `get_statistics()`
Get episode statistics.

**Returns**:
- `stats` (dict): Statistics with keys:
  - `"coverage"`: Fraction of image visited
  - `"path_length"`: Number of steps taken
  - `"unique_positions"`: Number of unique positions

---

### RLTrainer

**File**: `techniques/rl_navigation/trainer.py`

Two-phase trainer orchestrating the full training loop.

```python
from techniques.rl_navigation import RLTrainer

trainer = RLTrainer(
    image=rgb_image,
    encoder_name="dinov2_vits14",
    device="cuda",
    phase1_episodes=10000,
    phase2_episodes=5000,
    policy_lr=3e-4,
    predictor_lr=1e-3,
    encoder_lr=1e-5,
    # ... many more parameters (see config.py)
)
```

#### Methods

##### `train()`
Run full two-phase training.

```python
trainer.train()
```

##### `save_checkpoint(filename)`
Save training checkpoint.

**Parameters**:
- `filename` (str): Checkpoint filename

```python
trainer.save_checkpoint("checkpoint.pt")
```

##### `load_checkpoint(filename)`
Load training checkpoint.

**Parameters**:
- `filename` (str): Checkpoint filename

```python
trainer.load_checkpoint("checkpoint.pt")
```

---

## Configuration

### RLConfig

**File**: `techniques/rl_navigation/config.py`

Configuration dataclass for hyperparameters.

```python
from techniques.rl_navigation.config import RLConfig, get_config

# Get preset config
config = get_config("default")

# Or create custom
config = RLConfig(
    encoder_name="dinov2_vits14",
    phase1_episodes=10000,
    phase2_episodes=5000,
    policy_lr=3e-4,
    # ... etc
)
```

#### Available Presets

```python
config = get_config("default")      # Standard training
config = get_config("quick_test")   # Fast testing
config = get_config("rnd")          # Use RND
config = get_config("long")         # Extended training
```

---

## Extensions

### ExtendedActionSpace

**File**: `techniques/rl_navigation/extensions.py`

Extended action space with jump/scout actions.

```python
from techniques.rl_navigation.extensions import ExtendedActionSpace

action_space = ExtendedActionSpace(
    jump_distance=7,
    use_jumps=True,
    use_scouts=False
)

next_pos, action_type = action_space.get_next_position(
    current_pos=(100, 100),
    action_id=9,  # Jump action
    image_shape=(224, 224)
)
```

### HierarchicalPolicy

**File**: `techniques/rl_navigation/extensions.py`

Two-level policy for extended action spaces.

```python
from techniques.rl_navigation.extensions import HierarchicalPolicy

policy = HierarchicalPolicy(
    feature_dim=384,
    action_space=action_space,
    hidden_dim=256
)
```

---

## Constants

### Action Space

```python
# Base 8-connected actions
ACTIONS = {
    0: (0, 1),    # RIGHT
    1: (1, 1),    # DOWN_RIGHT
    2: (1, 0),    # DOWN
    3: (1, -1),   # DOWN_LEFT
    4: (0, -1),   # LEFT
    5: (-1, -1),  # UP_LEFT
    6: (-1, 0),   # UP
    7: (-1, 1),   # UP_RIGHT
}
```

---

## Type Hints

The codebase uses type hints. Example:

```python
from typing import Tuple, Dict, Optional
import torch

def get_features(
    position: Tuple[int, int],
    encoder: SemanticEncoder,
    image: torch.Tensor
) -> torch.Tensor:
    ...
```

---

## Next Steps

- See [Architecture](../techniques/rl_navigation/architecture.md) for system design
- Check [Training Guide](../techniques/rl_navigation/training.md) for usage
- Review [Extensions](../techniques/rl_navigation/extensions.md) for jump/scout actions
