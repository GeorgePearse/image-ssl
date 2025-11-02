# Extensions: Jump and Scout Actions

*Extended action spaces for long-range image exploration.*

## Overview

The base RL navigation system uses **8-connected movement** (up, down, left, right, and diagonals) with 1-pixel steps. The extensions module adds:

- **Jump Actions**: Long-range movement (5-10 pixels)
- **Scout Actions**: Peek at distant regions without committing

These are **optional extensions** that can be enabled when needed.

## Quick Start

```python
from techniques.rl_navigation import RLTrainer
from techniques.rl_navigation.extensions import (
    ExtendedActionSpace,
    HierarchicalPolicy,
)

# Create extended action space
action_space = ExtendedActionSpace(
    jump_distance=7,      # How far to jump
    use_jumps=True,       # Enable jump actions
    use_scouts=False,     # Disable scout actions (optional)
)

# Use hierarchical policy for extended actions
# (More efficient than flat policy with 16-24 actions)
policy = HierarchicalPolicy(
    feature_dim=384,
    action_space=action_space,
    hidden_dim=256,
)

# Note: Full training integration coming soon
# Current: Manual construction for experiments
```

## Extended Action Space

### Action Types

**Base Actions** (0-7): 8-connected movement
- 0: RIGHT, 1: DOWN-RIGHT, 2: DOWN, etc.
- 1 pixel step

**Jump Actions** (8-15): Long-range movement
- Same 8 directions
- Configurable distance (default: 7 pixels)
- Useful for crossing uniform regions quickly

**Scout Actions** (16-23): Non-committing peeks
- Look at distant region
- Return to current position
- Useful for planning

### Configuration

```python
action_space = ExtendedActionSpace(
    jump_distance=7,        # Pixels to jump
    use_jumps=True,         # Enable jumps
    use_scouts=False,       # Enable scouts (optional)
)

print(f"Total actions: {action_space.action_dim}")
# Output: 16 (base + jumps) or 24 (base + jumps + scouts)
```

## Hierarchical Policy

For extended action spaces (16-24 actions), a hierarchical policy is more efficient than a flat policy.

### Two-Level Decision Making

1. **Meta-Policy**: Choose action type (base, jump, or scout)
2. **Direction-Policy**: Choose direction (8 directions)

This factorization reduces the action space from 24 discrete actions to:
- 3 action types × 8 directions = 11 logits total

### Architecture

```python
class HierarchicalPolicy:
    def forward(self, features):
        shared = self.shared(features)

        # Meta-policy: action type
        type_logits = self.meta_policy(shared)    # (batch, 3)

        # Direction policy: direction
        dir_logits = self.direction_policy(shared)  # (batch, 8)

        # Value estimate
        value = self.critic(shared)

        return type_logits, dir_logits, value
```

### Advantages

✅ Fewer parameters to learn
✅ Better sample efficiency
✅ Structured exploration (try different types)
✅ Interpretable (which type is preferred?)

## Jump Actions

### Use Cases

**Good for**:
- Images with large uniform regions (sky, grass)
- Quickly reaching distant semantic regions
- Exploratory phase (early training)

**Not ideal for**:
- Dense semantic images (every pixel matters)
- Fine-grained exploration
- When 1-pixel precision needed

### Example

```python
# Get next position with jump action
action_id = 9  # DOWN_RIGHT jump
next_pos, action_type = action_space.get_next_position(
    current_pos=(100, 100),
    action_id=action_id,
    image_shape=(224, 224)
)

print(next_pos)       # (107, 107) - jumped 7 pixels
print(action_type)    # "jump"
```

## Scout Actions

### Use Cases

**Good for**:
- Planning ahead (look before you leap)
- Multi-step reasoning
- Avoiding dead ends

**Not ideal for**:
- Simple reactive policies
- When immediate rewards sufficient
- Computational budget limited

### Reward Modification

Scout actions receive modified rewards:

```python
from techniques.rl_navigation.extensions import ScoutingRewardModifier

modifier = ScoutingRewardModifier(
    scout_reward_scale=0.5,   # Scouts get 50% of normal reward
    scout_penalty=0.01,        # Small penalty to prevent abuse
)

# In environment step
if action_type == "scout":
    reward = modifier.modify_reward(
        reward=base_reward,
        action_type="scout",
        is_scout=True
    )
```

This prevents agents from just scouting forever without committing.

## Integration with Training

### Current Status

⚠️ Extensions are implemented but not yet integrated into the main training loop.

**What works**:
- ✅ ExtendedActionSpace class
- ✅ HierarchicalPolicy architecture
- ✅ ScoutingRewardModifier
- ✅ Standalone tests

**What needs integration**:
- ❌ Modify RLTrainer to support extended actions
- ❌ Update environment to handle jump/scout logic
- ❌ Add configuration presets for extensions

### Manual Usage

You can experiment with extensions by modifying the training script:

```python
# experiments/train_rl_navigator.py

from techniques.rl_navigation.extensions import (
    ExtendedActionSpace,
    HierarchicalPolicy,
)

# Create extended action space
action_space = ExtendedActionSpace(jump_distance=7, use_jumps=True)

# Replace standard policy with hierarchical
policy = HierarchicalPolicy(
    feature_dim=encoder.feature_dim,
    action_space=action_space,
    hidden_dim=256,
).to(device)

# Update environment to use extended action space
# (requires environment modifications - see TODO)
```

## Performance Considerations

### Memory

Extended action spaces don't significantly increase memory:
- Same rollout buffer size
- Policy has similar parameter count (hierarchical)

### Computation

Jumps and scouts affect training time:
- **Jump actions**: Slightly faster (fewer total steps per episode)
- **Scout actions**: Slower (must execute and revert)

### Sample Efficiency

Theory:
- Jumps may improve exploration in sparse images
- Scouts may improve planning but cost samples

Empirical validation needed (see [TODO](../../development/todo.md)).

## Testing Extensions

Run the standalone test:

```bash
# Note: Requires PyTorch installed
python techniques/rl_navigation/extensions.py
```

Expected output:
```
Testing Extended Action Space...
Total actions: 16
Base: 0-7, Jump: 8-15
Base action 0: (100, 100) -> (100, 101)
Jump action 9: (100, 100) -> (107, 107)
Extended action space test passed!

Testing Hierarchical Policy...
Hierarchical policy test passed!

All tests passed!
```

## Future Work

See [TODO List](../../development/todo.md) for:
- Full training integration
- Configuration presets (quick_test_jumps, etc.)
- Experimental validation (do jumps help?)
- Visualization of jump/scout actions

## References

Related work on hierarchical RL:
- Options Framework: Sutton et al. (1999)
- Hierarchical DQN: Kulkarni et al. (2016)
- HAM: Parr & Russell (1998)

## Next Steps

- See [Architecture](architecture.md) for system overview
- Check [Training Guide](training.md) for standard training
- Review code: `techniques/rl_navigation/extensions.py`
