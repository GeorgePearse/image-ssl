# Architecture Deep Dive

*This page provides detailed architectural information for the RL navigation system.*

## System Overview

The Visual Next Token system consists of several key components working together to enable curiosity-driven image navigation:

```
┌─────────────────────────────────────────────────────┐
│                    Image (RGB)                       │
│                  (H × W × 3)                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Semantic Encoder    │
          │     (DINOv2)         │
          │  Frozen → Fine-tuned │
          └──────────┬───────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           │    Features (z)   │ ◄── Detach for policy!
           │                   │
           └─────────┬─────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
  ┌─────────────┐     ┌──────────────────┐
  │   Policy    │     │ Forward Dynamics │
  │    (π)      │     │      (P)         │
  │  Actor-     │     │  Predict next    │
  │  Critic     │     │  features from   │
  │             │     │  current + action│
  └──────┬──────┘     └────────┬─────────┘
         │                     │
         │ Action              │ Predicted z'
         ▼                     ▼
  ┌────────────────────────────────────┐
  │   Environment Step                 │
  │   - Execute action (move)          │
  │   - Get actual next features (z')  │
  │   - Compute rolling-window         │
  │     accuracy reward                │
  └────────┬───────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ PPO Update  │
    │ - Policy    │
    │ - Value fn  │
    │ - Predictor │
    │ - Encoder*  │ * Phase 2 only
    └─────────────┘
```

## Component Details

### 1. Semantic Encoder

**File**: `techniques/rl_navigation/encoder.py`

The semantic encoder maps raw pixels to semantic feature vectors that are invariant to appearance variations.

**Key Methods**:
- `freeze()`: Freeze all parameters (Phase 1)
- `unfreeze_top_layers(n_layers=2)`: Unfreeze top transformer blocks (Phase 2)
- `get_patch_features_at_position()`: Extract local patch features

**Feature Dimensions**:
- `dinov2_vits14`: 384 dims
- `dinov2_vitb14`: 768 dims
- `dinov2_vitl14`: 1024 dims

### 2. Navigation Policy

**File**: `techniques/rl_navigation/policy.py`

PPO-based actor-critic architecture for action selection.

**Actor**: `features → action_logits` (8 directions)
**Critic**: `features → value_estimate` (expected return)

### 3. Forward Dynamics Model

**File**: `techniques/rl_navigation/forward_dynamics.py`

Predicts next semantic features given current features and action.

**Variants**:
- ICM: Action-conditioned prediction
- RND: Random network distillation

### 4. Image Navigation Environment

**File**: `techniques/rl_navigation/environment.py`

MDP formulation for image exploration.

**State**: Position, visited mask, semantic features
**Action Space**: 8-connected movement (or extended)
**Reward**: Rolling-window prediction accuracy

## Data Flow

### Training Loop

1. **Rollout Collection**:
   ```python
   state = env.reset()
   for step in range(rollout_steps):
       features = state["features"].detach()  # Critical!
       action, log_prob, value = policy.act(features)
       next_state, reward, done, info = env.step(action)
       buffer.store(features, action, reward, ...)
   ```

2. **PPO Update**:
   ```python
   advantages, returns = compute_gae(rewards, values, dones)
   for epoch in range(ppo_epochs):
       for batch in minibatches:
           policy_loss, value_loss = compute_losses(batch)
           optimizer.step()
   ```

3. **Predictor Update**:
   ```python
   pred_features = predictor(features_t, actions)
   predictor_loss = mse(pred_features, features_t1)
   predictor_optimizer.step()
   # In Phase 2, encoder gradients flow through this!
   ```

## Critical Design Choices

### Gradient Decoupling

**Problem**: If policy loss backprops into encoder, features become policy-specific.

**Solution**: Use `features.detach()` when passing to policy:

```python
# techniques/rl_navigation/trainer.py:203
with torch.no_grad():
    action, log_prob, value = self.policy.act(features.detach())
```

Encoder only updated via predictor gradients in Phase 2.

### Two-Phase Training

**Phase 1** (10k episodes):
- Encoder frozen
- Stable semantic space
- Policy learns navigation

**Phase 2** (5k episodes):
- Top 2 layers unfrozen
- Very small LR (1e-5)
- Task-specific adaptation

## Implementation Notes

### Memory Efficiency

- DINOv2 features precomputed and cached
- Rollout buffer uses fixed-size tensors
- Gradient checkpointing not needed (features detached)

### Computational Cost

- Forward pass: ~10ms (vits14 on GPU)
- Training step: ~50ms (includes PPO + predictor updates)
- Full Phase 1 (10k episodes): ~2-4 hours on V100

## Next Steps

See the [Training Guide](training.md) for practical tips on running experiments.
