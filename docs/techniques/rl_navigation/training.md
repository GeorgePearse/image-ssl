# Training Guide

*Practical guide to training RL navigation models.*

## Quick Start

```bash
# Test on synthetic image (fast)
python experiments/train_rl_navigator.py --config quick_test

# Train on your image
python experiments/train_rl_navigator.py \
    --image path/to/image.jpg \
    --config default

# Resume from checkpoint
python experiments/train_rl_navigator.py \
    --resume checkpoints/rl_navigator/phase1_ep1000.pt
```

## Configuration Presets

### quick_test
**Use for**: Testing, debugging, CI/CD

- Phase 1: 100 episodes
- Phase 2: 50 episodes
- Time: ~5-10 minutes on GPU

### default
**Use for**: Standard experiments

- Phase 1: 10,000 episodes
- Phase 2: 5,000 episodes
- Time: ~2-4 hours on V100

### rnd
**Use for**: Comparing intrinsic motivation methods

- Uses RND instead of ICM
- Same episode counts as default

### long
**Use for**: Maximum performance

- Phase 1: 20,000 episodes
- Phase 2: 10,000 episodes
- Larger model: `dinov2_vitb14`
- Time: ~8-12 hours on V100

## Training Phases

### Phase 1: Frozen Encoder

**Goal**: Learn navigation policy with stable semantic space

**What happens**:
- DINOv2 encoder completely frozen
- Policy learns from pretrained features
- Forward model learns semantic transitions

**Typical metrics**:
```
Episode 1000/10000
  Avg Reward: 8-12
  Avg Length: 200-400 steps
  Coverage: 30-60%
  Pred Error: 0.01-0.05
  Entropy: 1.5-2.0 (healthy exploration)
```

**Red flags**:
- Entropy < 0.5 → Policy collapsed, restart with higher entropy_coef
- Coverage < 10% → Coverage bonus too small, increase weight
- Avg Length < 50 → Episodes ending too early, check termination

### Phase 2: Fine-Tuned Encoder

**Goal**: Adapt encoder to task-specific semantic patterns

**What happens**:
- Top 2 transformer layers unfrozen
- Very small LR (1e-5)
- Encoder updated only via predictor gradients

**Typical metrics**:
```
Episode 1000/5000
  Avg Reward: 10-15 (slight improvement)
  Coverage: 40-70%
  Pred Error: 0.008-0.03 (decreases)
```

**Red flags**:
- Reward suddenly drops → LR too high, encoder diverging
- No improvement over Phase 1 → May not need Phase 2 for this image

## Hyperparameter Tuning

### Critical Parameters

#### reward_lambda (default: 0.1)
**Controls**: Exponential decay rate for future predictions

- **Lower** (0.05): More emphasis on distant predictions
- **Higher** (0.2): More emphasis on immediate predictions

**Tune if**: Agent seems too short-sighted or too random

#### coverage_bonus_weight (default: 0.1)
**Controls**: Penalty for revisiting regions

- **Lower** (0.05): More revisiting, deeper exploration
- **Higher** (0.5): Less revisiting, broader coverage

**Tune if**: Coverage too low or agent loops excessively

#### reward_horizon (default: 10)
**Controls**: How many steps ahead to predict

- **Lower** (5): Faster, more local
- **Higher** (20): Slower, more global planning

**Tune if**: Computational budget or exploration style needs

### PPO Parameters

Usually don't need tuning, but if unstable:

```python
clip_epsilon: 0.2    # Lower (0.1) for more conservative updates
policy_lr: 3e-4      # Lower (1e-4) if policy diverges
entropy_coef: 0.01   # Higher (0.05) for more exploration
```

## Monitoring Training

### Key Metrics to Watch

**Avg Reward**: Should increase over time
- Phase 1: 5 → 12
- Phase 2: 12 → 15

**Coverage**: Percentage of image visited
- Target: 50-80% (depends on image complexity)

**Pred Error**: Forward model accuracy
- Should decrease as model learns
- Too low (< 0.001) → May be overfitting

**Entropy**: Policy randomness
- Healthy: 1.0-2.0
- Too low (< 0.5) → Exploitation only
- Too high (> 2.5) → Not learning

**Policy Loss**: Should stabilize
- Healthy: 0.1-0.3
- Diverging → Reduce learning rate

### Tensorboard (Coming Soon)

```bash
# Future feature
tensorboard --logdir checkpoints/rl_navigator/logs
```

## Common Issues

### Training is slow

**Solutions**:
1. Use smaller model: `dinov2_vits14` (default)
2. Reduce `rollout_steps`: 2048 → 1024
3. Use CPU only for quick tests: `--device cpu`
4. Reduce image size in script (currently 512x512)

### Out of memory

**Solutions**:
1. Reduce `rollout_steps`: 2048 → 512
2. Reduce `ppo_batch_size`: 64 → 32
3. Use smaller model: `vitb14` → `vits14`
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Agent doesn't explore

**Symptoms**: Coverage < 20%, stays in one region

**Solutions**:
1. Increase `coverage_bonus_weight`: 0.1 → 0.5
2. Increase `entropy_coef`: 0.01 → 0.05
3. Check that reward is non-zero
4. Verify prediction error is being computed

### Training unstable

**Symptoms**: Reward/loss oscillates wildly

**Solutions**:
1. Reduce learning rates (all of them)
2. Increase `ppo_batch_size` for stability
3. Reduce `clip_epsilon`: 0.2 → 0.1
4. Check for NaN in losses (debug mode)

## Checkpointing

### Automatic Saves

Checkpoints saved every `save_interval` episodes (default: 1000):

```
checkpoints/rl_navigator/
├── phase1_ep1000.pt
├── phase1_ep2000.pt
├── ...
├── phase2_ep1000.pt
└── final.pt
```

### Manual Save

Press `Ctrl+C` during training:

```python
# trainer.py handles this
except KeyboardInterrupt:
    print("Saving checkpoint...")
    trainer.save_checkpoint("interrupted.pt")
```

### Resume Training

```bash
python experiments/train_rl_navigator.py \
    --resume checkpoints/rl_navigator/phase1_ep5000.pt
```

## Best Practices

### 1. Start Small
Always run `quick_test` first to verify:
- Code runs without errors
- Agent explores (coverage > 10%)
- Metrics look reasonable

### 2. Monitor Regularly
Check training every ~1000 episodes:
- Is coverage increasing?
- Is entropy healthy?
- Are there NaNs?

### 3. Compare Configurations
Run ablations to understand what matters:
- ICM vs RND
- Different reward horizons
- With/without Phase 2

### 4. Visualize Results
After training, always visualize:
```bash
python experiments/visualize_rl_paths.py \
    --checkpoint checkpoints/rl_navigator/final.pt \
    --n_episodes 5
```

## Advanced Topics

### Multi-Image Training

*Coming soon - currently single image only*

### Custom Reward Functions

Modify `environment.py:_compute_reward()`:

```python
def _compute_reward(self, prev_pos, action, current_pos):
    # Base: rolling-window accuracy
    base_reward = self._compute_lookahead_reward(current_pos)

    # Add custom rewards here
    semantic_bonus = self._compute_semantic_diversity(current_pos)

    return base_reward + semantic_bonus
```

### Distributed Training

*Coming soon - multi-GPU support*

## Next Steps

- See [Extensions](extensions.md) for jump/scout actions
- Check [TODO list](../../development/todo.md) for planned features
- Review [Architecture](architecture.md) for implementation details
