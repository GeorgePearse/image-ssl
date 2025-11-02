# Research References

This section contains detailed summaries of the key papers that form the theoretical foundation of our RL-based image navigation implementation.

## Overview

Our approach synthesizes ideas from five major research areas:

1. **Curiosity-Driven RL** - Using prediction error as intrinsic motivation
2. **Semantic Visual Features** - Self-supervised learning for robust representations
3. **Policy Optimization** - Stable, sample-efficient RL algorithms
4. **Advantage Estimation** - Variance reduction in policy gradients
5. **Hierarchical RL** - Extended action spaces for long-range planning

## Key Papers

### RL Navigation Foundation

These five papers directly inform our implementation:

| Paper | Year | Contribution | Our Implementation |
|-------|------|--------------|-------------------|
| [ICM](rl_navigation/01_curiosity_driven_exploration_ICM.md) | 2017 | Curiosity via prediction error | `ForwardDynamicsModel` |
| [RND](rl_navigation/02_random_network_distillation_RND.md) | 2018 | Random network distillation | `RNDIntrinsicMotivation` |
| [PPO](rl_navigation/03_proximal_policy_optimization_PPO.md) | 2017 | Clipped policy optimization | `PPOTrainer` |
| [DINOv2](rl_navigation/04_dinov2_visual_features.md) | 2023 | Semantic visual features | `SemanticEncoder` |
| [GAE](rl_navigation/05_generalized_advantage_estimation_GAE.md) | 2015 | Advantage estimation | `compute_gae()` |

### Quick Reference

#### Intrinsic Motivation

**When to use ICM vs RND?**

- **ICM (Forward Dynamics):**
  - Action-relevant exploration
  - Learn what actions lead where
  - More sample efficient
  - Can be less stable

- **RND (Random Network Distillation):**
  - State-space novelty
  - More stable training
  - Simpler implementation
  - Less action-focused

#### Feature Learning

**Why DINOv2 over other encoders?**

DINOv2 provides:
- ✅ Pre-trained on diverse data (142M images)
- ✅ Semantic invariance (car color problem)
- ✅ Patch-level features (natural for navigation)
- ✅ Multiple model sizes (21M - 1.1B params)
- ✅ Strong zero-shot transfer

Alternatives considered:
- CLIP: Language-biased, less dense features
- ResNet: Lower-level features, less semantic
- MAE: Reconstruction-focused, not semantic

#### Policy Optimization

**Why PPO over other methods?**

PPO balances:
- ✅ Sample efficiency (vs A3C, REINFORCE)
- ✅ Stability (vs TRPO complexity)
- ✅ Simplicity (vs SAC, TD3)
- ✅ Works with discrete actions

For image navigation:
- Episodes can be 500+ steps
- Need stable learning (don't want collapse)
- Discrete action space (8 directions)
- PPO is battle-tested choice

## How Papers Relate

```
Image → DINOv2 → Features
         [#4]        │
                     │
         ┌───────────┴──────────┐
         │                      │
         ▼                      ▼
    Policy (π)           Forward Model (P)
    PPO [#3]             ICM/RND [#1,#2]
         │                      │
         │                      ▼
         │              Prediction Error
         │              (Intrinsic Reward)
         │                      │
         └──────────┬───────────┘
                    ▼
              PPO Update
              with GAE [#5]
```

### The Critical Insights

**1. Flip the Reward ([ICM](rl_navigation/01_curiosity_driven_exploration_ICM.md))**

Don't predict accurately → predict **poorly**!

High prediction error = novel/surprising = informative

**2. Semantic Space ([DINOv2](rl_navigation/04_dinov2_visual_features.md))**

Predict in feature space, not pixel space

Red car ≈ Blue car in embeddings, ≠ in pixels

**3. Stable Updates ([PPO](rl_navigation/03_proximal_policy_optimization_PPO.md))**

Clip policy updates to prevent catastrophic collapse

```python
ratio = π_new / π_old
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clipped_ratio * A)
```

**4. Two-Phase Training (Novel)**

Phase 1: Frozen encoder → stable learning
Phase 2: Fine-tune encoder → task adaptation

**5. Gradient Decoupling (Novel)**

Policy uses `features.detach()` → no policy gradients into encoder

Encoder only updated via predictor loss

## Additional Reading

### Hierarchical RL

For jump/scout actions extension:

- Sutton et al., "Between MDPs and semi-MDPs" (1999)
- Kulkarni et al., "Hierarchical Deep RL" (2016)
- Bacon et al., "The Option-Critic Architecture" (2017)

### Exploration in RL

Related intrinsic motivation methods:

- Schmidhuber, "Formal Theory of Creativity" (2010)
- Stadie et al., "Incentivizing Exploration" (2015)
- Badia et al., "Never Give Up" (2020)

### Self-Supervised Vision

Other feature learning approaches:

- Chen et al., "SimCLR" (2020)
- He et al., "Masked Autoencoders" (2021)
- Caron et al., "DINOv1" (2021)

## Citation Guide

When citing our work or building upon it, please cite the relevant foundation papers.

**Minimal citation** (just ICM + DINOv2):
```bibtex
@inproceedings{pathak2017curiosity,
  title={Curiosity-driven exploration by self-supervised prediction},
  author={Pathak, Deepak and Agrawal, Pulkit and Efros, Alexei A and Darrell, Trevor},
  booktitle={ICML},
  year={2017}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

**Complete citation** (all five papers):

See individual paper pages for full BibTeX.

## Python RL Frameworks

When implementing RL systems, you can choose from several mature frameworks. This comprehensive list is sourced from [awesome-deep-rl](https://github.com/kengz/awesome-deep-rl), sorted by GitHub stars.

### Most Popular Frameworks (10K+ Stars)

| Framework | Stars | Description |
|-----------|-------|-------------|
| **[Ray RLlib](https://github.com/ray-project/ray)** | 39.6K ★ | An open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications |
| **[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)** | 18.8K ★ | Unity Machine Learning Agents Toolkit - train agents in Unity environments |
| **[OpenAI Baselines](https://github.com/openai/baselines)** | 16.5K ★ | High-quality implementations of reinforcement learning algorithms (now superseded by Stable-Baselines3) |
| **[Google Dopamine](https://github.com/google/dopamine)** | 10.8K ★ | A research framework for fast prototyping of reinforcement learning algorithms |

### Widely-Used Frameworks (3K-10K Stars)

| Framework | Stars | Description |
|-----------|-------|-------------|
| **[Tianshou](https://github.com/thu-ml/tianshou)** | 8.9K ★ | Reinforcement learning platform based on pure PyTorch with elegant API |
| **[DeepMind OpenSpiel](https://github.com/deepmind/open_spiel)** | 4.8K ★ | Environments and algorithms for research in general reinforcement learning and search/planning in games |
| **[Stable Baselines](https://github.com/hill-a/stable-baselines)** | 4.3K ★ | Fork of OpenAI Baselines with improvements (now use Stable-Baselines3 instead) |
| **[pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)** | 3.8K ★ | PyTorch implementations of A2C, PPO, ACKTR, and GAIL |
| **[DeepMind Acme](https://github.com/deepmind/acme)** | 3.8K ★ | Research framework for reinforcement learning with distributed training support |
| **[Facebook ReAgent](https://github.com/facebookresearch/ReAgent)** | 3.7K ★ | Platform for Reasoning systems (Reinforcement Learning, Contextual Bandits, etc.) |
| **[ShangtongZhang DeepRL](https://github.com/ShangtongZhang/DeepRL)** | 3.4K ★ | Modularized implementation of Deep RL algorithms in PyTorch |
| **[Catalyst](https://github.com/catalyst-team/catalyst)** | 3.4K ★ | Accelerated deep learning and reinforcement learning framework |
| **[TensorForce](https://github.com/tensorforce/tensorforce)** | 3.3K ★ | TensorFlow library for applied reinforcement learning |
| **[DeepMind TRFL](https://github.com/deepmind/trfl)** | 3.1K ★ | TensorFlow Reinforcement Learning - building blocks for RL agents |
| **[PyTorch TorchRL](https://github.com/pytorch/rl)** | 3.1K ★ | Official PyTorch library for reinforcement learning |
| **[TensorFlow Agents](https://github.com/tensorflow/agents)** | 3.0K ★ | Library for reinforcement learning in TensorFlow |

### Established Frameworks (1K-3K Stars)

| Framework | Stars | Description |
|-----------|-------|-------------|
| **[RLkit](https://github.com/rail-berkeley/rlkit)** | 2.8K ★ | Reinforcement learning framework and algorithms in PyTorch (UC Berkeley) |
| **[NervanaSystems Coach](https://github.com/NervanaSystems/coach)** | 2.3K ★ | Reinforcement Learning Coach by Intel AI Lab |
| **[rlpyt](https://github.com/astooke/rlpyt)** | 2.3K ★ | Reinforcement Learning in PyTorch with efficient implementations |
| **[Facebook ELF](https://github.com/facebookresearch/ELF)** | 2.1K ★ | Platform for game research with AlphaGoZero/AlphaZero reimplementation |
| **[garage](https://github.com/rlworkgroup/garage)** | 2.0K ★ | Toolkit for reproducible reinforcement learning research |
| **[MAgent](https://github.com/geek-ai/MAgent)** | 1.7K ★ | Platform for many-agent reinforcement learning |
| **[d3rlpy](https://github.com/takuseno/d3rlpy)** | 1.6K ★ | Offline deep reinforcement learning library |
| **[Softlearning](https://github.com/rail-berkeley/softlearning)** | 1.4K ★ | Framework for training maximum entropy policies in continuous domains (UC Berkeley) |
| **[SLM Lab](https://github.com/kengz/SLM-Lab)** | 1.3K ★ | Modular deep reinforcement learning framework in PyTorch |
| **[ChainerRL](https://github.com/chainer/chainerrl)** | 1.2K ★ | Deep reinforcement learning library built on top of Chainer |

### Emerging & Specialized Frameworks (Under 1K Stars)

| Framework | Stars | Description |
|-----------|-------|-------------|
| **[MushroomRL](https://github.com/MushroomRL/mushroom-rl)** | 913 ★ | Python library for reinforcement learning experiments |
| **[skrl](https://github.com/Toni-SM/skrl)** | 893 ★ | Modular RL library (PyTorch/JAX) with NVIDIA Isaac Gym, Omniverse support |
| **[RLtools](https://github.com/rl-tools/rl-tools)** | 886 ★ | Fastest deep RL library for continuous control in pure C++ (Python bindings available) |
| **[AgileRL](https://github.com/AgileRL/AgileRL)** | 839 ★ | Deep RL library focused on improving development by introducing **RLOps** - MLOps for reinforcement learning. Features evolutionary hyperparameter optimization and accelerated training |
| **[OpenRL](https://github.com/OpenRL-Lab/openrl)** | 785 ★ | Open-source general reinforcement learning research framework |
| **[Rofunc](https://github.com/Skylark0924/Rofunc)** | 668 ★ | Full-process Python library for robot learning from demonstration and robot manipulation |
| **[reaver](https://github.com/inoryy/reaver)** | 561 ★ | Modular deep RL framework focused on StarCraft II tasks |
| **[pytorch-rl](https://github.com/navneet-nmk/pytorch-rl)** | 451 ★ | Model-free deep reinforcement learning algorithms in PyTorch |
| **[RLgraph](https://github.com/rlgraph/rlgraph)** | 323 ★ | Modular computation graphs for deep reinforcement learning |
| **[K-Scale ksim-gym](https://github.com/kscalelabs/ksim-gym)** | 303 ★ | Making robots useful with RL - built on top of K-Sim |
| **[Maze](https://github.com/enlite-ai/maze)** | 283 ★ | Application-oriented deep RL framework for real-world decision problems |
| **[K-Scale ksim](https://github.com/kscalelabs/ksim)** | 211 ★ | Modular framework for training policies in simulation |

### Additional Libraries

| Framework | Stars | Description |
|-----------|-------|-------------|
| **[DI-engine](https://github.com/opendilab/DI-engine)** | 3.5K ★ | Generalized decision intelligence engine supporting various deep RL algorithms |
| **UMass Autonomous Learning Library** | - | PyTorch library for building deep RL agents (see [awesome-deep-rl](https://github.com/kengz/awesome-deep-rl)) |
| **vel** | - | Research on bringing velocity to deep-learning (see [awesome-deep-rl](https://github.com/kengz/awesome-deep-rl)) |

### Quick Reference: Top Picks by Use Case

| Use Case | Recommended Framework | Why? |
|----------|----------------------|------|
| **Getting Started** | Stable-Baselines3 | Best docs, easiest API, PyTorch-based |
| **Production/Scale** | Ray RLlib | Battle-tested, distributed, 39K+ stars |
| **RLOps/MLOps** | **AgileRL** | **Evolutionary hyperparameter optimization, accelerated training, RL-specific ops** |
| **Learning RL** | CleanRL | Single-file implementations, highly readable |
| **Research** | Tianshou, Acme | Fast, modular, well-maintained |
| **Game Environments** | Unity ML-Agents, OpenSpiel | Game-specific optimizations |
| **Offline RL** | d3rlpy | Specialized for offline/batch RL |
| **Robotics** | skrl, Rofunc | NVIDIA Isaac support, robot-specific |
| **Multi-Agent** | MAgent, PettingZoo | Many-agent scenarios |
| **Pure Performance** | RLtools | C++ implementation, fastest |
| **Understanding Code** | CleanRL, DeepRL | Clean implementations, educational |
| **Hyperparameter Tuning** | AgileRL, Ray RLlib | Evolutionary optimization / distributed tuning |

---

### Our Implementation

**Visual Next Token** uses a **custom PPO implementation** because:

- ✅ Full control over gradient flow (detached features for policy)
- ✅ Tight integration with DINOv2 encoder
- ✅ Two-phase training (freeze → fine-tune encoder)
- ✅ Custom reward computation (rolling-window accuracy)
- ✅ Educational value (understand PPO internals)

**If you want to use an existing framework**, here's how you'd adapt:

#### Stable-Baselines3 Example

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# 1. Wrap environment in Gymnasium interface
class GymImageNavEnv(gym.Env):
    def __init__(self, image, encoder, predictor):
        self.env = ImageNavigationEnv(image, encoder, predictor)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(encoder.feature_dim,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(8)

    def reset(self, seed=None):
        state = self.env.reset()
        return state["features"].cpu().numpy(), {}

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state["features"].cpu().numpy(), reward, done, False, info

# 2. Train with SB3
env = GymImageNavEnv(image, encoder, predictor)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=100000)
```

#### CleanRL Example

CleanRL's single-file implementations are excellent for understanding and modifying algorithms. You could adapt [cleanrl/ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) by:

1. Replacing environment with `ImageNavigationEnv`
2. Modifying reward computation for rolling-window accuracy
3. Adding two-phase training logic

### Choosing a Framework

**Use Stable-Baselines3 if**:
- You want quick experiments with standard algorithms
- You need reliable, battle-tested implementations
- Documentation and community support are important

**Use CleanRL if**:
- You want to understand algorithm internals
- You plan to modify or extend algorithms
- You prefer minimal dependencies

**Use AgileRL if**:
- You want evolutionary hyperparameter optimization built-in
- You need RLOps workflows (experiment tracking, reproducibility)
- You want accelerated training with automatic tuning
- You're building production RL pipelines

**Use Ray RLlib if**:
- You need distributed training across many machines
- You're running hyperparameter sweeps at scale
- You're deploying to production with complex orchestration

**Use Custom Implementation (like ours) if**:
- You need tight integration with custom components
- You want full control over training dynamics
- You're doing research on novel RL methods

---

### Additional Resources

For the complete list of deep RL resources including papers, tutorials, and more frameworks, see:

**[awesome-deep-rl](https://github.com/kengz/awesome-deep-rl)** - A curated list of awesome Deep RL resources

This comprehensive repository includes:
- Research papers organized by topic
- Additional frameworks and libraries
- Tutorials and courses
- Benchmarks and environments
- Books and documentation

**Note**: Star counts were fetched on 2025-11-02 and may have changed since then.

## Next Steps

<div class="grid cards" markdown>

-   :material-file-document:{ .lg .middle } __Read the Papers__

    ---

    Detailed summaries with code connections

    [:octicons-arrow-right-24: RL Navigation Papers](rl_navigation/01_curiosity_driven_exploration_ICM.md)

-   :fontawesome-solid-brain:{ .lg .middle } __Understand the Architecture__

    ---

    How papers combine into working system

    [:octicons-arrow-right-24: Architecture](../techniques/rl_navigation/architecture.md)

-   :material-code-braces:{ .lg .middle } __See the Code__

    ---

    Implementation details and API

    [:octicons-arrow-right-24: API Reference](../api/rl_navigation.md)

</div>
