# smart-hvac-rl-lab

> **A reproducible RL sandbox for learning HVAC control policies in simulated buildings.**

## Overview

`smart-hvac-rl-lab` is a fully simulated, self-contained research prototype
that trains a reinforcement-learning agent to control heating/cooling in a
single-zone building. No physical sensors, no hardware, no paid APIs — only
open-source Python.

**Stack**: NumPy · SciPy · Gymnasium · Stable-Baselines3 (SAC / PPO) · Matplotlib

---

## Quickstart

```bash
# 1. Clone and enter
cd "smart-hvac-rl-lab"

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate                         # Windows

# 3. Install
pip install -e ".[dev]"

# 4. Run baselines
python scripts/run_baselines.py --day-type cold --n-episodes 10

# 5. Train SAC (300k steps, ~5-10 min on CPU)
python scripts/train_rl.py --algo sac --timesteps 300000

# 6. Run tests
pytest tests/ -v
```

## Repository Structure

```
smart-hvac-rl-lab/
├── src/smart_hvac/
│   ├── core/           # RC thermal model + Parameters dataclass
│   ├── envs/           # Gymnasium HvacEnv
│   ├── control/        # Thermostat & PID baselines
│   ├── rl/             # SAC & PPO training scripts
│   └── evaluation/     # Rollout runner, metrics, plots
├── scripts/            # CLI entry points
├── notebooks/          # 01_baseline · 02_train_sac · 03_policy_analysis
├── tests/              # Pytest unit tests
├── results/            # models/ · figures/ · logs/
└── pyproject.toml
```

## Research framing

The project uses a **1-node RC thermal model** (discretised via exact ZOH) as
the environment transition kernel, formulated as an MDP and solved with SAC.
Baselines are a hysteresis thermostat and a discrete PID controller.

### Possible extensions
- Multi-zone building (2-3 coupled RC nodes, multi-agent RL)
- Time-of-use electricity tariffs (`λ_e` time-varying)
- CO₂ emissions as an additional cost term
- Model-based RL / differentiable MPC
