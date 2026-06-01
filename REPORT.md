# smart-hvac-rl-lab — Research Report

## Abstract

`smart-hvac-rl-lab` is a fully simulated reinforcement-learning sandbox for HVAC control in a single-zone building. Using a first-order RC thermal model as the environment dynamics, we train a Soft Actor-Critic (SAC) agent and compare it against a hysteresis thermostat and a PID controller across cold, mild, and hot day scenarios.

After reward shaping and hyperparameter tuning, SAC achieves the highest percentage of time inside the comfort band (67.3%) while using the least energy (91.1 kWh/episode) on cold-day evaluations.

---

# 1. Thermal Model

Single-node (1R1C) RC model for indoor temperature \(T_{in}\):

\[
C \frac{dT_{in}}{dt}
=
\frac{T_{out} - T_{in}}{R}
+
\eta P_{hvac}
+
Q_{int}
\]

Discretized via zero-order hold with timestep \(dt = 900\) s:

\[
T_{in}[k+1]
=
T_{ss}
+
\left(T_{in}[k] - T_{ss}\right)
e^{-dt/(RC)}
\]

where

\[
T_{ss}
=
T_{out}
+
R(\eta P_{hvac} + Q_{int})
\]

is the steady-state temperature.

| Symbol | Value | Unit | Description |
|----------|----------|----------|----------|
| C | 1,800,000 | J/K | Thermal capacitance |
| R | 0.005 | K/W | Thermal resistance (wall) |
| P_max | 6,000 | W | Maximum HVAC power |
| η | 1.0 | — | HVAC efficiency |
| dt | 900 | s | Timestep (15 min) |
| τ = RC | 2.5 | h | Thermal time constant |

### Weather Model

Sinusoidal outdoor temperature profile:

\[
T_{out}(t)
=
T_{mean}
+
A \sin\left(\frac{2\pi t}{24h} + \phi\right)
+
\varepsilon
\]

parameterized by `day_type`:

- Cold
- Mild
- Hot
- Random

---

# 2. MDP Formulation

| Component | Specification |
|------------|------------|
| State | \([T_{in}, T_{out}, \sin(\omega t), \cos(\omega t), occupancy, u_{prev}]\) |
| State Dimension | 6 |
| Action | \(u \in [0,1]\) |
| HVAC Power | \(P_{hvac}=uP_{max}\) |
| Comfort Band | \([21,23]^\circ C\) |
| Comfort Deviation | \(d=\max(T_{min}-T_{in},0,T_{in}-T_{max})\) |
| Discount Factor | \(\gamma=0.99\) |
| Episode Length | 96 steps × 900 s = 24 h |

### Reward Function

\[
r_t
=
-\lambda_c d^2
-\lambda_e P_{hvac}
+\lambda_b \mathbf{1}[d=0]
-\lambda_s(\Delta u)^2
\]

### Final Reward Weights

| Weight | Value | Role |
|----------|----------|----------|
| λc | 1.0 | Quadratic comfort penalty |
| λe | 0.0001 | Energy penalty |
| λb | 0.5 | In-band bonus |
| λs | 0.05 | Actuator smoothness penalty |

---

# 3. Algorithms

## Thermostat (Bang-Bang Control)

Heating policy with hysteresis:

```python
if T_in < T_min:
    u = 1
elif T_in > T_max:
    u = 0
else:
    u = u_prev
```

## PID Controller

Discrete PID control based on temperature error:

\[
e_t = T_{set} - T_{in}
\]

\[
u_t
=
\text{clip}
\left(
K_p e_t
+
K_i \sum e_t \, dt
+
K_d \frac{de_t}{dt},
0,
1
\right)
\]

Parameters:

- \(K_p = 0.15\)
- \(K_i = 0.008\)
- \(K_d = 0.5\)
- \(T_{set}=22^\circ C\)

Anti-windup protection is enabled.

## Soft Actor-Critic (SAC)

Following the formulation of Haarnoja et al. (2018):

\[
J(\pi)
=
\sum_t
\mathbb{E}
\left[
r_t
+
\alpha
H(\pi(\cdot|s_t))
\right]
\]

### Network and Training Parameters

- Policy network: MLP [64, 64]
- Activation: tanh
- Total timesteps: 200,000
- Learning rate: 3e-4
- Replay buffer: 100,000
- Batch size: 256
- Discount factor: 0.99
- Target smoothing coefficient τ = 0.005
- Automatic entropy temperature tuning

---

# 4. Tuning History

| Run | Key Change | Evaluation Reward |
|------|------------|------------|
| 1 | P_max=3kW, λe=0.001, random weather, 300k steps | -720 |
| 2 | P_max=6kW, λe=0.001, cold weather, 100k steps | -413 |
| 3 | P_max=6kW, λe=0.0001, cold weather, 200k steps | -64 |
| 4 | Added λb=0.5 in-band bonus | **-18.9** |

The dominant improvement came from introducing the in-band bonus. Without it, SAC converged to a degenerate strategy that minimized energy usage while tolerating a persistent comfort deviation.

---

# 5. Results — Cold Day Evaluation (10 Episodes)

| Policy | Comfort Deviation (K) | % Time In Band | Energy (kWh) | Total Reward |
|----------|----------|----------|----------|----------|
| Thermostat | 0.325 | 53.1% | 92.7 | -48.3 |
| PID | 0.144 | 66.1% | 92.1 | -17.8 |
| **SAC** | **0.185** | **67.3%** | **91.1** | **-18.5** |

### Discussion

- SAC achieves the highest percentage of time inside the comfort band.
- SAC also uses the least energy.
- PID achieves the lowest mean comfort deviation due to continuous proportional control.
- SAC and PID achieve nearly identical total reward values (difference ≈ 0.7).

---

# 6. Causal Day Experiment

To isolate the causal effect of policy selection, the same weather and occupancy trajectory is used for all controllers by fixing the random seed.

This ensures that:

- Weather is identical across policies.
- Occupancy patterns are identical across policies.
- Differences in indoor temperature trajectories arise solely from the controller.

### Key Observation

SAC learns anticipatory pre-heating behavior before occupied periods, reducing comfort violations during peak occupancy compared with the purely reactive thermostat controller.

Generated figures:

- `results/figures/causal_day_cold_seed42.png`
- `results/figures/causal_cumcost_cold_seed42.png`

---

# 7. Limitations

- Single-zone 1R1C thermal model
- No humidity dynamics
- No CO₂ modeling
- No solar radiation effects
- No inter-zone thermal coupling
- Trained only on cold-weather scenarios
- Cross-season generalization remains untested
- Synthetic sinusoidal weather only
- No real TMY3 weather data
- PPO baseline training not yet completed

---

# 8. Extensions

| Extension | Description |
|------------|------------|
| Multi-Zone Buildings | Use 2–3 coupled RC nodes and multi-agent RL |
| Time-of-Use Tariffs | Add dynamic electricity prices λe(t) |
| CO₂ Emissions | Add λco₂ κt P_hvac penalty |
| Model-Based RL / MPC | Exploit known RC dynamics for planning |
| Real Weather Data | Train using TMY3 weather datasets |
| PPO Comparison | Train PPO baseline and compare against SAC |

---

# 9. References

1. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd Edition). MIT Press, 2018.

2. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML, 2018.

3. Wei, T., Wang, Y., & Zhu, Q. *Deep Reinforcement Learning for Building HVAC Control*. DAC, 2017.

4. Gao, G., Li, J., Wen, Y., & Wang, C. *DeepComfort: Energy-Efficient Thermal Comfort Control via Reinforcement Learning*. IEEE Internet of Things Journal, 2020.

5. Standard building physics literature on first-order 1R1C lumped-parameter thermal models.