# GlucoMind-14

A physiologically grounded glucose-insulin ODE simulator for Type 1 Diabetes
(T1D) under Multiple Daily Injection (MDI) therapy.

The model extends the Dalla Man et al. (2006, 2007) glucose-insulin framework
with smooth renal clearance, circadian modulation, and stochastic
stress/exercise events. It is designed for generating realistic virtual
patient cohorts and synthetic blood glucose monitoring data.

## Features

- **11-state ODE model** of glucose–insulin dynamics
  - Plasma and interstitial glucose
  - Two-compartment gastric emptying
  - Subcutaneous + basal insulin pharmacokinetics
  - Remote insulin action (minimal-model style)
  - Smooth renal clearance above the 180 mg/dL renal threshold
  - Circadian modulation of insulin sensitivity and EGP
  - Stress (↑ EGP, ↓ insulin clearance) and exercise (↑ glucose uptake)

- **Realistic MDI event scheduling**
  - Stochastic meal timing/size, carb counting error, bolus dosing error
  - Daily basal insulin with dose noise
  - Heteroscedastic BGM measurement noise (SD = α + β·Gp)

- **Virtual patient populations**
  - Lognormal parameter sampling with insulin-sensitivity correlation coupling
  - Physiological bounds enforced with clipping

- **Parameter estimation & identifiability**
  - Maximum a posteriori estimation via L-BFGS-B
  - Profile likelihoods
  - Fisher Information Matrix and rank-saturation analysis

- **Clinical metrics**
  - TIR / TBR / TAR, mean/std glucose, fasting glucose, postprandial peak

## Installation

```bash
pip install -r requirements.txt
```

For development with tests and plotting:

```bash
pip install ".[dev]"
```

Or install the package itself:

```bash
pip install -e .
```

## Quickstart

Run a default 3-day test trajectory (saves a plot to `output/figures/`):

```bash
python run.py
```

Run a custom-duration simulation:

```bash
python run.py --days 7
```

Run a 100-patient population study:

```bash
python run.py --population 100
```

## Library usage

```python
import numpy as np
from src import (
    Parameters, N_STATES, simulate_dense, get_bgm_times,
    generate_mdi_schedule, sample_bgm_observations, time_in_range,
)

# 1. Configure a single patient
p = Parameters()

# 2. Build a 3-day MDI protocol
days = 3
events = generate_mdi_schedule(days=days)
t_span = (0.0, days * 24.0)

# 3. Simulate
state0 = np.zeros(N_STATES)
state0[0] = 120.0   # plasma glucose
state0[1] = 90.0    # interstitial glucose
t, y = simulate_dense(state0, events, p, t_span, dt=0.1)

# 4. Glycemic metrics
print(f"TIR: {100 * time_in_range(y[:, 0]):.1f}%")
print(f"Mean glucose: {y[:, 0].mean():.0f} mg/dL")

# 5. Synthetic BGM observations
bgm_times = get_bgm_times(events, t_span)
obs = sample_bgm_observations(bgm_times, t, y, p, seed=42)
```

### Generate a virtual population

```python
from src import generate_population

patients = generate_population(n_patients=100, seed=123)
```

### Estimate parameters from observations

```python
from src import estimate_map

p_opt, info = estimate_map(
    observations=obs,
    bgm_times=bgm_times,
    events=events,
    initial_state=state0,
    p_guess=Parameters(),
    p_nominal=Parameters(),
    param_cv=...,
    fixed_params={"V_Gi": 12.0, "EGP_0": 60.0},
    estimable_names=["k_u", "k_cl", "k_x", "k_EGP"],
    t_span=t_span,
)
```

## Model overview

| State | Description | Units |
|---|---|---|
| `Gp` | Plasma glucose | mg/dL |
| `Gi` | Interstitial glucose | mg/dL |
| `Qsto1`, `Qsto2` | Stomach content (solid, liquid) | g |
| `Isc1`, `Isc2` | Subcutaneous insulin | mU |
| `Ibase1`, `Ibase2` | Basal insulin compartments | mU |
| `Ibase` | Active basal insulin | mU |
| `Ip` | Plasma insulin | mU/L |
| `X` | Remote insulin action | — |

Full parameter table with units is available in `src/model.py` (`Parameters`).

## Project layout

```
GlucoMind14/
├── src/
│   ├── model.py          # ODE model, parameters, RHS
│   ├── simulate.py       # ODE integration (sparse & dense)
│   ├── events.py         # MDI event scheduling
│   ├── observe.py        # BGM observation model
│   ├── population.py     # Virtual patient generation
│   ├── estimation.py     # MAP estimation & profile likelihood
│   ├── identifiability.py# FIM & rank-saturation analysis
│   └── analysis.py       # Clinical glycemic metrics
├── config/
│   └── default.yaml      # Simulation defaults
├── tests/                # pytest suite
├── run.py                # CLI entry point
└── pyproject.toml
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this software in research, please cite:

> GlucoMind AI. *GlucoMind-14: A physiologically grounded glucose-insulin
> simulator for Type 1 Diabetes.* 2026. https://github.com/abderraouf-boudjema/GlucoMind-14

## Disclaimer

## Known Limitations

- **No hypoglycemic self-correction behavior.** The virtual population sampling
  does not currently model patient corrective action in response to
  hypoglycemia (e.g., corrective carbohydrate intake triggered by a low BGM
  reading). As a result, the tail of a generated population (~5th percentile,
  typically the most insulin-sensitive virtual patients) can show extended,
  clinically implausible hypoglycemic episodes rather than the
  self-terminating episodes seen in real patients. Users training downstream
  models on population-level data should either filter these tail cases or
  treat them as representing worst-case/untreated scenarios rather than
  typical patient behavior.
- **Plasma–interstitial lag is a fixed model constant**, not patient-specific
  or empirically re-fit to CGM data beyond the original Dalla Man
  parameterization. This is a known source of divergence between `Gp` and
  `Gi` readings, most visible during rapid glucose excursions.
- This is a **synthetic data generator**, not a validated clinical model. No
  claims are made about accuracy against real patient cohorts; parameter
  estimation and identifiability tools are provided for research use, not
  clinical calibration.
This is a simulation model for research and educational use. It is **not**
a medical device and must not be used for clinical decision-making.
