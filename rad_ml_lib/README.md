# Space Radiation-Tolerant Neural Network Library

A comprehensive library for implementing radiation-tolerant neural networks for space applications.

## Features

- Bit-level radiation protection mechanisms
- Multiple defense strategies (TMR, Reed-Solomon, etc.)
- Hardware-accelerated radiation simulation
- Customizable protection levels
- Environment-specific optimizations
- Performance analysis tools

## Installation

```bash
# Install from PyPI
pip install rad-ml-lib

# Or install from source
git clone https://github.com/yourusername/Space-Radiation-Tolerant.git
cd Space-Radiation-Tolerant
pip install -e .
```

## Quick Start

```python
import torch
from rad_ml_lib import protect_network, DefenseConfig, Environment

# Create a standard PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Create a protection configuration for Jupiter environment
config = DefenseConfig.for_environment(Environment.JUPITER)

# Apply radiation protection
protected_model = protect_network(model, config)

# Use the protected model as normal
inputs = torch.randn(1, 784)
outputs = protected_model(inputs)
```

## Protection Methods

- **Triple Modular Redundancy (TMR)**: Replicates critical components three times and uses voting
- **Reed-Solomon Coding**: Uses error correction codes for data protection
- **Adaptive Protection**: Dynamically adjusts protection based on environmental conditions
- **Physics-Driven**: Uses models of radiation physics to optimize protection

## Environments

- Earth
- Low Earth Orbit (LEO)
- Geostationary Earth Orbit (GEO)
- Lunar
- Mars
- Jupiter
- South Atlantic Anomaly (SAA)
- Solar Storms

## Citation

If you use this library in your research, please cite:

```
@software{nuguru2025radiationnn,
  author = {Nuguru, Rishab},
  title = {Space Radiation-Tolerant Neural Networks},
  year = {2025},
  url = {https://github.com/yourusername/Space-Radiation-Tolerant}
}
```

## License

GNU General Public License v3.0
