# Frequently Asked Questions (FAQ)

---

## What is the rad_ml Framework?

The `rad_ml` framework is an open-source toolkit for developing, simulating, and validating **radiation-tolerant machine learning and embedded systems**. It provides tools for error injection, adaptive protection, checkpointing, and mission simulation—enabling robust AI and software for space, avionics, and other radiation-prone environments.

---

## How is this different from GEANT4?

- **GEANT4** is a comprehensive, physics-accurate particle transport toolkit used for simulating the passage of particles through matter (e.g., in high-energy physics, medical physics, and shielding design).
- **rad_ml** focuses on the *system and ML level*: simulating the effects of radiation on neural networks, embedded systems, and mission-critical software, with fast, abstracted models suitable for rapid prototyping and research.
- If you need detailed, physics-level simulation, use GEANT4. If you want to study ML robustness, error mitigation, or system-level effects, use rad_ml.

---

## What hardware is this for?

- Designed for **space-grade, avionics, and mission-critical hardware** (e.g., FPGAs, microcontrollers, radiation-hardened processors).
- Also useful for research on commercial off-the-shelf (COTS) hardware in harsh environments.

---

## How do I use the framework?

1. **Install dependencies** (see [README.md](README.md)).
2. **Explore the [Documentation Index](README.md#documentation-index)** for module overviews and usage guides.
3. Use provided examples in the `examples/` directory to get started with:
   - Radiation-aware neural network training
   - Mission simulation
   - Error injection and recovery
4. Integrate with your own ML models or embedded systems as needed.

---

## What are the limitations?

- Not a replacement for full physics simulation.
- Focuses on system-level and ML-level effects, not detailed material/geometry modeling.
- Some features are experimental; see [Research Module](include/rad_ml/research/RESEARCH.md) for advanced capabilities.

---

## Can I simulate specific particle types (protons, heavy ions, etc.)?

**Yes!** While the framework abstracts over particle type at the system level, you can accurately simulate the effects of any particle (proton, heavy ion, neutron, etc.) by setting the appropriate flux and cross-section parameters for your scenario.

- For a detailed guide, see: [Simulating Specific Particle Types](include/rad_ml/radiation/PARTICLE_TYPES.md)
- Advanced users can also use custom properties (LET, energy, TID, SEL probability) for more detailed modeling.

---

## How do I contribute?

- See [CONTRIBUTING.md](CONTRIBUTING.md) (if available) or open an issue/pull request on GitHub.
- Contributions are welcome for new error models, protection strategies, documentation, and more!

---

## Where can I learn more?

- [API Documentation](include/rad_ml/api/API.md)
- [Radiation Modeling](include/rad_ml/radiation/RADIATION.md)
- [Recovery & Checkpointing](include/rad_ml/core/recovery/RECOVERY.md)
- [Research Tools](include/rad_ml/research/RESEARCH.md)
- [Simulating Specific Particle Types](include/rad_ml/radiation/PARTICLE_TYPES.md)
- [README.md](README.md) for project overview and quickstart

---

*If your question isn't answered here, please check the documentation or open an issue on GitHub!*
