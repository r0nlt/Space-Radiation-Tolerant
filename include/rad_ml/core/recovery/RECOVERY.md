# Recovery Core Module — `rad_ml/core/recovery/`

---

## Overview

The `rad_ml/core/recovery` directory provides the core infrastructure for **checkpointing and recovery** in the `rad_ml` framework. It is essential for enabling robust, fault-tolerant operation in radiation-prone and mission-critical environments.

> This module ensures that critical data and system state can be reliably restored after radiation-induced errors, faults, or system interruptions, supporting both engineering resilience and scientific reliability.

---

## Purpose and Responsibilities

- **Fault Recovery:**
  - Ensures reliable restoration of system state after faults or upsets.
- **Checkpointing:**
  - Periodically saves ("checkpoints") the state of neural networks, memory, or other critical components.
- **State Management:**
  - Manages storage, retrieval, and validation of checkpoints, ensuring data integrity even in the presence of radiation-induced corruption.
- **Recovery Logic:**
  - Detects when recovery is needed (e.g., after error detection or system reset) and restores the system state from the most recent valid checkpoint.
- **Integration:**
  - Works in conjunction with error detection, memory protection, and adaptive protection modules for holistic resilience.

---

## Key Component: [`checkpoint_manager.hpp`](checkpoint_manager.hpp)

- **`CheckpointManager` Class:**
  - **Checkpoint Creation:**
    - Supports periodic or event-driven checkpointing of system state (e.g., neural network weights, memory regions).
  - **Integrity Validation:**
    - Uses checksums, redundancy, or other mechanisms to validate checkpoint integrity.
  - **State Restoration:**
    - Restores system state from the most recent valid checkpoint upon error detection or system restart.
  - **Storage Options:**
    - Supports both in-memory and persistent (disk/flash) checkpoint storage for flexibility in different deployment scenarios.
  - **Configuration:**
    - Allows configuration of checkpoint intervals, retention policies, and validation strategies.

---

## Typical Use Cases

- **Neural Network Training/Inference:**
  - Periodically checkpointing model weights and optimizer state to recover from soft errors or power loss.
- **Mission-Critical Systems:**
  - Ensuring that spacecraft, satellites, or other autonomous systems can recover from radiation-induced upsets without losing mission progress.
- **Testing and Simulation:**
  - Enabling fault injection and recovery scenarios to validate system robustness.

---

## Technical Integration

- **Error Detection:**
  - Triggers recovery when uncorrectable errors are detected by memory or computation protection modules.
- **Memory Protection:**
  - Coordinates with memory scrubbing and ECC/TMR modules to ensure checkpoints are themselves protected.
- **Adaptive Protection:**
  - Can escalate checkpointing frequency in response to increased error rates or environmental hazards.
- **API Integration:**
  - Exposed via high-level API functions for easy use in application code and mission simulators.

---

## Summary

> The `rad_ml/core/recovery` directory is a foundational component for resilience in the `rad_ml` framework, providing robust checkpointing and recovery mechanisms to ensure reliable operation in the face of radiation and other hazards.
