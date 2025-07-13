# How Radiation Affects Computing

High-energy particles from space radiation strike semiconductor materials in computing hardware, they can cause several types of errors:

- **Single Event Upset (SEU)**: A change in state caused by one ionizing particle striking a sensitive node in a microelectronic device
- **Multiple Bit Upset (MBU)**: Multiple bits flipped from a single particle strike
- **Single Event Functional Interrupt (SEFI)**: A disruption of normal operations (typically requiring a reset)
- **Single Event Latch-up (SEL)**: A potentially destructive condition involving parasitic circuit elements creating a low-resistance path

These effects can corrupt data in memory, alter computational results, or even permanently damage hardware. In space environments where maintenance is impossible, radiation tolerance becomes critical for mission success.

Space-Radiation-Tolerant addresses these challenges through software-based protection mechanisms that detect and correct radiation-induced errors, allowing ML systems to operate reliably even in harsh radiation environments. The software framework is intended to work alongside hardware protection strategies to achieve enhanced protection through hybrid protection methods.
