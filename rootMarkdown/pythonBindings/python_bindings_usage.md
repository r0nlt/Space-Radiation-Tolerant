# Python Bindings Usage (v0.9.5)

As of v0.9.5, the framework now provides Python bindings for key radiation protection features, making the technology more accessible to data scientists and machine learning practitioners. Python bindings will soon be expanded upon for tensor flow and pytorch integration in the near future.

## Basic Usage with TMR Protection

```python
import rad_ml_minimal as rad_ml
from rad_ml_minimal.rad_ml.tmr import StandardTMR

# Initialize the framework
rad_ml.initialize()

# Create a TMR-protected integer
protected_value = StandardTMR(42)

# Use the protected value
print(f"Protected value: {protected_value.value}")

# Check integrity
if protected_value.check_integrity():
    print("Value integrity verified")

# Simulate a radiation effect
# In production code, this would happen naturally in radiation environments
# This is just for demonstration purposes
protected_value._v1 = 43  # Corrupt one copy

# Check integrity again
if not protected_value.check_integrity():
    print("Corruption detected!")

    # Attempt to correct the error
    if protected_value.correct():
        print(f"Error corrected, value restored to {protected_value.value}")

# Shutdown the framework
rad_ml.shutdown()
```

## Advanced TMR Demonstration

For a comprehensive demonstration of TMR protection against radiation effects:

```python
import rad_ml_minimal
from rad_ml_minimal.rad_ml.tmr import EnhancedTMR
import random

# Initialize
rad_ml_minimal.initialize()

# Create TMR-protected values of different types
protected_int = EnhancedTMR(100)
protected_float = EnhancedTMR(3.14159)

# Simulate radiation-induced bit flips on these values
def simulate_bit_flip(value, bit_position):
    """Flip a specific bit in the binary representation of a value"""
    if isinstance(value, int):
        return value ^ (1 << bit_position)
    elif isinstance(value, float):
        import struct
        ieee = struct.pack('>f', value)
        i = struct.unpack('>I', ieee)[0]
        i ^= (1 << bit_position)
        return struct.unpack('>f', struct.pack('>I', i))[0]

# Test error correction capabilities
print("Testing TMR protection...")

# Protect data operations in radiation environments
for _ in range(10):
    # Your data operations here
    result = protected_int.value * 2

    # Simulate random radiation effects
    if random.random() < 0.3:  # 30% chance of radiation effect
        bit = random.randint(0, 31)
        corrupted_value = simulate_bit_flip(protected_int.value, bit)
        # In a real scenario, radiation would directly affect memory
        # This is just for demonstration
        protected_int._v2 = corrupted_value

        print(f"Radiation effect simulated, bit {bit} flipped")

        # Verify integrity and correct if needed
        if not protected_int.check_integrity():
            print("Corruption detected!")
            if protected_int.correct():
                print("Error successfully corrected")
            else:
                print("Error correction failed")

# Shutdown
rad_ml_minimal.shutdown()
```

## Using Python with C++ Integration Points

For projects using both Python and C++:

```python
import rad_ml_minimal as rad_ml

# Initialize with specific environment settings
rad_ml.initialize(radiation_environment=rad_ml.RadiationEnvironment.MARS)

# Create protected data structures
# ... your code here ...

# At the language boundary (Python to C++), use the serialization utilities
# to maintain protection across the boundary
serialized_data = rad_ml.serialize_protected_data(your_protected_data)

# Pass serialized_data to C++ components
# ...

# Then in C++:
// auto protected_data = rad_ml::deserialize_protected_data(serialized_data);
// Use protected data in C++ with full radiation protection...

# Finally, shutdown properly
rad_ml.shutdown()
```
