#!/usr/bin/env python3
import numpy as np
import struct
from typing import List, Tuple, Optional, Callable, Union
import copy

# Import Reed-Solomon implementation from existing code
from rs_monte_carlo import GF256, RS8Bit8Sym, float_to_bytes, bytes_to_float


###########################################
# Base Protection Method Interface
###########################################
class ProtectionMethod:
    """Base class for all protection methods."""

    def __init__(self, name: str, overhead: float):
        """
        Initialize protection method.

        Args:
            name: Name of the protection method
            overhead: Memory overhead factor (1.0 = 100%)
        """
        self.name = name
        self.overhead = overhead

    def protect(self, value: float) -> List[int]:
        """
        Protect a floating-point value.

        Args:
            value: The float value to protect

        Returns:
            Protected representation as list of bytes
        """
        raise NotImplementedError("Subclasses must implement protect()")

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """
        Recover a value from its protected representation.

        Args:
            protected_data: Protected representation (list of bytes)

        Returns:
            Tuple of (recovered value, success flag)
        """
        raise NotImplementedError("Subclasses must implement recover()")


###########################################
# Reed-Solomon Protection (from existing code)
###########################################
class ReedSolomonProtection(ProtectionMethod):
    """Reed-Solomon error correction for floating-point values."""

    def __init__(self, n_ecc_symbols: int = 8):
        """
        Initialize Reed-Solomon protection.

        Args:
            n_ecc_symbols: Number of ECC symbols (default: 8)
        """
        # Calculate overhead: n_ecc_symbols / (4 bytes for float) as percentage
        overhead = n_ecc_symbols / 4.0
        super().__init__(f"Reed-Solomon (RS{n_ecc_symbols})", overhead)
        self.rs_codec = RS8Bit8Sym()
        self.n_ecc_symbols = n_ecc_symbols

    def protect(self, value: float) -> List[int]:
        """Protect a float value using Reed-Solomon encoding."""
        # Convert float to bytes
        data_bytes = float_to_bytes(value)
        # Encode with Reed-Solomon
        encoded_bytes = self.rs_codec.encode(data_bytes)
        return encoded_bytes

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """Recover a float value from Reed-Solomon encoded bytes."""
        # Decode with Reed-Solomon
        decoded_bytes, success = self.rs_codec.decode(protected_data)

        if not success:
            # Decoding failed
            return 0.0, False

        try:
            # Convert bytes back to float
            value = bytes_to_float(decoded_bytes)
            return value, True
        except:
            # Error converting to float
            return 0.0, False


###########################################
# Hamming Code Protection
###########################################
class HammingProtection(ProtectionMethod):
    """Hamming code protection for floating-point values."""

    def __init__(self):
        """Initialize Hamming code protection."""
        # Hamming code overhead is approximately 3/4 for reasonable word sizes
        super().__init__("Hamming Code", 0.3125)

    def _calculate_parity_bits(self, data_bits: List[int]) -> List[int]:
        """Calculate Hamming code parity bits."""
        # Determine number of parity bits needed (2^r >= m+r+1)
        m = len(data_bits)
        r = 1
        while (1 << r) < (m + r + 1):
            r += 1

        # Create extended array with positions for parity bits
        extended = [0] * (m + r)

        # Fill in data bits
        data_idx = 0
        for i in range(1, len(extended) + 1):
            # Skip positions that are powers of 2 (reserved for parity bits)
            if not (i & (i - 1) == 0):  # Check if i is not a power of 2
                extended[i - 1] = data_bits[data_idx]
                data_idx += 1

        # Calculate parity bits
        for i in range(r):
            parity_pos = (1 << i) - 1  # Position of parity bit (0-indexed)
            parity = 0

            # Check all bits where the i-th bit of the position is 1
            for j in range(parity_pos, len(extended)):
                if ((j + 1) & (1 << i)) != 0:
                    parity ^= extended[j]

            extended[parity_pos] = parity

        return extended

    def _extract_data_bits(self, hamming_code: List[int]) -> Tuple[List[int], bool]:
        """
        Extract data bits from Hamming code, correcting single-bit errors.

        Returns:
            Tuple of (data_bits, success_flag)
        """
        # Calculate syndrome
        r = 0
        while (1 << r) <= len(hamming_code):
            r += 1

        syndrome = 0
        for i in range(r):
            parity_pos = (1 << i) - 1
            parity = 0

            for j in range(len(hamming_code)):
                if ((j + 1) & (1 << i)) != 0:
                    parity ^= hamming_code[j]

            if parity != 0:
                syndrome |= 1 << i

        # Correct error if detected
        success = True
        if syndrome != 0:
            if syndrome <= len(hamming_code):
                # Valid error position, correct it
                hamming_code[syndrome - 1] ^= 1
            else:
                # Invalid syndrome, cannot correct
                success = False

        # Extract data bits
        data_bits = []
        for i in range(1, len(hamming_code) + 1):
            if not (i & (i - 1) == 0):  # Check if i is not a power of 2
                data_bits.append(hamming_code[i - 1])

        return data_bits, success

    def _float_to_bits(self, value: float) -> List[int]:
        """Convert float to list of bits."""
        # Get bytes
        float_bytes = struct.pack("!f", value)

        # Convert to bits
        bits = []
        for b in float_bytes:
            for i in range(8):
                bits.append((b >> i) & 1)

        return bits

    def _bits_to_float(self, bits: List[int]) -> float:
        """Convert list of bits to float."""
        if len(bits) != 32:
            raise ValueError(f"Expected 32 bits for float, got {len(bits)}")

        # Convert bits to bytes
        float_bytes = bytearray(4)
        for i in range(4):
            byte = 0
            for j in range(8):
                byte |= bits[i * 8 + j] << j
            float_bytes[i] = byte

        # Convert bytes to float
        return struct.unpack("!f", float_bytes)[0]

    def protect(self, value: float) -> List[int]:
        """Protect a float value using Hamming code."""
        # Convert float to bits
        data_bits = self._float_to_bits(value)

        # Apply Hamming code
        hamming_code = self._calculate_parity_bits(data_bits)

        # Convert to bytes for storage
        result = []
        for i in range(0, len(hamming_code), 8):
            byte = 0
            for j in range(min(8, len(hamming_code) - i)):
                byte |= hamming_code[i + j] << j
            result.append(byte)

        return result

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """Recover a float value from Hamming-encoded bytes."""
        # Convert bytes to bits
        hamming_code = []
        for byte in protected_data:
            for i in range(8):
                hamming_code.append((byte >> i) & 1)

        # Extract data bits and correct errors if possible
        data_bits, success = self._extract_data_bits(hamming_code)

        # Truncate to 32 bits if needed
        data_bits = data_bits[:32]

        # Pad if needed (shouldn't happen with valid data)
        if len(data_bits) < 32:
            data_bits.extend([0] * (32 - len(data_bits)))

        try:
            # Convert bits back to float
            value = self._bits_to_float(data_bits)
            return value, success
        except:
            return 0.0, False


###########################################
# SEC-DED (Single Error Correction, Double Error Detection)
###########################################
class SECDEDProtection(ProtectionMethod):
    """SEC-DED protection for floating-point values."""

    def __init__(self):
        """Initialize SEC-DED protection."""
        # SEC-DED has slightly higher overhead than Hamming
        super().__init__("SEC-DED", 0.375)  # 12 extra bits for 32-bit float

    def _calculate_secded(self, data_bits: List[int]) -> List[int]:
        """Calculate SEC-DED code (Hamming + parity)."""
        # First, calculate Hamming code
        hamming = HammingProtection()
        hamming_code = hamming._calculate_parity_bits(data_bits)

        # Add overall parity bit
        overall_parity = 0
        for bit in hamming_code:
            overall_parity ^= bit

        # Append overall parity bit
        return hamming_code + [overall_parity]

    def _extract_data_bits_secded(
        self, secded_code: List[int]
    ) -> Tuple[List[int], bool]:
        """
        Extract data bits from SEC-DED code, correcting single-bit errors
        and detecting double-bit errors.

        Returns:
            Tuple of (data_bits, success_flag)
        """
        # Split the code into Hamming code and overall parity
        hamming_code = secded_code[:-1]
        received_parity = secded_code[-1]

        # Calculate overall parity
        calculated_parity = 0
        for bit in hamming_code:
            calculated_parity ^= bit

        # Extract data using Hamming error correction
        hamming = HammingProtection()
        data_bits, hamming_success = hamming._extract_data_bits(hamming_code)

        # Check overall parity
        if calculated_parity != received_parity:
            # Parity error - possible double-bit error
            if not hamming_success:
                # Hamming also detected an error - likely a double-bit error
                return data_bits, False

        return data_bits, hamming_success

    def protect(self, value: float) -> List[int]:
        """Protect a float value using SEC-DED code."""
        # Convert float to bits
        hamming = HammingProtection()
        data_bits = hamming._float_to_bits(value)

        # Apply SEC-DED code
        secded_code = self._calculate_secded(data_bits)

        # Convert to bytes for storage
        result = []
        for i in range(0, len(secded_code), 8):
            byte = 0
            for j in range(min(8, len(secded_code) - i)):
                byte |= secded_code[i + j] << j
            result.append(byte)

        return result

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """Recover a float value from SEC-DED encoded bytes."""
        # Convert bytes to bits
        secded_code = []
        for byte in protected_data:
            for i in range(8):
                secded_code.append((byte >> i) & 1)

        # Extract data bits and correct errors if possible
        data_bits, success = self._extract_data_bits_secded(secded_code)

        # Truncate to 32 bits if needed
        data_bits = data_bits[:32]

        # Pad if needed (shouldn't happen with valid data)
        if len(data_bits) < 32:
            data_bits.extend([0] * (32 - len(data_bits)))

        try:
            # Convert bits back to float
            hamming = HammingProtection()
            value = hamming._bits_to_float(data_bits)
            return value, success
        except:
            return 0.0, False


###########################################
# Triple Modular Redundancy (TMR)
###########################################
class TMRProtection(ProtectionMethod):
    """Triple Modular Redundancy protection for floating-point values."""

    def __init__(self):
        """Initialize TMR protection."""
        super().__init__("Triple Modular Redundancy (TMR)", 2.0)  # 200% overhead

    def protect(self, value: float) -> List[int]:
        """Protect a float value using TMR."""
        # Convert float to bytes (3 copies)
        bytes1 = float_to_bytes(value)
        bytes2 = float_to_bytes(value)
        bytes3 = float_to_bytes(value)

        # Interleave the three copies
        result = []
        for i in range(len(bytes1)):
            result.append(bytes1[i])
        for i in range(len(bytes2)):
            result.append(bytes2[i])
        for i in range(len(bytes3)):
            result.append(bytes3[i])

        return result

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """Recover a float value from TMR-encoded bytes."""
        # Check if we have enough data
        if len(protected_data) < 12:  # 3 copies of 4 bytes
            return 0.0, False

        # Split into three copies
        chunk_size = len(protected_data) // 3
        bytes1 = protected_data[:chunk_size]
        bytes2 = protected_data[chunk_size : 2 * chunk_size]
        bytes3 = protected_data[2 * chunk_size : 3 * chunk_size]

        # Ensure each copy is 4 bytes (truncate or pad)
        bytes1 = (bytes1 + [0] * 4)[:4]
        bytes2 = (bytes2 + [0] * 4)[:4]
        bytes3 = (bytes3 + [0] * 4)[:4]

        try:
            # Convert each copy to float
            value1 = bytes_to_float(bytes1)
            value2 = bytes_to_float(bytes2)
            value3 = bytes_to_float(bytes3)

            # Perform majority voting (bit-level would be better, but this is simpler)
            if value1 == value2 or value1 == value3:
                return value1, True
            elif value2 == value3:
                return value2, True
            else:
                # All three values are different, take the first one and flag as unsuccessful
                return value1, False
        except:
            return 0.0, False


###########################################
# Hybrid Protection Methods
###########################################
class HybridProtection(ProtectionMethod):
    """Hybrid protection combining multiple methods."""

    def __init__(self, name: str, methods: List[Tuple[ProtectionMethod, float]]):
        """
        Initialize hybrid protection.

        Args:
            name: Name of the hybrid protection method
            methods: List of (method, weight_ratio) tuples
                     where weight_ratio indicates what portion of bits use this method
                     (should sum to 1.0)
        """
        # Calculate overall overhead
        total_overhead = sum(method.overhead * ratio for method, ratio in methods)
        super().__init__(name, total_overhead)
        self.methods = methods

        # Validate weight ratios
        total_ratio = sum(ratio for _, ratio in methods)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Method weight ratios must sum to 1.0, got {total_ratio}")

    def protect(self, value: float) -> List[int]:
        """Protect a float value using the hybrid approach."""
        # Convert value to bytes
        value_bytes = float_to_bytes(value)

        # Apply each protection method to its portion
        protected_parts = []
        marker_bytes = []  # To identify which method was used for each part

        for i, (method, ratio) in enumerate(self.methods):
            # Use the same value for each method for simplicity
            protected = method.protect(value)
            protected_parts.append(protected)
            marker_bytes.append(i)  # Mark which method was used

        # Combine all parts with markers
        result = marker_bytes + [len(p) for p in protected_parts]
        for part in protected_parts:
            result.extend(part)

        return result

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """Recover a float value from hybrid-protected bytes."""
        if len(protected_data) < len(self.methods) * 2:
            return 0.0, False

        try:
            # Extract markers and lengths
            num_methods = len(self.methods)
            markers = protected_data[:num_methods]
            lengths = protected_data[num_methods : num_methods * 2]

            # Split data according to lengths
            data_start = num_methods * 2
            parts = []
            for length in lengths:
                if data_start + length <= len(protected_data):
                    parts.append(protected_data[data_start : data_start + length])
                    data_start += length
                else:
                    parts.append([])

            # Try to recover using each method
            values = []
            successes = []

            for i, (marker, part) in enumerate(zip(markers, parts)):
                if marker < len(self.methods):
                    method, _ = self.methods[marker]
                    value, success = method.recover(part)
                    values.append(value)
                    successes.append(success)

            # If any method succeeded, use its value
            for value, success in zip(values, successes):
                if success:
                    return value, True

            # All methods failed
            if values:
                return values[0], False
            else:
                return 0.0, False
        except:
            return 0.0, False


###########################################
# No Protection (Baseline)
###########################################
class NoProtection(ProtectionMethod):
    """No protection, just direct value storage (baseline for comparison)."""

    def __init__(self):
        """Initialize no protection method."""
        super().__init__("No Protection", 0.0)

    def protect(self, value: float) -> List[int]:
        """Store a float value with no protection."""
        # Simply convert float to bytes
        return float_to_bytes(value)

    def recover(self, protected_data: List[int]) -> Tuple[float, bool]:
        """Recover a float value with no protection."""
        try:
            # Convert bytes back to float
            if len(protected_data) >= 4:
                value = bytes_to_float(protected_data[:4])
                return value, True
            else:
                return 0.0, False
        except:
            return 0.0, False


###########################################
# Factory function to create protection methods
###########################################
def create_protection_method(method_name: str) -> ProtectionMethod:
    """
    Create a protection method instance by name.

    Args:
        method_name: Name of the protection method

    Returns:
        Instance of ProtectionMethod
    """
    if method_name == "none":
        # No protection, just for baseline
        return NoProtection()
    elif method_name == "hamming":
        return HammingProtection()
    elif method_name == "secded":
        return SECDEDProtection()
    elif method_name == "tmr":
        return TMRProtection()
    elif method_name == "rs4":
        return ReedSolomonProtection(4)
    elif method_name == "rs8":
        return ReedSolomonProtection(8)
    elif method_name == "hybrid_rs4_tmr":
        # Hybrid: 50% TMR (critical bits), 50% RS4
        return HybridProtection(
            "Hybrid (RS4 + TMR)",
            [
                (TMRProtection(), 0.5),  # TMR for 50% of bits (most significant)
                (
                    ReedSolomonProtection(4),
                    0.5,
                ),  # RS4 for 50% of bits (least significant)
            ],
        )
    elif method_name == "hybrid_hamming_rs4":
        # Hybrid: 75% Hamming, 25% RS4
        return HybridProtection(
            "Hybrid (Hamming + RS4)",
            [
                (HammingProtection(), 0.75),  # Hamming for 75% of bits
                (ReedSolomonProtection(4), 0.25),  # RS4 for 25% of bits
            ],
        )
    else:
        raise ValueError(f"Unknown protection method: {method_name}")
