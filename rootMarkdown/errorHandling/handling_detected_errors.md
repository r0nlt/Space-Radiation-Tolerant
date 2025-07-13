# Handling Detected Errors

```cpp
// Execute with error detection
auto result = protection.executeProtected<float>([&]() {
    return performComputation();
});

// Check if errors were detected and corrected
if (result.error_detected) {
    if (result.error_corrected) {
        logger.info("Error detected and corrected");
    } else {
        logger.warning("Error detected but could not be corrected");
        fallbackStrategy();
    }
}
```
