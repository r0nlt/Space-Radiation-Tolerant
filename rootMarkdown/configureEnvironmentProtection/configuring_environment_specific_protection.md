# Configuring Environment-Specific Protection

```cpp
// Configure for LEO (Low Earth Orbit) environment
sim::RadiationEnvironment leo = sim::createEnvironment(sim::Environment::LEO);
protection.updateEnvironment(leo);

// Perform protected operations in LEO environment
// ...

// Configure for SAA crossing (South Atlantic Anomaly)
sim::RadiationEnvironment saa = sim::createEnvironment(sim::Environment::SAA);
protection.updateEnvironment(saa);
protection.enterMissionPhase(MissionPhase::SAA_CROSSING);

// Perform protected operations with enhanced protection for SAA
// ...
```
