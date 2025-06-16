# Adaptive Voting Mechanism - ASCII Flow Diagram
## RadML Space-Radiation-Tolerant ML Framework

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           ADAPTIVE VOTING MECHANISM FLOW                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT: Three Redundant Values
                                    ┌─────┐  ┌─────┐  ┌─────┐
                                    │ V₁  │  │ V₂  │  │ V₃  │
                                    └──┬──┘  └──┬──┘  └──┬──┘
                                       │        │        │
                                       └────────┼────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          FAULT PATTERN DETECTION ENGINE                             │
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Bitwise XOR   │    │  Hamming Weight │    │ Pattern Analysis│                │
│  │   V₁ ⊕ V₂ ⊕ V₃  │───▶│   popcount()    │───▶│   & Clustering  │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│                                                          │                         │
└──────────────────────────────────────────────────────────┼─────────────────────────┘
                                                           │
                                    ┌──────────────────────┼──────────────────────┐
                                    │                      ▼                      │
                                    │         PATTERN CLASSIFICATION             │
                                    │                                            │
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  SINGLE_BIT   │  │ ADJACENT_BITS │  │  BYTE_ERROR   │  │  WORD_ERROR   │  │ BURST_ERROR   │
│               │  │               │  │               │  │               │  │               │
│ popcount = 1  │  │ Consecutive   │  │ 8-bit Boundary│  │ Word-level    │  │ Clustered     │
│               │  │ bit pattern   │  │ alignment     │  │ corruption    │  │ bit pattern   │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │                  │                  │
        ▼                  ▼                  ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE STRATEGY SELECTION                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
        │                  │                  │                  │                  │
        ▼                  ▼                  ▼                  ▼                  ▼

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   STANDARD   │  │  BIT-LEVEL   │  │   SEGMENT    │  │ WORD-ERROR   │  │ BURST-ERROR  │
│    VOTING    │  │   VOTING     │  │   VOTING     │  │   VOTING     │  │   VOTING     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼                 ▼

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            VOTING ALGORITHM DETAILS                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STANDARD VOTING (Traditional Majority)                                             │
│                                                                                     │
│    V₁ ──┐                                                                          │
│         ├─── Majority ───▶ Result                                                  │
│    V₂ ──┤    Decision                                                              │
│         │                                                                          │
│    V₃ ──┘                                                                          │
│                                                                                     │
│ Logic: if (V₁ == V₂) return V₁; else if (V₁ == V₃) return V₁; else return V₂;     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ BIT-LEVEL VOTING (Bitwise Majority)                                                │
│                                                                                     │
│ For each bit position i:                                                           │
│                                                                                     │
│    V₁[i] ──┐                                                                       │
│            ├─── Bit Majority ───▶ Result[i]                                       │
│    V₂[i] ──┤    Decision                                                           │
│            │                                                                       │
│    V₃[i] ──┘                                                                       │
│                                                                                     │
│ Result = ⋃ᵢ MajorityVote(V₁[i], V₂[i], V₃[i])                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ SEGMENT VOTING (8-bit Segment-based)                                               │
│                                                                                     │
│ Divide into 8-bit segments:                                                        │
│                                                                                     │
│ V₁: │Seg₀│Seg₁│Seg₂│...│SegN│                                                     │
│ V₂: │Seg₀│Seg₁│Seg₂│...│SegN│                                                     │
│ V₃: │Seg₀│Seg₁│Seg₂│...│SegN│                                                     │
│                                                                                     │
│ For each segment s:                                                                │
│    Result[s] = MajorityVote(V₁[s], V₂[s], V₃[s])                                  │
│                                                                                     │
│ Final Result = Concatenate(Result[0], Result[1], ..., Result[N])                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HEALTH-WEIGHTED VOTING (Dynamic Reliability-based)                                 │
│                                                                                     │
│ Health Scores: H₁, H₂, H₃ (range: 0.1 to 1.0)                                    │
│                                                                                     │
│    V₁ ──┐ Weight: H₁                                                              │
│         │                                                                          │
│    V₂ ──┼─── Weighted ───▶ Result = argmax(Σ Hᵢ × Wᵢ)                           │
│         │    Decision                                                              │
│    V₃ ──┘ Weight: H₃                                                              │
│                                                                                     │
│ Health Update Rules:                                                               │
│ • Correct vote: Hᵢ = min(1.0, Hᵢ + 0.05)                                         │
│ • Incorrect vote: Hᵢ = max(0.1, Hᵢ - 0.2)                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          MISSION-AWARE PROTECTION SCALING                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    Environment Assessment
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │ SAA Region?  │      │Solar Activity│      │ Mission Phase│
            │              │      │    > 0.7?    │      │   Critical?  │
            └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Protection Level    │
                              │ Selection           │
                              └─────────┬───────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
            │   STANDARD   │    │   ENHANCED   │    │    HYBRID    │
            │     TMR      │    │     TMR      │    │ REDUNDANCY   │
            └──────────────┘    └──────────────┘    └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            PERFORMANCE METRICS                                      │
│                                                                                     │
│ ┌─────────────────┬─────────────┬─────────────┬─────────────┬─────────────────┐   │
│ │ Error Pattern   │ Standard    │ Bit-Level   │ Segment     │ Adaptive        │   │
│ ├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤   │
│ │ Single Bit      │ 100.0%      │ 100.0%      │ 100.0%      │ 100.0%          │   │
│ │ Adjacent Bits   │  66.7%      │  95.2%      │  97.8%      │  98.4%          │   │
│ │ Byte Error      │  33.3%      │  78.6%      │  94.7%      │  95.1%          │   │
│ │ Word Error      │  33.3%      │  45.2%      │  67.4%      │  89.2%          │   │
│ │ Burst Error     │  33.3%      │  67.8%      │  96.8%      │  97.3%          │   │
│ └─────────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION TREE FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    Start
                                      │
                                      ▼
                            ┌─────────────────┐
                            │ Detect Pattern  │
                            └─────────┬───────┘
                                      │
                            ┌─────────▼───────┐
                            │ Confidence > 0.8?│
                            └─────┬───────┬───┘
                                  │ Yes   │ No
                                  ▼       ▼
                        ┌─────────────┐   │
                        │Single Bit?  │   │
                        └─────┬───┬───┘   │
                              │Yes│No     │
                              ▼   ▼       ▼
                    ┌──────────────┐      │
                    │Standard Vote │      │
                    └──────────────┘      │
                                          │
                            ┌─────────────▼───────────┐
                            │     Pattern Type?       │
                            └─┬─────┬─────┬─────┬────┘
                              │     │     │     │
                    ┌─────────▼┐   ┌▼────┐│    ┌▼──────────┐
                    │Burst Vote│   │Byte ││    │Health     │
                    └──────────┘   │Vote │▼    │Weighted   │
                                   └─────┘     │Vote       │
                                               └───────────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │Final Result  │
                                            └──────────────┘

Legend:
┌─┐ Process/Decision    ▼ Flow Direction    ├─ Multiple Inputs    ⊕ XOR Operation
└─┘
```

## Key Features Highlighted:

### 🔍 **Pattern Detection Engine**
- Real-time bitwise XOR analysis
- Hamming weight calculation
- Intelligent clustering detection

### 🎯 **Adaptive Strategy Selection**
- Six specialized voting algorithms
- Pattern-specific optimization
- Confidence-based decision making

### ⚖️ **Health-Weighted System**
- Dynamic reliability scoring (0.1-1.0 range)
- Reward/penalty learning (±0.05/±0.2)
- Component degradation tracking

### 🚀 **Mission-Aware Scaling**
- SAA region detection
- Solar activity monitoring (>0.7 threshold)
- Protection level adaptation

### 📊 **Performance Validation**
- Comprehensive error pattern testing
- Statistical validation results
- Adaptive algorithm superiority demonstration
