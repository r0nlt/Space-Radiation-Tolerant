# 🧭 Quality-Diversity (MAP-Elites + Novelty) in Auto Architecture Search

## Quick Navigation
- [What QD Is (In This Framework)](#what-qd-is-in-this-framework)
- [Where It Lives in the Code](#where-it-lives-in-the-code)
- [How It Integrates with the Evolutionary Algorithm](#how-it-integrates-with-the-evolutionary-algorithm)
- [Why QD Helps Auto-Architecture Search](#why-qd-helps-auto-architecture-search)
- [Practical Guidance](#practical-guidance)
- [Metrics and Logs](#metrics-and-logs)
- [Deep Dive](#deep-dive-how-the-advanced-qd-algorithm-works)
- [Spatial Map & 2D Projection](#spatial-map-and-2d-map-elites-projection)
- [QD Laboratory Report](#qd-laboratory-report-structured)

## What QD Is (In This Framework)
- **Goal**: Maintain a diverse set of high-quality architectures instead of converging to a single peak.
- **Mechanism**: A MAP-Elites archive keyed by a physics-informed behavior descriptor; each cell stores the best (elite) config discovered for that niche.
- **Exploration Boost**: Novelty score computed via k-nearest neighbors (k=5) in behavior space guides exploration to underrepresented niches.

Behavior descriptor (6D, radiation-aware):
- Complexity, Protection efficiency, Computational cost, Radiation tolerance, Graceful degradation, Power efficiency.

## Where It Lives in the Code
- Enable via example app CLI:
  ```bash
  ./examples/auto_arch_search_example --qd --adv-qd --trials 5
  ```
- Programmatic toggle:
  ```cpp
  searcher.enableAdvancedQualityDiversity(true);
  ```
- Implementation references:
  - [`include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp`](../../include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp)
  - [`include/rad_ml/research/auto_arch/quality_diversity.hpp`](../../include/rad_ml/research/auto_arch/quality_diversity.hpp)
  - [`include/rad_ml/research/auto_arch_search.hpp`](../../include/rad_ml/research/auto_arch_search.hpp) (QD toggles)
  - [`examples/auto_arch_search_example.cpp`](../../examples/auto_arch_search_example.cpp) (CLI wiring)

## How It Integrates with the Evolutionary Algorithm
- Standard GA loop (selection, crossover, mutation, elitism) runs as usual.
- After evaluation of each individual:
  - Compute behavior descriptor and novelty score.
  - Update archive cell if combined fitness improves the elite.
- At generation boundaries:
  - Sample diverse elites from archive (mixed by fitness, novelty, and uniform niches).
  - Replace worst K individuals to inject diversity and quality.

Combined fitness used for archive updates:
```cpp
// 0.8 × preservation + 0.2 × novelty × 100
// See: advanced_quality_diversity.hpp::calculateCombinedFitness
```

## Why QD Helps Auto-Architecture Search
- Avoids premature convergence by preserving multiple strong-but-different designs.
- Encourages discovering robust architectures across protection strategies (e.g., `FULL_TMR`, `ADAPTIVE_TMR`, `SPACE_OPTIMIZED`).
- Improves search coverage of the architecture-protection-performance trade space under radiation.

## Practical Guidance
- Start with smaller grids for short runs to get visible coverage:
  - 6D × 5 bins early; scale to ×10 for longer runs.
- Novelty weight:
  - Begin around 0.2; increase if exploration stalls.
- Elite injection rate:
  - Replace ~20% of population each generation for balance.

## Metrics and Logs
- Per-generation QD logs:
  - Coverage (%), occupied cells, count of elites injected into population.
- CSV outputs include preservation and protection distributions; expect early small % coverage with large grids (1e6 cells at 6D × 10).

## Relationship to Adaptive Mutation and Operator Credit
- QD complements adaptive mutation by keeping niches diverse while mutation rate adapts to diversity/plateau signals.
- Operator-credit learning still applies; QD influences which niches are exploited, not how operators are credited.

## Quick Checklist
- Enable `--qd --adv-qd` for MAP-Elites + Novelty.
- Monitor occupied cells and elites injected; do not over-interpret % coverage in short runs.
- Tune novelty weight and grid resolution based on diversity and progress.

---

## Deep Dive: How the Advanced QD Algorithm Works

### Archive Data Structures
- `ArchiveCell` fields:
  - `best_config`, `fitness_score`, `behavior`, `generation_discovered`, `novelty_score`, `objective_vector`.
  - Empty cells use `-inf` fitness for quick checks (`isEmpty`).
- Grid parameters:
  - `BEHAVIORAL_DIMENSIONS = 6`, `BASE_GRID_RESOLUTION = 10` (default), `MAX_GRID_RESOLUTION = 50`.
  - Total cells = `resolution^6` (e.g., 10^6 = 1,000,000).
- Novelty store:
  - `novelty_archive_` keeps recent behavior descriptors (capped at 1000) for KNN.

### Behavior Descriptor Calculation
Source: `calculateRadiationAwareBehavior(config, result)`
- Architectural complexity: `log(total_params+1)/log(1e6)` (∈ [0,1] approx)
- Protection efficiency: `errors_corrected / max(1, errors_detected)` (1.0 if none detected)
- Computational cost: min(1, `(execution_time_ms/1000) * (1 + architectural_complexity)`)
- Radiation tolerance: `accuracy_preservation / 100`
- Graceful degradation: `1 - clamp((baseline - radiation)/baseline, 0, 1)`
- Power efficiency: `1 / (1 + protection_overhead + architectural_complexity)` where overhead depends on `protection_level`.

### Discretization and Indexing
- Clamp each descriptor to [0,1], map to `coord = floor(value * (resolution-1))`.
- `coordsToIndex` flattens 6D coords into a single index using base-`resolution` multipliers.
- Implications:
  - With large grids/short runs, expect tiny % coverage; track occupied cell count and elite injections.

### Novelty Calculation (KNN)
Source: `calculateNoveltyScore(behavior)`
- Distance: Euclidean in 6D descriptor space.
- If archive size < K (=5), return 1.0 to bootstrap exploration.
- Otherwise, `partial_sort` k smallest distances and return their mean.
- Novelty archive maintenance: append per accepted elite; cap at 1000 by evicting oldest.

### Combined Fitness and Archive Update Policy
- Combined fitness: `0.8 * preservation + 0.2 * novelty * 100`.
- `addToArchive(config, result, generation)`:
  - Compute descriptor and novelty.
  - Discretize → cell index; lock archive; if empty or fitness improved, replace elite and log metadata.
  - Update novelty archive and return `true` if cell improved.

### Batched Evaluation Support
Source: `evaluatePopulationBatch(population, generation, evaluator)`
- Asynchronously evaluates batches using `std::async`, calls `evaluateAndArchive` per config.
- Updates coverage stats once batch completes.

### Elite Sampling for GA Injection
Source: `sampleDiverseElites(sample_size)`
- Mix of:
  - 40% highest combined fitness elites
  - 30% highest novelty elites
  - 30% uniformly random occupied cells
- Returned configs replace worst-K individuals in GA to maintain quality and diversity.

### Archive Analytics
Source: `getAnalytics()`
- Reports: `total_occupied_cells`, `coverage_percentage`, `average_fitness`, `fitness_variance`, and `behavioral_diversity` (mean pairwise distance).
- Coverage is also maintained incrementally via `updateArchiveStatistics()`.

### Generation-Level Pseudocode
```cpp
// Evaluate and archive
for (const auto& config : population) {
  auto result = testConfiguration(config, mc_trials);
  qd.addToArchive(config, result, gen);
}

// GA core
select_parents();
produce_offspring_via_crossover_and_mutation();
apply_elitism();

// Inject QD elites
auto elites = qd.sampleDiverseElites(replacement_k);
replace_worst_k(population, elites);
```

### Tuning Knobs and Trade-offs
- Grid resolution:
  - Short runs: 6D × 5; longer runs: 6D × 10; only increase beyond if occupancy growth continues.
- Novelty K and archive size:
  - Keep K=5; increase only if behaviors cluster too tightly; cap size ~1000 for stable statistics.
- Fitness weights:
  - Start 0.8 preservation / 0.2 novelty; raise novelty if exploration stalls, lower if quality regresses.
- Elite injection rate:
  - ~20% of population per generation; reduce if quality oscillates; increase if diversity collapses.

### Diagnostics to Monitor
- QD coverage: occupied cells and trend per generation (prefer absolute over % early).
- Elites injected per generation: should be >0 most generations once archive warms up.
- Novelty trend: mean/variance over accepted elites; falling novelty may indicate niche saturation.
- Diversity by protection strategy: distribution across `NONE, CHECKSUM_ONLY, SELECTIVE_TMR, FULL_TMR, ADAPTIVE_TMR, SPACE_OPTIMIZED`.
- Behavioral diversity (mean pairwise distance) from `getAnalytics()`.

### Common Pitfalls
- Descriptor scaling: ensure execution time and parameter counts keep values within [0,1] after transforms.
- Overly fine grids: stalls coverage; use coarser grids until runs are longer.
- Novelty starvation: if novelty archive is too small or too homogenous, KNN becomes uninformative; allow archive to grow before judging.
- Objective vector: stored for future multi-objective analysis; current combined fitness remains scalar.

See also: `FAQ/GENETICS/Genetic_Algorithm_Architecture.md` for the base GA and adaptive mutation details.

## Spatial Map and 2D MAP-Elites Projection

### Intuition: 6D Space → Discrete Grid
- The behavior descriptor is 6D; each dimension is normalized to [0,1].
- We discretize each dimension into R bins (default R=10), yielding R^6 cells.
- Each evaluated configuration lands in exactly one cell.

Discretization (same as implementation):
```cpp
size_t to_coord(double v, size_t R) {
  double clamped = std::max(0.0, std::min(1.0, v));
  return static_cast<size_t>(clamped * (R - 1));
}
```

### 2D Projection Example (Radiation tolerance vs Computational cost)
We often visualize a 2D slice or projection to explain coverage. Below, rows are radiation tolerance (low→high), columns are computational cost (low→high). "●" = occupied cell (elite exists), "+" = empty cell.

```
Radiation tol ↑

[9]  ●  +  +  ●  +  +  +  ●  +  +
[8]  +  +  +  +  +  +  +  +  +  +
[7]  +  ●  +  +  +  +  +  +  +  +
[6]  +  +  +  +  ●  +  +  +  +  +
[5]  +  +  +  +  +  +  +  +  ●  +
[4]  +  +  +  +  +  +  +  +  +  +
[3]  +  +  ●  +  +  +  +  +  +  +
[2]  +  +  +  +  +  +  +  +  +  +
[1]  +  +  +  +  +  +  +  +  +  +
[0]  +  +  +  +  +  +  +  +  +  +
      0  1  2  3  4  5  6  7  8  9  → Computational cost
```

- A point with `(radiation_tolerance=0.71, computational_cost=0.32)` goes to coords `(7, 3)` for `R=10`.
- The archive actually tracks all 6 dimensions, but this projection helps interpret progress.

### Worked Step-Through (Single Sample)
Given a test result:
- `accuracy_preservation = 92.4` → radiation_tolerance = 0.924
- `baseline=85.0`, `radiation=80.3` → graceful_degradation = `1 - (85-80.3)/85 ≈ 0.945`
- `errors_detected=120`, `errors_corrected=100` → protection_efficiency ≈ 0.833
- `execution_time_ms=210` and moderate complexity (say 0.3) → computational_cost ≈ `min(1, 0.21 * 1.3) ≈ 0.273`
- Assume power_efficiency computed from protection overhead and complexity.

Discretize (R=10):
- `coord = floor(value * 9)`, so `radiation_tolerance=0.924 → 8`, `computational_cost=0.273 → 2`, etc.
- Compute novelty via KNN in 6D and combined fitness `0.8*92.4 + 0.2*novelty*100`.
- If better than current elite in that 6D cell, replace.

### Minimal Visualization Script (Optional)
A tiny snippet to visualize a 2D projection from sampled behaviors:
```python
import numpy as np
import matplotlib.pyplot as plt

R = 10
# Example occupied coordinates (row=rad_tol, col=comp_cost)
occupied = [(9,0),(9,3),(9,7),(7,1),(6,4),(5,8),(3,2)]
A = np.zeros((R, R))
for r,c in occupied:
    A[r, c] = 1

plt.imshow(A, origin='lower', cmap='Greens', extent=[0,R-1,0,R-1])
plt.colorbar(label='Occupied')
plt.xlabel('Computational cost (bin)')
plt.ylabel('Radiation tolerance (bin)')
plt.title('2D MAP-Elites Projection (example)')
plt.grid(False)
plt.show()
```

### How Elite Injection Looks on the Map
- Each generation, a subset of elites are sampled from the archive: 40% by highest fitness, 30% by highest novelty, 30% random occupied cells.
- These sampled elites are injected into the GA by replacing the worst K individuals, pushing the population toward new and diverse niches that the map highlights.

### Tips for 2D Slices
- Choose dimensions with intuitive trade-offs (e.g., tolerance vs cost, tolerance vs power) for presentations.
- Keep R small (e.g., 5–10) for plots; the archive itself can remain large.
- Track the growth of occupied bins over generations to demonstrate exploration progress.

## QD Laboratory Report (Structured)

### Objective
- Evaluate the impact of Advanced QD (MAP-Elites + Novelty) on evolutionary auto-architecture search under radiation conditions.
- Measure coverage growth, elite injections, and preservation across protection strategies.

### Materials
- Source code:
  - `include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp` (QD implementation)
  - `include/rad_ml/research/auto_arch_search.hpp` (QD toggles, exporters)
  - `examples/auto_arch_search_example.cpp` (CLI integration, CSV writing)
- Executable: `./examples/auto_arch_search_example`
- Outputs:
  - `auto_arch_search_results.csv`, `auto_search_results.csv`
  - `operator_stats.csv` and parameterized copies
  - `run_summaries.csv`

### Methods
- Evolutionary GA with adaptive mutation and elitism.
- QD enabled via CLI: `--qd --adv-qd`.
- Behavior descriptor: 6D radiation-aware features; discretized into `R^6` cells.
- Novelty: KNN (k=5) average Euclidean distance in behavior space.
- Archive update: replace cell elite if `0.8×preservation + 0.2×novelty×100` improves.
- Elite injection: per generation, replace worst-K with sampled elites (40% fitness, 30% novelty, 30% random occupied).

### Procedure
1. Build and run:
   ```bash
   make && ./examples/auto_arch_search_example \
     --qd --adv-qd \
     --trials 5 --schedule 2 --freeze 4 \
     --save-gen 1 --save-iter 5
   ```
2. Observe console logs for QD coverage, occupied cells, elites injected.
3. Inspect CSVs:
   - `auto_arch_search_results.csv`: per-architecture metrics.
   - `operator_stats.csv`: per-generation operator metrics.
   - `run_summaries.csv`: run metadata summary.
4. Optional: visualize 2D projections using the snippet above.

### Measurements Collected
- Coverage: occupied cells and coverage percentage over generations.
- Elites injected: count per generation (should be >0 once warmed).
- Accuracy preservation distribution by protection level.
- Execution time distributions and correlation with preservation.
- Behavioral diversity (mean pairwise distance) via `getAnalytics()`.

### Analysis Plan
- Coverage growth: plot occupied cells vs. generation; target monotonic increase early.
- Elite injection effectiveness: track change in best preservation following injections.
- Diversity by protection: compare preservation medians for `FULL_TMR`, `ADAPTIVE_TMR`, `SPACE_OPTIMIZED`, etc.
- Trade-space maps: 2D slices (tolerance vs. cost; tolerance vs. power) to highlight niche filling.
- Operator dynamics: use `tools/plot_operator_stats.py` to correlate adaptive rate/diversity and operator success.

### Reproducibility
- CLI flags: document `--trials`, `--schedule`, `--freeze` and seed behavior if applicable.
- Archive resolution: state `R` and dimensions; prefer `R=5` for short runs; `R=10` for longer.
- Include exact commit hash and environment summary (compiler, CPU/GPU, OS).
- Save parameterized `operator_stats_trials{N}_sched{K}_freeze{G}.csv` (already implemented in example).

### References (Internal)
- [`README.md`](../../README.md) → Advanced Quality Diversity section
- [`AUTO_ARCH_SEARCH_GUIDE.md`](../../AUTO_ARCH_SEARCH_GUIDE.md) → Running with QD, CSV formats
- [`autoarchsearchwriteup.md`](../../autoarchsearchwriteup.md) → QD findings
- [`FAQ/GENETICS/Genetic_Algorithm_Architecture.md`](./Genetic_Algorithm_Architecture.md) → GA + adaptive mutation
- Plotting utility: [`tools/plot_operator_stats.py`](../../tools/plot_operator_stats.py)

### Expected Outcomes
- Early small % coverage with large grids; steady occupied-cell growth.
- Consistent elite injection per generation and upward trend in best preservation.
- Healthy diversity across protection strategies with top performers typically in `SPACE_OPTIMIZED/ADAPTIVE_TMR/FULL_TMR` depending on scenario.

## Explain Like I'm 12 (Plain-Language Overview)
- Imagine a huge city grid. Each block represents a "kind of behavior" an architecture can have (fast/slow, tolerant/fragile, etc.).
- Every time we test a new architecture, we place it on the block that matches its behavior. If it's the best we've seen on that block, it becomes the block's "champion."
- We also like tourists (novelty): if a new architecture is unlike the ones we've already seen (far away on the map), we give it extra points.
- Each generation, we take some champions from different blocks and add them back into the population so the search explores many neighborhoods, not just one.

## The 5-Step Loop (Simple)
1) Build some architectures and test them (baseline vs radiation).
2) Turn test results into a behavior point in a 6D map (0..1 per dimension).
3) Put the architecture into its map cell if it's the best there (quality) and note how different it is (novelty).
4) Breed new architectures with selection, crossover, and mutation.
5) Swap out some weak architectures with diverse champions from the map (keeps exploration alive).

## Tiny Toy Example (Numbers You Can Follow)
- Let R=5 bins per dimension. We only look at 2D here (for drawing): radiation_tolerance and computational_cost.
- Say an architecture has `radiation_tolerance=0.72` and `computational_cost=0.31`.
- Discretize to bins: `floor(0.72 * (5-1))=floor(2.88)=2`, `floor(0.31*4)=1` → goes to cell `(2,1)`.
- Novelty: compute 6D distance to previous champions; if we only had a few, we give a novelty of ~1.0 to bootstrap.
- Combined fitness: `0.8*preservation + 0.2*novelty*100`. If preservation=92.0 and novelty≈1.0, then ≈ `0.8*92 + 0.2*100 = 93.6`.
- If `93.6` beats the current champion in `(2,1)`, it becomes the champion there.

## Visualizing Without Code (Emoji/ASCII)
- 2D slice (tolerance vs cost). `●`=champion present, `+`=empty.
```
Radiation tol ↑

[4]  ●  +  +  +  +
[3]  +  +  ●  +  +
[2]  +  ●  +  +  +
[1]  +  +  +  ●  +
[0]  +  +  +  +  +
      0  1  2  3  4  → Computational cost
```
- Over generations, watch `●` spread to more places: that's coverage growing.

## FAQ (Quick Q&A)
- Q: Why not just keep the single best model?
  - A: In radiation environments, the "best" can change with mission constraints. QD preserves many strong-but-different options.
- Q: Why novelty?
  - A: It rewards exploring new parts of the map, preventing the search from getting stuck.
- Q: Why is percent coverage so small at first?
  - A: The map is huge (R^6 cells). Use absolute occupied cells and elites injected as early indicators.
- Q: How do I know it's working?
  - A: You should see >0 elites injected most generations and the number of occupied cells increasing over time.

## Cheat Sheet (At a Glance)
- Discretization: `coord = floor(clamp01(value) * (R-1))`.
- Novelty (K=5): average distance to 5 nearest behaviors.
- Combined fitness (for archive updates): `0.8*preservation + 0.2*novelty*100`.
- Elite sampling per generation: ~40% top fitness, 30% top novelty, 30% random occupied.
- Good defaults: R=10 (R=5 for short runs), novelty weight≈0.2, replacement≈20%.

## Cells in MAP-Elites (Clear Explanation)

### What is a "cell"?
- A cell is one discrete niche in the behavior space. In our 6D map (values in [0,1]), each dimension is split into R bins.
- The full archive is a 6D grid of size `R^6`. Each grid position (cell) stores the "elite": the best architecture ever found for that niche.

### How do behaviors map to cells?
- Normalize each descriptor dimension to [0,1].
- Discretize to an integer coordinate per dim: `coord = floor(clamp01(value) * (R-1))`.
- Flatten 6D coords to a single index (row-major) to address the archive.

Code alignment:
- Discretization: `discretizeBehavior(...)`
- Indexing: `coordsToIndex(...)`
- See: [`advanced_quality_diversity.hpp`](../../include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp)

### Cell lifecycle (empty → candidate → elite)
1) Empty: fitness is `-inf` (no elite yet).
2) Candidate arrives: compute combined fitness `0.8×preservation + 0.2×novelty×100`.
3) Replace rule: if the cell is empty OR the candidate's combined fitness is higher than the cell's fitness, the candidate becomes the new elite for that cell.
4) Side effects on accept:
   - Store behavior, novelty, objective vector, and generation.
   - Append behavior to novelty archive (for future KNN).

Function alignment:
- Archive update: `addToArchive(config, result, generation)`
- Combined fitness: `calculateCombinedFitness(...)`
- Novelty update: `updateNoveltyArchive(...)`

### What if two candidates land in the same cell?
- Only the one with higher combined fitness remains as the elite. The other is discarded for archive purposes (but may still be part of the GA population if selected).

### Why discretize? (Intuition)
- The grid forces us to keep "the best of each kind" instead of only "the best overall."
- This preserves multiple strong-but-different solutions (e.g., low-cost but medium-tolerance vs. higher-cost but top-tolerance) that are valuable for different mission constraints.

### How do cells drive the GA each generation?
- After evaluation, we sample elites from diverse cells (by fitness, novelty, and random occupied cells).
- Those elites are injected by replacing the worst-K population members, keeping the search from collapsing onto a single niche.

Sampling alignment:
- `sampleDiverseElites(sample_size)` returns a mixed set: ~40% best `fitness_score`, ~30% best `novelty_score`, ~30% uniform occupied.

### Minimal numeric example (3D slice for clarity)
- Let R=5, and consider dims: tolerance, cost, power.
- Behavior `(tol=0.76, cost=0.22, power=0.58)` → coords `(floor(0.76*4)=3, floor(0.22*4)=0, floor(0.58*4)=2)` → cell `(3,0,2)`.
- Novelty is computed against stored behaviors; suppose novelty=0.9.
- Combined fitness with preservation=93.0: `0.8*93 + 0.2*0.9*100 = 74.4 + 18.0 = 92.4`.
- If cell `(3,0,2)` is empty or has fitness < 92.4, this config becomes the elite.

### Purpose within AutoArch
- Portfolio of options: Cells ensure the search maintains multiple top designs tailored to different trade-offs (tolerance, cost, power, etc.).
- Robust exploration: Elite injection from diverse cells prevents premature convergence and improves coverage.
- Mission alignment: Different missions can prioritize different niches; the archive surfaces elites per niche for quick, informed selection.
- Faster learning: Novelty incentivizes exploring underrepresented regions, discovering architectures the plain GA might never reach.

### How to read progress
- Occupied cells over time: growth indicates broader exploration.
- Elites injected per generation: consistent >0 shows QD is influencing the population.
- 2D projections: quick visual of which niches have champions (dots) and which remain empty.

# Mathematical Formulation (LaTeX)

## Behavior Descriptor

Let the 6D behavior descriptor be $\mathbf{x} = (x_{ac}, x_{pe}, x_{cc}, x_{rt}, x_{gd}, x_{pw})$.

- **Architectural complexity** (from layer sizes $s_1,\dots,s_m$):
$$\text{params} = \sum_{i=1}^{m-1} s_i s_{i+1}, \quad x_{ac} = \frac{\ln(\text{params}+1)}{\ln(10^6)}$$

- **Protection efficiency** (errors detected $E_d$, corrected $E_c$):
$$x_{pe} = \begin{cases}
\frac{E_c}{E_d}, & E_d > 0 \\
1, & E_d = 0
\end{cases}$$

- **Computational cost** (execution time in ms $t_{ms}$):
$$x_{cc} = \min\left\{1, \frac{t_{ms}}{1000}(1 + x_{ac})\right\}$$

- **Radiation tolerance** (accuracy preservation $P$ in %):
$$x_{rt} = \frac{P}{100}$$

- **Graceful degradation** (baseline $A_b$, radiation $A_r$):
$$\delta = \max\left\{0, \frac{A_b - A_r}{\max(10^{-9}, A_b)}\right\}, \quad x_{gd} = 1 - \delta$$

- **Power efficiency** (protection overhead $o$ by level):
$$x_{pw} = \frac{1}{1 + o + x_{ac}}$$

**Typical overhead mapping:** NONE $\to 0.0$, CHECKSUM $\to 0.1$, SELECTIVE_TMR $\to 0.5$, FULL_TMR $\to 1.0$, ADAPTIVE_TMR $\to 0.7$, SPACE_OPTIMIZED $\to 0.3$.

## Discretization and Indexing (MAP grid)

For grid resolution $R$ per dimension, clamp and discretize each component:
$$\tilde{x}_i = \min\{1,\max\{0, x_i\}\}, \quad c_i = \left\lfloor \tilde{x}_i(R-1) \right\rfloor, \quad i=1,\dots,6$$

Flatten 6D coordinates to a 1D index (row-major with base $R$):
$$\text{index}(\mathbf{c}) = \sum_{i=1}^{6} c_i R^{i-1}$$

## Novelty (K-Nearest Neighbors in behavior space)

With Euclidean distance $d(\mathbf{x},\mathbf{y}) = \sqrt{\sum_{i=1}^6 (x_i - y_i)^2}$ and behavior archive $\mathcal{A}$, define novelty:
$$\eta(\mathbf{x}) = \begin{cases}
1, & |\mathcal{A}| < K \\
\frac{1}{K}\sum_{\mathbf{y} \in \mathcal{N}_K(\mathbf{x})} d(\mathbf{x},\mathbf{y}), & \text{otherwise}
\end{cases}$$

Here $\mathcal{N}_K(\mathbf{x})$ are the $K$ nearest neighbors to $\mathbf{x}$ in $\mathcal{A}$.

## Combined Fitness and Archive Update Rule

Combined fitness used to compare candidates within a cell ($\alpha=0.8$):
$$F(\mathbf{x}, P) = \alpha P + (1-\alpha) \cdot 100 \cdot \eta(\mathbf{x})$$

Cell replacement policy for cell $\mathbf{c}$:
$$\text{if cell is empty or } F_{\text{candidate}} > F_{\text{cell}} \Rightarrow \text{replace elite with candidate}$$

## Coverage and Diversity Metrics

- **Coverage** (fraction of occupied cells):
$$\text{Coverage} = \frac{N_{\text{occupied}}}{R^6}$$

- **Behavioral diversity** (mean pairwise distance among occupied cells with behaviors $\{\mathbf{x}^{(1)},\dots,\mathbf{x}^{(n)}\}$):
$$\Delta = \frac{2}{n(n-1)} \sum_{1 \leq i < j \leq n} d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})$$

## Elite Sampling Mix (for GA Injection)

Each generation, sample elites for injection with proportions:
$$40\% \text{ highest } F, \quad 30\% \text{ highest } \eta, \quad 30\% \text{ uniform over occupied cells}$$

## 2D Projection for Visualization

Given dimensions $p,q \in \{1,\dots,6\}$, define 2D occupancy matrix $A \in \{0,1\}^{R \times R}$:
$$A_{c_p, c_q} = \begin{cases}
1, & \exists \text{ occupied cell with } (c_p, c_q) \text{ on dims } (p,q) \\
0, & \text{otherwise}
\end{cases}$$

This projection aids visualizing coverage growth along two intuitive axes (e.g., tolerance vs cost).

## Variable Glossary and Code Cross-References

- params
  - Meaning: total trainable parameters from adjacent layer products.
  - Code: sum over `config.layer_sizes[i-1] * config.layer_sizes[i]` in `calculateRadiationAwareBehavior(...)`.
  - Units/Range: count (≥0). Used inside `log(params+1)` then normalized by `log(1e6)`.

- x_ac (architectural_complexity)
  - Meaning: normalized log-parameters.
  - Formula: `log(params+1)/log(1e6)`.
  - Code: `b.architectural_complexity`.
  - Range: approximately [0,1] after normalization; clamped later before discretization.

- E_d, E_c (errors_detected, errors_corrected)
  - Meaning: total detected/corrected faults during radiation testing.
  - Code: `result.errors_detected`, `result.errors_corrected`.
  - Units: counts (integers ≥0).

- x_pe (protection_efficiency)
  - Meaning: fraction of detected errors that were corrected (fallback 1 if none detected).
  - Formula: `E_c / E_d` if `E_d>0`, else 1.
  - Code: `b.protection_efficiency`.
  - Range: [0,1].

- t_ms (execution_time_ms)
  - Meaning: wall-clock execution time per evaluation.
  - Code: `result.execution_time_ms`.
  - Units: milliseconds (ms).

- x_cc (computational_cost)
  - Meaning: normalized cost proxy combining time and complexity.
  - Formula: `min(1, (t_ms/1000) * (1 + x_ac))`.
  - Code: `b.computational_cost`.
  - Range: [0,1].

- P (accuracy_preservation)
  - Meaning: preservation percentage under radiation relative to baseline.
  - Code: `result.accuracy_preservation`.
  - Units: percent (0–100). Used both raw and normalized.

- x_rt (radiation_tolerance)
  - Meaning: normalized preservation.
  - Formula: `P / 100`.
  - Code: `b.radiation_tolerance`.
  - Range: [0,1].

- A_b, A_r (baseline_accuracy, radiation_accuracy)
  - Meaning: accuracies before/after radiation injection.
  - Code: `result.baseline_accuracy`, `result.radiation_accuracy`.
  - Units: percent (0–100).

- x_gd (graceful_degradation)
  - Meaning: 1 minus relative drop from baseline to radiation accuracy (clamped to [0,1]).
  - Formula: `1 - max(0, (A_b - A_r) / max(1e-9, A_b))`.
  - Code: `b.graceful_degradation`.
  - Range: [0,1].

- o (protection_overhead)
  - Meaning: relative hardware/compute overhead by protection level.
  - Code: `getProtectionOverhead(config.protection_level)`.
  - Mapping: NONE=0.0, CHECKSUM_ONLY=0.1, SELECTIVE_TMR=0.5, FULL_TMR=1.0, ADAPTIVE_TMR=0.7, SPACE_OPTIMIZED=0.3.

- x_pw (power_efficiency)
  - Meaning: proxy inverse to overhead and complexity.
  - Formula: `1 / (1 + o + x_ac)`.
  - Code: `b.power_efficiency`.
  - Range: (0,1].

- R (grid resolution)
  - Meaning: number of bins per descriptor dimension.
  - Code: `current_grid_resolution_` (default 10; `BASE_GRID_RESOLUTION`).
  - Range: integers ≥2; capped by `MAX_GRID_RESOLUTION` (50).

- c_i (cell coordinate per dimension)
  - Meaning: discretized index along dimension i.
  - Formula: `c_i = floor(clamp01(x_i) * (R-1))`.
  - Code: `discretizeBehavior(...)`.
  - Range: integers in `[0, R-1]`.

- index(c)
  - Meaning: flattened 1D index for the 6D cell.
  - Formula: `sum_{i=1..6} c_i * R^{i-1}`.
  - Code: `coordsToIndex(...)`.

- d(x,y) (behavior distance)
  - Meaning: Euclidean distance in 6D descriptor space.
  - Formula: `sqrt(sum_i (x_i - y_i)^2)`.
  - Code: `calculateBehavioralDistance(...)`.

- η(x) (novelty)
  - Meaning: average distance to K nearest archived behaviors; equals 1 when archive is too small.
  - Formula: `1` if `|A|<K`, else `(1/K) * sum_{y in N_K(x)} d(x,y)`.
  - Code: `calculateNoveltyScore(...)`; K=`K_NEAREST_NEIGHBORS` (5).

- F(x,P) (combined fitness)
  - Meaning: scalar used for cell elite comparison (not GA selection directly).
  - Formula: `0.8 * P + 0.2 * 100 * η(x)`.
  - Code: `calculateCombinedFitness(...)`.
  - Note: `P` is in percent; novelty is scaled by 100 to match magnitude.

- Coverage
  - Meaning: fraction of occupied cells.
  - Formula: `N_occupied / R^6`.
  - Code: computed in `updateArchiveStatistics()` and `getAnalytics()`.

- Behavioral diversity Δ
  - Meaning: mean pairwise distance among occupied-cell behaviors.
  - Formula: `(2/(n(n-1))) * sum_{i<j} d(x^(i), x^(j))`.
  - Code: `calculateBehavioralDiversity(...)`.

Edge cases and clamping:
- All descriptor components are clamped to [0,1] before discretization: see `discretizeBehavior(...)`.
- Division by zero avoided via `max(1e-9, A_b)` in degradation and `E_d>0` guard in protection efficiency.
- Novelty bootstrap: returns `1.0` until at least K descriptors exist, ensuring early exploration signal.

## Code Cross-Reference: Implementation Map

### Core Data Types
- Behavior and archive manager: [`advanced_quality_diversity.hpp`](../../include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp)
- Search interface/toggles/CSV export: [`auto_arch_search.hpp`](../../include/rad_ml/research/auto_arch_search.hpp)
- Example CLI and run wiring: [`examples/auto_arch_search_example.cpp`](../../examples/auto_arch_search_example.cpp)
- Evolutionary loop implementation: [`evolutionary.cpp`](../../src/rad_ml/research/auto_arch/evolutionary.cpp)
- Types and tester: [`types.hpp`](../../include/rad_ml/research/auto_arch/types.hpp), [`architecture_tester.hpp`](../../include/rad_ml/research/architecture_tester.hpp)

### Descriptor, Discretization, Indexing
- Compute descriptor (6D): `calculateRadiationAwareBehavior(config, result)` → populates `architectural_complexity`, `protection_efficiency`, `computational_cost`, `radiation_tolerance`, `graceful_degradation`, `power_efficiency`.
- Discretize to grid: `discretizeBehavior(...)` → returns 6 coords `c_i ∈ [0, R-1]`.
- Flatten index: `coordsToIndex(...)` → maps 6D coords to 1D cell index.
- Protection overhead mapping: `getProtectionOverhead(config.protection_level)`.

### Novelty and Distances
- Euclidean distance: `calculateBehavioralDistance(const Behavior&, const Behavior&)`.
- KNN novelty: `calculateNoveltyScore(const Behavior&)` (K=5, bootstrap to 1.0 when archive < K).
- Novelty archive maintenance: `updateNoveltyArchive(const Behavior&)` (size capped ~1000).

### Archive Update and Analytics
- Add/replace elite: `addToArchive(const NetworkConfig&, const ArchitectureTestResult&, size_t generation)`.
- Combined fitness: `calculateCombinedFitness(const ArchitectureTestResult&, double novelty)` (0.8×pres + 0.2×100×novelty).
- Coverage stats: `updateArchiveStatistics()`; analytics struct and getter: `ArchiveAnalytics`, `getAnalytics()`.

### Sampling and GA Injection
- Sample elites for replacement: `sampleDiverseElites(size_t sample_size)` → 40% best `fitness_score`, 30% best `novelty_score`, 30% uniform occupied.
- Replace worst-K in GA (in example integration): performed after GA offspring generation using sampled elites.

### GA Wiring and Execution
- Toggle QD from code: `enableQualityDiversity(bool)`, `enableAdvancedQualityDiversity(bool)` in [`auto_arch_search.hpp`](../../include/rad_ml/research/auto_arch_search.hpp).
- Example CLI flags and usage: `--qd`, `--adv-qd` in [`examples/auto_arch_search_example.cpp`](../../examples/auto_arch_search_example.cpp) →
  `searcher.enableQualityDiversity(qd_enabled_cli);` and `searcher.enableAdvancedQualityDiversity(adv_qd_enabled_cli);`.
- Evolutionary search loop (selection, elitism, mutation): [`evolutionary.cpp`](../../src/rad_ml/research/auto_arch/evolutionary.cpp).

### Results and Logs
- Export results: `exportResults(const std::string& filename)` in [`auto_arch_search.hpp`](../../include/rad_ml/research/auto_arch_search.hpp).
- Example writes: `auto_arch_search_results.csv`, `run_summaries.csv`, and `operator_stats.csv` in [`examples/auto_arch_search_example.cpp`](../../examples/auto_arch_search_example.cpp).
- Operator plots: [`tools/plot_operator_stats.py`](../../tools/plot_operator_stats.py).

### End-to-End Trace (From CLI to Cell Update)
1) CLI flags parsed in example → toggles QD via `enableQualityDiversity(...)`, `enableAdvancedQualityDiversity(...)`.
2) GA evaluates configs → produces `ArchitectureTestResult` (see `architecture_tester.hpp`).
3) QD compute: `calculateRadiationAwareBehavior(...)` → `discretizeBehavior(...)` → `coordsToIndex(...)`.
4) Novelty: `calculateNoveltyScore(...)` (uses `calculateBehavioralDistance(...)` and novelty archive).
5) Combined fitness: `calculateCombinedFitness(...)`.
6) Archive update: `addToArchive(...)` (store elite, novelty, objectives, generation; update novelty archive).
7) Analytics update: `updateArchiveStatistics()`, `getAnalytics()`.
8) Sampling for injection: `sampleDiverseElites(...)` → replace worst-K in population in the GA step.
9) Persist results: `exportResults(...)` and example CSV/plot utilities.


## Wide Networks (Wider Layer Options)

You can test wider architectures via the example’s `--widths` flag (comma-separated list):

```bash
./examples/auto_arch_search_example --qd --adv-qd \
  --trials 100 \
  --widths 32,64,128,256,512,1024
```

Notes:
- Wider layers increase parameter count → raises `x_ac` (architectural complexity), and may increase `x_cc` (computational cost) via execution time.
- QD will naturally place these in higher-complexity, potentially higher-cost cells; elites will reflect preservation vs cost trade-offs.
- If exploration collapses to very wide models, increase novelty weight slightly (e.g., 0.25) or include more mid-range widths to keep multiple niches competitive.
