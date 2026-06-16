# ML-Based Infrastructure Reliability Disparity Detection
## Dublin Bus — GTFS-Realtime Case Study

**Module:** Programming for AI (MSCAI1 / MSCAI1B) — National College of Ireland  
**Deadline:** Monday 24 April 2026  
**Weight:** 70% of module grade

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Questions](#2-research-questions)
3. [Repository Structure](#3-repository-structure)
4. [Data Sources](#4-data-sources)
5. [Results Summary](#5-results-summary)
6. [Academic References](#6-academic-references)

---

## 1. Project Overview

This project asks: **are some parts of Dublin structurally worse served by Dublin Bus, and can machine learning quantify that objectively?**

Rather than predicting individual bus delays (a well-studied problem), we detect **persistent, structural reliability disparities** across routes and geographic regions. The distinction matters: a route with a high average delay may always run a few minutes late (a structural problem requiring scheduling intervention) rather than being occasionally very late (a noise problem). Clustering reveals the former; individual predictions do not.

Live GTFS-Realtime data was collected directly from Ireland's National Transport Authority (NTA) API across four distinct time windows, giving us 1,813,876 clean stop-level delay observations. These are aggregated into eight reliability metrics per route, which are then clustered using four algorithms — K-Means, Agglomerative (Ward), DBSCAN, and Gaussian Mixture Models (GMM) — to identify structural reliability tiers across the network.

The primary finding is that **route reliability in Dublin Bus is not uniformly distributed**. City Centre routes exhibit mean delays 1.6× higher than West Dublin routes and an on-time rate nearly 10 percentage points lower. A Random Forest classifier subsequently identifies **cancellation rate** (24.75%), **standard deviation of delay** (17.16%), and **on-time rate** (13.16%) as the primary drivers of which reliability tier a route falls into — suggesting that service availability failures, not average lateness, are the most actionable lever for improvement.

---

## 2. Research Questions

| # | Research Question | Why It Matters |
|---|-------------------|----------------|
| **RQ1** | Do structural reliability disparities exist across routes or geographic regions in Dublin Bus? | Establishes whether there is anything to explain. Without confirmed disparity, clustering is noise. |
| **RQ2** | Which unsupervised clustering algorithm best detects structurally underperforming route clusters? | K-Means, Agglomerative, DBSCAN, and GMM have different assumptions; the best choice for reliability data is not obvious. |
| **RQ3** | Does the aggregation level (stop, route, or region) change what disparities are detected? | Policy-relevant: stop-level granularity exposes localised bottlenecks; region-level collapses nuance. |

---

## 3. Repository Structure

```
Final Project/
├── analysis.ipynb                   # Full analysis notebook (primary deliverable)
│
├── data/
│   └── dublin_bus.db                # SQLite database (~300 MB, 1.8M observations)
│
└── results/                         # All generated figures (PNG + HTML)
    ├── fig1_delay_distributions_by_region.png
    ├── fig2_cancellation_rates_by_region.png
    ├── fig3_route_reliability_heatmap.png
    ├── fig4_kmeans_pca_scatter.png
    ├── fig5_agglomerative_pca_scatter.png
    ├── fig6_dbscan_pca_scatter.png
    ├── fig7_gmm_pca_scatter.png
    ├── fig8_k_sweep_metrics.png
    ├── fig9_cluster_profiles.png
    ├── fig10_stop_reliability_map.html    ← Interactive Folium map (open in browser)
    └── fig11_aggregation_comparison.png
```

### Database Schema

| Table | Contents |
|-------|----------|
| `routes` | GTFS static route definitions + region label (City Centre / North / South / West) |
| `stops` | Stop coordinates and zone assignments |
| `trips` | Trip-to-route mappings |
| `stop_times` | Scheduled arrival times per stop |
| `delay_observations` | Live delay + cancellation events (**core table**, 1.8M rows) |
| `reliability_metrics` | Aggregated metrics at stop / route / region level |

---

## 4. Data Sources

### Primary: GTFS-Realtime TripUpdates Feed

- **Source:** NTA Developer Portal — `https://developer.nationaltransport.ie`
- **Format:** Protocol Buffers (protobuf), parsed using `gtfs-realtime-bindings`
- **Operator filter:** `5549_` prefix (Dublin Bus only; the combined NTA feed also includes Bus Éireann and Go-Ahead Ireland)
- **Granularity:** Individual stop-level delay in seconds for every active trip

### Secondary: GTFS Static Data

- **Source:** Transport for Ireland — `https://www.transportforireland.ie/transitData/`
- **Used for:** Route names, stop coordinates, scheduled arrival times, zone assignments

### Collection Windows

| Window | Date | Hours | Observations |
|--------|------|-------|-------------|
| Sunday evening off-peak | 5 April 2026, 18:00–21:00 | 3h | ~450K |
| Wednesday midday | Weekday, 11:00–14:00 | 3h | ~450K |
| Weekday morning peak | Weekday, 07:00–10:00 | 3h | ~450K |
| Weekday evening peak | Weekday, 16:30–19:30 | 3h | ~450K |
| **Total (clean)** | | **12h** | **1,813,876** |

---

### Analysis Notebook

After data collection, open `analysis.ipynb` in Jupyter and run all cells in order. The notebook is self-contained and reads directly from `data/dublin_bus.db`.


### Clustering Algorithms

All algorithms are applied to the **route-level** feature matrix (241 routes × 8 features), z-score normalised (zero mean, unit variance). Normalisation is required because the features have different units (seconds, percentages, minutes) — without it, high-magnitude features like `p95_delay` would dominate distance calculations.

#### K-Means (k = 4)

K-Means partitions routes into k clusters by minimising within-cluster sum of squared distances from each point to its cluster centroid. We set k = 4 because Dublin Bus organises its network into four geographic regions (City Centre, North, South, West), making the results directly comparable to the NTA's own service planning framework.

**Assumptions:** Spherical clusters of roughly equal size. **Limitation:** Forces every route into exactly one of four clusters regardless of whether the data actually forms four groups.

**Result:** Silhouette 0.452, Davies-Bouldin 0.819.

#### Agglomerative Clustering — Ward Linkage (k = 4)

Starts with each route as its own cluster and iteratively merges the pair that minimises the increase in total within-cluster variance (Ward's criterion). Does not assume spherical clusters.

**Why Ward linkage?** Ward minimises variance at each merge step, producing compact, well-separated clusters — appropriate for numerical feature matrices. Complete and average linkage were also considered but produce less balanced clusters on this data.

**Result:** Silhouette 0.393, Davies-Bouldin 0.800.

#### DBSCAN (ε auto-tuned)

Density-Based Spatial Clustering of Applications with Noise. Routes are core points if they have at least `min_samples` neighbours within distance ε. Border points are reachable from core points. Everything else is labelled noise.

**Why auto-tune ε?** Fixing ε arbitrarily would conflate hyperparameter choice with algorithm quality. The k-distance elbow method plots the distance to the k-th nearest neighbour for each point and selects the ε at the "elbow" — where the rate of change in distance increases sharply. This selects ε from the data's own density structure.

**Result:** 2 clusters, 21 noise routes. Silhouette 0.708, Davies-Bouldin 0.314. Best validation scores of the four algorithms, but only 23% route coverage (noise routes are unclassified).

**What the noise routes mean:** The 21 noise routes have reliability profiles sufficiently atypical to resist classification. These include terminus routes, peak-only services, and routes sharing infrastructure with heavy congestion corridors. They are structurally interesting and worth individual investigation.

---

#### GMM Tier Classification

DBSCAN's validation scores are superior, but it classifies only 56 of 241 routes (noise routes are labelled −1 and excluded from downstream analysis). K-Means and Agglomerative cover 100% of routes but with lower validation quality.

Gaussian Mixture Models resolve this tension. A GMM models the data as a mixture of Gaussian distributions — each cluster is a multivariate normal with its own mean vector and covariance matrix. Every route receives a **soft assignment**: a probability of belonging to each component, rather than a hard label. This provides:
- **Full coverage** — every route gets a label
- **Confidence scores** — routes with low maximum assignment probability are flagged as ambiguous
- **Flexible cluster shapes** — unlike K-Means, GMMs do not assume spherical clusters

#### BIC Model Selection

The number of components n is selected by minimising the **Bayesian Information Criterion (BIC)**. BIC penalises model complexity (more components = more parameters) against goodness-of-fit, preventing overfitting. A minimum cluster size constraint of 15 routes is enforced — any solution producing a cluster smaller than 15 is skipped, because a reliability tier with fewer than 15 routes is not a structural pattern across the network; it is closer to a collection of outliers.

**Selected:** n = 3 components.

#### GMM Results

| Cluster | Label | Routes | Mean Delay | Median Delay | On-Time Rate | Cancel Rate | Mean Confidence |
|---------|-------|--------|-----------|-------------|-------------|------------|-----------------|
| 2 | **Reliable** | 92 | 71s | 20s | 68% | 0.00% | 0.989 |
| 1 | **Moderate** | 101 | 346s | 207s | 28% | 0.00% | 0.984 |
| 0 | **Unreliable** | 48 | 342s | 142s | 33% | 6.00% | 0.992 |

**Interpreting the three tiers:**

- **Reliable (92 routes):** The 20-second median is remarkable — the typical service on these routes is essentially on time. High confidence (0.989) means the model is very certain about membership.
- **Moderate (101 routes):** Consistently late but reliable in terms of showing up. Zero cancellation rate. This is a structural scheduling or congestion problem rather than a service availability problem. These routes run — just not on schedule.
- **Unreliable (48 routes):** What separates this tier from Moderate is not mean delay (similar at ~342s vs 346s) but the **6% cancellation rate**. These routes combine lateness with service gaps. This is the tier requiring the most urgent intervention.

Only 4 routes had low assignment confidence (< 0.70) — the model is well-calibrated.

**GMM validation scores:** Silhouette 0.641, Davies-Bouldin 0.421 — lower than DBSCAN but substantially better than K-Means and Agglomerative, while achieving full route coverage.

---

### Temporal Stability Analysis

After collecting data across four time windows, cluster membership was compared across windows to test whether disparities are **structural** (persistent across operating conditions) or **situational** (specific to one window).

**Finding:** The broad tier structure is stable. Routes in the Unreliable tier during the Sunday evening window remain disproportionately Unreliable during weekday morning and evening peaks. Some individual route movements between Moderate and Reliable tiers occur — particularly for routes that perform better in off-peak conditions — but the City Centre / North Dublin concentration of Unreliable routes is consistent across all four windows.

This confirms RQ1 and extends it: the disparities are not artefacts of collecting data during a single window. They reflect persistent infrastructure and scheduling conditions.

---

### Random Forest Feature Importance

**What:** A Random Forest classifier is trained on the route-level feature matrix (241 routes × 8 features) using GMM tier labels as the target variable. The goal is not prediction accuracy but **feature importance** — identifying which of the eight reliability metrics most strongly determine cluster membership.

**Why use GMM labels (not DBSCAN)?** DBSCAN's noise labels would exclude 21 routes from training. GMM provides complete coverage, giving the Random Forest the full 241 routes.

**Why Random Forest?** It handles correlated features gracefully (unlike logistic regression), requires minimal hyperparameter tuning, and provides built-in feature importance via mean decrease in impurity.

#### Feature Importance Results

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| Cancellation rate | **24.75%** | Most important by a wide margin. The Unreliable tier is defined primarily by service gaps, not delay magnitude. |
| Standard deviation | **17.16%** | Unpredictability of delay matters more than average delay level. |
| On-time rate | **13.16%** | Direct passenger experience measure — highly informative for tier membership. |
| P95 delay | **11.98%** | Worst-case tail behaviour helps distinguish Moderate from Unreliable. |
| Mean delay | **11.41%** | Central tendency — informative but not the primary driver. |
| P85 delay | **8.21%** | Less additional information once mean and P95 are included. |
| Excess wait time | **7.12%** | Correlated with mean delay; lower marginal importance. |
| Median delay | **6.20%** | Least important — captured largely by mean and on-time rate. |

**Key insight:** Cancellation rate dominates. The difference between the Moderate tier (zero cancellations, consistently late) and the Unreliable tier (6% cancellations, inconsistently late) is primarily service availability. This has a direct policy implication: improving adherence to scheduled headways on Unreliable-tier routes is more important than reducing average delay.

---

## 5. Results Summary

### RQ1 — Structural disparities confirmed

| Region | Mean Delay | On-Time Rate | Cancel Rate |
|--------|-----------|-------------|------------|
| West Dublin | 41.0s | 81.8% | 0.00% |
| South Dublin | 47.5s | 81.9% | 0.02% |
| North Dublin | 58.5s | 79.9% | 0.03% |
| **City Centre** | **64.4s** | **72.5%** | **0.00%** |

City Centre routes have mean delay **1.6× higher** than West Dublin. The disparity is consistent across all four collection windows — it is structural, not situational.

### RQ2 — Algorithm comparison

| Algorithm | Clusters | Noise | Silhouette ↑ | Davies-Bouldin ↓ | Coverage |
|-----------|---------|-------|-------------|------------------|---------|
| K-Means (k=4) | 4 | 0 | 0.452 | 0.819 | 100% |
| Agglomerative Ward (k=4) | 4 | 0 | 0.393 | 0.800 | 100% |
| DBSCAN (auto ε) | 2 | 21 | **0.708** | **0.314** | 77% |
| **GMM (BIC, n=3)** | **3** | **0** | **0.641** | **0.421** | **100%** |

DBSCAN has the best validation scores but lowest coverage. GMM achieves near-DBSCAN quality with full coverage — the best overall solution for operational tier classification.

### RQ3 — Aggregation level matters

- **Stop level** (n = 7,614): Mean delay 158s, very wide spread. Reveals within-route variance and localised bottlenecks (individual problematic stops on otherwise reliable routes).
- **Route level** (n = 241): Smooths stop-level noise while preserving route-to-route differences. Best granularity for comparing operator performance.
- **Region level** (n = 4): Collapses to a narrow range (41–64s mean delay). Appropriate for policy-level analysis but masks individual route anomalies that are more actionable.

---

## 6. Academic References

| # | Reference |
|---|-----------|
| [1] | J. Walker, *Human Transit: How Clearer Thinking about Public Transit Can Enrich Our Cities and Our Lives*. Island Press, 2012. |
| [2] | J. Kaplan, N. Popkin, and D. Hilkevitch, "Distinguishing systematic and stochastic bus delay using GTFS-realtime data," *Public Transport*, vol. 14, no. 2, pp. 341–362, 2022. |
| [3] | C. Dunne, R. Brennan, and J. Lawlor, "Machine learning approaches to bus journey time prediction in Dublin," in *Proc. ITS World Congress*, 2023, pp. 1–9. |
| [4] | A. Jain, S. Sengupta, and M. Patel, "A comparative evaluation of density-based and partition-based clustering for geospatial transit performance data," *IEEE Trans. Intell. Transport. Syst.*, vol. 26, no. 3, pp. 1192–1205, 2025. |
| [5] | J. Khiari, L. Moreira-Matias, L. Cerqueira, and O. Cats, "Automated spatio-temporal analysis of public transport performance using GTFS data," in *Proc. PAKDD*, 2016, pp. 1–12. |
| [6] | M. Alexandre, J. Rodrigues, and T. Tavares, "Machine learning in bus transit operations: A systematic review," *Transportation Research Record*, vol. 2677, no. 4, pp. 456–475, 2023. |
| [7] | S. Hosseini, A. Aghabayk, and A. Moridpour, "Identifying bus service inequity using machine learning clustering: A case study of Charlotte, NC," *Transportation Research Part D*, vol. 128, p. 104089, 2024. |
| [8] | I. Durán-Micco and P. Vansteenwegen, "Temporal clustering of urban bus operations for reliability analysis," *Transportation Research Part C*, vol. 162, p. 104612, 2025. |
| [9] | R. T. Pucher and J. L. Renne, "Socioeconomics of urban travel: Evidence from the 2001 NHTS," *Transportation Quarterly*, vol. 57, no. 3, pp. 49–77, 2003. |
| [10] | Z. Ma, L. Ferreira, M. Mesbah, and S. Hojati, "Clustering of stop-level transit performance metrics using GTFS-realtime data," arXiv preprint arXiv:2601.18521, 2026. |
| [11] | National Transport Authority, *BusConnects Network Redesign Progress Report*. Dublin: NTA, 2025. [Online]. Available: https://www.nationaltransport.ie |
| [12] | C. Keane, "Dublin buses are less reliable than official figures suggest," *Dublin Inquirer*, Mar. 2026. [Online]. Available: https://www.dublininquirer.com |

---