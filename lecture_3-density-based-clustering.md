# Lecture 3: Density-based Clustering

## Overview

This lecture introduces **density-based clustering**, which departs from the rigid geometric structure of representative-based methods. Instead of using fixed representatives, density-based methods define clusters as high-density regions separated by regions of lower density. This allows them to find clusters of arbitrary shapes and automatically determine the number of clusters. We'll study two main algorithms: DBSCAN and DENCLUE, and also cover methods for evaluating clustering quality.

---

## Why Density-Based Clustering?

Representative-based clustering (like k-means) has limitations:
- Requires specifying the number of clusters $k$ in advance
- Assumes clusters have convex shapes (spherical, elliptical)
- Struggles with clusters that have irregular boundaries

**Density-based clustering** addresses these by using a different intuition: clusters are **dense regions** in the data space, separated by regions of lower object density. This allows finding clusters of arbitrary shapes without needing to specify $k$ beforehand.

### Applications

Density-based clustering is particularly useful for:
- **Spatial data analysis**: Detecting areas with similar characteristics (population density, landslide hazards)
- **Image analysis**: Identifying homogeneous color regions in biomedical images
- **Trajectory analysis**: Monitoring moving objects in airspace
- **Astrophysics**: Analyzing arrival directions of photons

---

## DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is the foundational density-based clustering algorithm. It's based on the intuition that points in a cluster should be "density-reachable" from other points in that cluster.

### Basic Concept

For any point in a cluster, the **local point density** around that point must exceed some threshold. The set of points from one cluster is **spatially connected** through dense regions.

DBSCAN defines density using two parameters:
- **$\varepsilon$ (epsilon)**: Radius for the neighborhood of a point
- **MinPts**: Minimum number of points required in a neighborhood to consider it dense

### Key Definitions

**$\varepsilon$-neighborhood** of a point $q$:
$$N_\varepsilon(q) = \{p \in D \mid dist(p, q) \leq \varepsilon\}$$

This is the set of all points within distance $\varepsilon$ of $q$.

**Core object** (or core point): A point $q$ is a core object with respect to $\varepsilon$ and MinPts if:
$$|N_\varepsilon(q)| \geq MinPts$$

In other words, a core object has at least MinPts points in its $\varepsilon$-neighborhood. Core objects are the "backbone" of clusters - they define dense regions.

**Directly density-reachable**: A point $p$ is directly density-reachable from $q$ within $\varepsilon$, MinPts if:
- $p \in N_\varepsilon(q)$ (p is in q's neighborhood)
- $q$ is a core object

This is a one-way relationship - if $p$ is directly density-reachable from $q$, it doesn't mean $q$ is directly density-reachable from $p$ (unless $p$ is also a core object).

**Density-reachable**: A point $p$ is density-reachable from $q$ if there exists a chain of points $p_1, p_2, \ldots, p_n$ such that:
- $p_1 = q$, $p_n = p$
- $p_{i+1}$ is directly density-reachable from $p_i$ for all $i$

This is the **transitive closure** of directly density-reachable. Density-reachability is **asymmetric** - if $p$ is density-reachable from $q$, $q$ might not be density-reachable from $p$ (if $p$ is not a core object).

**Density-connected**: A point $p$ is density-connected to $q$ if there exists a point $o$ such that both $p$ and $q$ are density-reachable from $o$.

Density-connectivity is **symmetric** - if $p$ is density-connected to $q$, then $q$ is density-connected to $p$.

### Density-Based Cluster Definition

A **density-based cluster** is a non-empty subset $S$ of the database $D$ that satisfies:

1. **Maximality**: If $p \in S$ and $q$ is density-reachable from $p$, then $q \in S$. This ensures we include all reachable points.

2. **Connectivity**: Each object in $S$ is density-connected to all other objects in $S$. This ensures the cluster is a single connected component.

A **density-based clustering** of database $D$ is a partition $\{S_1, \ldots, S_n; N\}$ where:
- $S_1, \ldots, S_n$ are all density-based clusters
- $N = D \setminus \{S_1, \ldots, S_n\}$ is the **noise** - objects not belonging to any cluster

### Types of Points

DBSCAN classifies points into three types:
- **Core points**: Have at least MinPts points in their $\varepsilon$-neighborhood
- **Border points**: Are within the $\varepsilon$-neighborhood of a core point but don't have enough neighbors themselves
- **Noise points**: Are neither core points nor border points - they don't belong to any cluster

### The DBSCAN Algorithm

The key insight: Each object in a density-based cluster $C$ is density-reachable from **any** of its core objects, and nothing else is density-reachable from those core objects.

```
DBSCAN(D, ε, MinPts):
    for each point o in D do
        if o is not yet classified then
            if o is a core-object then
                collect all objects density-reachable from o
                assign them to a new cluster
            else
                assign o to NOISE
```

The algorithm works by:
1. Starting from an unclassified point
2. If it's a core object, finding all density-reachable points (by performing successive $\varepsilon$-neighborhood queries)
3. Assigning all reachable points to the same cluster
4. Repeating until all points are classified

Border points can be assigned to one or multiple clusters (depending on implementation - some assign them to the first cluster found, others to all clusters they're reachable from).

### Example

Consider points with $\varepsilon = 2.0$ and MinPts = 3. The algorithm:
1. Finds core objects (points with at least 3 neighbors within distance 2.0)
2. For each core object, collects all density-reachable points
3. Forms clusters from these collections
4. Marks remaining points as noise

### Performance

The runtime complexity depends on how efficiently we can answer $\varepsilon$-neighborhood queries:

- **Without support (worst case)**: $O(n)$ per query → $O(n^2)$ total
- **With tree-based support** (e.g., R*-tree): $O(\log n)$ per query → $O(n \log n)$ total
- **With direct neighborhood access**: $O(1)$ per query → $O(n)$ total

When dimensionality is high, DBSCAN is typically $O(n^2)$ because spatial index structures become less effective. However, with proper indexing, DBSCAN can be faster than CLARANS for many datasets.

### Tuning Parameters $\varepsilon$ and MinPts

Choosing good parameters is crucial but challenging. The idea is to use the point density of the least dense cluster as parameters.

**Heuristic method using k-distance**:

1. **Fix MinPts**: Default is $2 \times d - 1$ where $d$ is the dimension of the data space

2. **k-distance function**: For a point $p$, $k$-distance$(p)$ is the distance from $p$ to its $k$-th nearest neighbor

3. **k-distance plot**: Sort all $k$-distances in decreasing order and plot them

4. **Find the "border object"**: Look for the first "valley" in the plot - this represents the transition from dense to sparse regions

5. **Set $\varepsilon$**: Set $\varepsilon$ to the $k$-distance of the border object

The intuition: In a k-distance plot, points in dense clusters will have small $k$-distances, while points in sparse regions (or noise) will have large $k$-distances. The "valley" marks the boundary between these regions.

**Problematic cases**: When clusters have very different densities, the k-distance plot may have multiple valleys, making it hard to choose a single $\varepsilon$ value. This is a fundamental limitation of DBSCAN.

### Advantages

- **No need to specify $k$**: The number of clusters is determined automatically
- **Arbitrary shapes**: Can find clusters with any shape, not just convex ones
- **Robust to outliers**: Outliers are naturally identified as noise
- **Handles noise explicitly**: Noise points are explicitly separated from clusters

### Disadvantages

- **Parameter sensitivity**: Choosing appropriate $\varepsilon$ and MinPts requires domain knowledge and can be difficult
- **Different densities**: If clusters have very different densities, DBSCAN struggles. It can't handle both dense and sparse clusters well with a single $\varepsilon$ value
- **High-dimensional data**: Performance degrades in high dimensions

---

## DENCLUE

DENCLUE (DENsity-based CLUstering) generalizes DBSCAN by using **kernel density estimation** instead of the simple discrete neighborhood model. This provides a smoother, more flexible notion of density.

### Motivation

DBSCAN uses a relatively simple density model:
- Points within $\varepsilon$ contribute fully
- Points outside $\varepsilon$ don't contribute at all

This creates a "hard cutoff" that can be problematic. DENCLUE uses **density estimation** techniques to create a smoother model where all points contribute, but with different weights based on distance.

### Density Estimation

**Density estimation** is a non-parametric technique that:
- Determines an unknown probability density function
- Doesn't assume a fixed probability model
- Estimates the probability density at each point in the dataset

DBSCAN actually uses a simplified version of density estimation (the discrete kernel). DENCLUE uses more sophisticated kernels, particularly the Gaussian kernel.

### Univariate Density Estimation

For one-dimensional data, we model the data as a random variable $X$ with observations $x_1, x_2, \ldots, x_n$.

We can estimate the **cumulative distribution function** by counting:
$$\hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n} I(x_i \leq x)$$

where $I$ is an indicator function (1 if true, 0 otherwise).

The **density function** is estimated by taking the derivative. For a window of width $h$ centered at $x$:
$$\hat{f}(x) = \frac{k/n}{h} = \frac{k}{nh}$$

where $k$ is the number of points in the window. The density estimate is the ratio of points in the window to the volume of the window.

### Kernel Estimator

Instead of a fixed window, we use a **kernel function** $K$ that:
- Is non-negative: $K(x) \geq 0$
- Is symmetric: $K(-x) = K(x)$
- Integrates to 1: $\int K(x) dx = 1$

The kernel is essentially a probability distribution function. The density estimator becomes:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where $h$ is the **bandwidth** parameter (like the window width).

**Discrete kernel** (used by DBSCAN):
$$K(z) = \begin{cases} 1 & \text{if } |z| \leq 1/2 \\ 0 & \text{otherwise} \end{cases}$$

This creates a non-smooth, step-like density estimate.

**Gaussian kernel** (used by DENCLUE):
$$K(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)$$

This yields:
$$K\left(\frac{x - x_i}{h}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x - x_i)^2}{2h^2}\right)$$

The Gaussian kernel creates a **smooth** density estimate where:
- Points close to $x$ contribute more
- Points far from $x$ contribute less (but still contribute)
- $x$ plays the role of the mean, $h$ plays the role of the standard deviation

### Multivariate Density Estimation

For $d$-dimensional data $\mathbf{x} = (x_1, x_2, \ldots, x_d)$, the window becomes a hypercube with edge length $h$ and volume $h^d$.

The density estimator is:
$$\hat{f}(\mathbf{x}) = \frac{1}{nh^d} \sum_{i=1}^{n} K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$

where the multivariate kernel satisfies $\int K(\mathbf{z}) d\mathbf{z} = 1$.

**Multivariate Gaussian kernel** (assuming identity covariance matrix):
$$K(\mathbf{z}) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{\mathbf{z}^T\mathbf{z}}{2}\right)$$

### Density Attractors

A **density attractor** $\mathbf{x}^*$ is a local maximum of the probability density distribution. These are the "peaks" in the density landscape - natural cluster centers.

DENCLUE finds attractors using a **gradient-based approach**. The gradient tells us how the density changes as we move in different directions.

For the Gaussian kernel, the gradient at point $\mathbf{x}$ is:

$$\nabla \hat{f}(\mathbf{x}) = \frac{1}{nh^{d+2}} \sum_{i=1}^{n} K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right) \cdot (\mathbf{x}_i - \mathbf{x})$$

This is a weighted sum where:
- Each point $\mathbf{x}_i$ contributes a vector $(\mathbf{x}_i - \mathbf{x})$ pointing from $\mathbf{x}$ toward $\mathbf{x}_i$
- The weight is $K((\mathbf{x} - \mathbf{x}_i)/h)$ - points closer to $\mathbf{x}$ have more influence
- The further $\mathbf{x}_i$ is from $\mathbf{x}$, the less influence it has

**Hill-climbing**: Starting at a point $\mathbf{x}$, we can follow the gradient to find a local maximum:
$$\mathbf{x}^{t+1} = \mathbf{x}^t + \eta \nabla \hat{f}(\mathbf{x}^t)$$

where $\eta$ is a step size. This converges to a density attractor $\mathbf{x}^*$.

**Alternative**: Instead of hill-climbing, we can solve for $\nabla \hat{f}(\mathbf{x}) = 0$:

$$\mathbf{x} = \frac{\sum_{i=1}^{n} K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right) \mathbf{x}_i}{\sum_{i=1}^{n} K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)}$$

Since $\mathbf{x}$ appears on both sides, we use this as an iterative update rule.

### Density-Based Cluster Definition

A set $C$ of data points is a **density-based cluster** with respect to density threshold $\xi$ if there exists a set of density attractors $\mathbf{x}_1^*, \mathbf{x}_2^*, \ldots, \mathbf{x}_m^*$ such that:

1. Each point $\mathbf{x}$ in $C$ is attracted to some attractor $\mathbf{x}_i^*$ (follows the gradient to that attractor)

2. Each density attractor $\mathbf{x}_i^*$ exceeds the density threshold: $\hat{f}(\mathbf{x}_i^*) \geq \xi$

3. Any two density attractors $\mathbf{x}_i^*$ and $\mathbf{x}_j^*$ are density-reachable: there exists a path from $\mathbf{x}_i^*$ to $\mathbf{x}_j^*$ such that for all points $\mathbf{y}$ on the path, $\hat{f}(\mathbf{y}) \geq \xi$

This extends DBSCAN's notion: points are either dense (above threshold) or within the influence of dense points, and any two points are connected through a path of dense points.

### The DENCLUE Algorithm

```
DENCLUE(D, h, ξ):
    For each point x in D:
        Find density attractor x* that x converges to
        If f(x*) ≥ ξ:
            Add x* to set of attractors (if not already there)
            Add x to the cluster of x*
    Merge clusters whose attractors are density-reachable
    Return clusters
```

**Complexity**: $O(n^2 t)$ where $n$ is the number of points and $t$ is the number of iterations to find attractors.

### Advantages and Disadvantages

**Advantages**:
- **Smooth density model**: Gaussian kernel provides smooth, continuous density estimates
- **Flexible**: Can handle clusters with arbitrary shapes
- **Automatic cluster count**: Number of clusters determined automatically
- **Generalizes DBSCAN**: DBSCAN is a special case (discrete kernel)

**Disadvantages**:
- **Parameter sensitivity**: Requires choosing bandwidth $h$ and density threshold $\xi$
- **Computational cost**: More expensive than DBSCAN due to kernel computations
- **High-dimensional data**: Performance issues in high dimensions

---

## Clustering Evaluation

How do we know if our clustering is good? Evaluation measures help us assess clustering quality and compare different clusterings.

### Internal vs External Measures

**Internal measures**: Based only on the clustering data itself (distances, densities, etc.). Examples:
- Total distance (TD) from k-means
- Silhouette coefficient (from Lecture 2)

**External measures**: Use external information not part of the clustering data, typically ground-truth labels. Examples:
- Purity
- F-measure
- Normalized Mutual Information (NMI)

**Relative measures**: Compare two clusterings instead of giving an absolute "goodness" value. Some measures (like silhouette) can be used both ways.

### External Measures (Using Ground Truth)

When we have a **correct** or **ground-truth clustering** known a priori:
- $y_i \in \{1, 2, \ldots, k\}$: Ground-truth cluster membership for point $\mathbf{x}_i$
- $\mathcal{T} = \{T_1, T_2, \ldots, T_k\}$: Ground-truth clustering where $T_j = \{\mathbf{x}_i \in D \mid y_i = j\}$
- $\mathcal{C} = \{C_1, \ldots, C_r\}$: Our clustering with $r$ clusters
- $\hat{y}_i \in \{1, 2, \ldots, r\}$: Cluster label assigned by our algorithm

### Contingency Table

A **contingency table** (or confusion matrix) is an $r \times k$ table $\mathbf{N}$ where:
$$\mathbf{N}(i, j) = n_{ij} = |C_i \cap T_j|$$

This counts how many points are in both cluster $C_i$ (from our clustering) and ground-truth cluster $T_j$.

We also define:
- $n_i = |C_i|$: Number of points in cluster $C_i$
- $m_j = |T_j|$: Number of points in ground-truth cluster $T_j$

The contingency table can be computed in $O(n)$ time by examining each point's ground-truth and cluster labels.

### Purity

**Purity** quantifies the extent to which a cluster $C_i$ contains entities from only one ground-truth partition:

$$purity_i = \frac{1}{n_i} \max_{j=1\ldots k} \{n_{ij}\}$$

The purity of cluster $C_i$ is the fraction of points that belong to the most common ground-truth cluster.

The **purity of the clustering** $\mathcal{C}$ is the weighted sum:

$$purity = \sum_{i=1}^{r} \frac{n_i}{n} purity_i = \frac{1}{n} \sum_{i=1}^{r} \max_{j=1\ldots k} \{n_{ij}\}$$

Purity ranges from 0 to 1, with 1 indicating perfect clustering (each cluster contains only points from one ground-truth cluster).

**Limitation**: Purity doesn't penalize having too many clusters. If we put each point in its own cluster, purity = 1, but that's not useful.

### Maximum Matching

**Maximum matching** finds the best one-to-one mapping between clusters and ground-truth clusters to maximize the number of correctly matched points.

**Computation**:
1. Create a weighted bipartite graph:
   - Vertices: $\mathcal{C} \cup \mathcal{T}$ (clusters and ground-truth clusters)
   - Edges: $(C_i, T_j)$ with weight $w(C_i, T_j) = n_{ij}$

2. Find a **matching** $M$: a subset of pairwise non-adjacent edges (no vertex appears in more than one edge)

3. Find the **maximum weight matching**: the matching that maximizes $\sum_{e \in M} w(e)$

4. Compute: $$match = \frac{\sum_{e \in M} w(e)}{n}$$

This can be computed using the Hungarian algorithm for the assignment problem.

Maximum matching is more robust than purity because it finds the optimal alignment between clusters and ground truth.

### F-Measure

The **F-measure** combines precision and recall, familiar from information retrieval.

For cluster $C_i$:
- **$j_i$**: Ground-truth cluster with maximum overlap: $j_i = \arg\max_{j=1}^{k} n_{ij}$

- **Precision**: Same as purity
  $$prec_i = \frac{1}{n_i} \max_{j=1}^{k} \{n_{ij}\} = \frac{n_{ij_i}}{n_i}$$

- **Recall**: Fraction of ground-truth cluster $T_{j_i}$ that's in cluster $C_i$
  $$recall_i = \frac{n_{ij_i}}{m_{j_i}} = \frac{n_{ij_i}}{|T_{j_i}|}$$

- **F-measure**: Harmonic mean of precision and recall
  $$F_i = \frac{2 \cdot prec_i \cdot recall_i}{prec_i + recall_i} = \frac{2n_{ij_i}}{n_i + m_{j_i}}$$

The **F-measure of clustering** $\mathcal{C}$ is the mean cluster-wise F-measure:

$$F = \frac{1}{r} \sum_{i=1}^{r} F_i$$

F-measure balances precision and recall, penalizing both false positives and false negatives.

### Conditional Entropy

**Entropy** measures uncertainty. The entropy of a clustering $\mathcal{C}$ is:

$$H(\mathcal{C}) = -\sum_{i=1}^{r} p_{C_i} \log p_{C_i}$$

where $p_{C_i} = n_i/n$ is the probability of cluster $C_i$.

The **cluster-specific entropy** of ground truth $\mathcal{T}$ given cluster $C_i$ is:

$$H(\mathcal{T}|C_i) = -\sum_{j=1}^{k} \frac{n_{ij}}{n_i} \log \frac{n_{ij}}{n_i}$$

This measures how "mixed" cluster $C_i$ is with respect to ground-truth clusters.

The **conditional entropy** of $\mathcal{T}$ given $\mathcal{C}$ is:

$$H(\mathcal{T}|\mathcal{C}) = -\sum_{i=1}^{r} \sum_{j=1}^{k} \frac{n_{ij}}{n} \log \frac{n_{ij}}{n_i} = -\sum_{i=1}^{r} \sum_{j=1}^{k} p_{ij} \log \frac{p_{ij}}{p_{C_i}}$$

- **Perfect clustering**: $H(\mathcal{T}|\mathcal{C}) = 0$ (each cluster contains only one ground-truth class)
- **Worst case**: $H(\mathcal{T}|\mathcal{C}) = \log k$ (clusters are uniformly mixed)

Lower conditional entropy means better clustering.

### Mutual Information

**Mutual Information** quantifies the amount of shared information between clustering $\mathcal{C}$ and ground truth $\mathcal{T}$:

$$I(\mathcal{C}, \mathcal{T}) = \sum_{i=1}^{r} \sum_{j=1}^{k} p_{ij} \log \frac{p_{ij}}{p_{C_i} p_{T_j}}$$

This measures the dependence between the observed joint probability $p_{ij}$ and the expected joint probability $p_{C_i} p_{T_j}$ under independence.

- If $\mathcal{C}$ and $\mathcal{T}$ are independent: $p_{C_i} p_{T_j} = p_{ij}$ → $I(\mathcal{C}, \mathcal{T}) = 0$
- Higher mutual information means stronger dependence (better alignment)

**Problem**: Mutual information has no upper bound, making it hard to interpret.

**Normalized Mutual Information (NMI)**:

$$NMI(\mathcal{C}, \mathcal{T}) = \frac{I(\mathcal{C}, \mathcal{T})}{\sqrt{H(\mathcal{C}) H(\mathcal{T})}}$$

NMI ranges from 0 to 1, with 1 indicating perfect clustering.

### Discussion

**Internal measures**:
- Make no assumptions about ground truth
- Often the only option in unsupervised learning
- Examples: silhouette coefficient, total distance

**External measures**:
- Require ground-truth labels (which may not be available)
- Useful for validation and comparison
- Examples: purity, F-measure, NMI

**Important note**: Different measures capture different aspects of good clustering. There's active research on what makes a clustering "good" - for example, should a cluster be allowed to match multiple ground-truth clusters? The choice of measure depends on the application.

---

## Key Takeaways

- **Density-based clustering** defines clusters as high-density regions separated by low-density regions. This allows finding clusters of arbitrary shapes without specifying $k$ in advance.

- **DBSCAN** uses a simple discrete kernel model with two parameters ($\varepsilon$ and MinPts). It's efficient and robust to outliers, but struggles with clusters of different densities.

- **DENCLUE** generalizes DBSCAN using kernel density estimation (typically Gaussian). It provides smoother density models but is more computationally expensive.

- **Clustering evaluation** uses internal measures (based on data only) or external measures (using ground truth). Different measures capture different aspects of clustering quality.

- **External measures** like purity, F-measure, and NMI help validate clusterings when ground truth is available, but require careful interpretation.

---

## Connections

This lecture builds on:
- **Distance metrics** from Lecture 1 (used in $\varepsilon$-neighborhoods)
- **Representative-based clustering** from Lecture 2 (contrast with density-based approaches)

This lecture sets up:
- **Hierarchical clustering** (next topic) which creates clusterings at multiple resolutions
- **Outlier detection** which can use density-based ideas
- The importance of **evaluation** for validating clustering results

Density-based methods complement representative-based methods: use density-based when you don't know $k$ or have non-convex clusters; use representative-based when you know $k$ and clusters are roughly spherical.

---

## References

- Lecture slides: `03-Dens-Clus.pdf`, `0X-Clus-Eval.pdf`
- DMA book: Chapter 15 (Validation), Chapter 17.1.1, 17.1.2
- Lecture notes: Part I, Section 2 (Density-based clustering), Section 5 (Clustering evaluation)
