# Lecture 2: Representative-based Clustering

## Overview

This lecture introduces clustering as a fundamental data mining task and focuses on **representative-based clustering** approaches. These methods characterize each cluster by a representative (like a center point or medoid) and iteratively improve the clustering. We'll study three main algorithms: k-means, k-medoids, and Expectation-Maximization (EM), each with different strengths and assumptions.

---

## What is Clustering?

**Clustering** is the task of grouping a set of data objects into clusters, where a **cluster** is a collection of data objects that are:
- **Similar to one another** within the same cluster
- **Dissimilar to objects** in other clusters

Think of it as unsupervised classification - we don't have predefined classes or labels. The algorithm discovers the groups on its own.

### Why Clustering Matters

Clustering serves two main purposes:
1. **As a stand-alone tool** to get insight into data distribution - understanding what natural groups exist in your data
2. **As a preprocessing step** for other algorithms - organizing data before applying other techniques

### Applications

Clustering has widespread applications across many domains:

- **Pattern Recognition and Image Processing**: Organizing images by visual similarity
- **Spatial Data Analysis**: Creating thematic maps in GIS, detecting spatial clusters
- **Web Mining**: Document clustering (web content mining), user behavior analysis (web usage mining)
- **Biology**: Clustering gene expression data, taxonomy classification
- **Marketing**: Customer segmentation to develop targeted programs
- **Social Networks**: Finding communities of people with similar interests (we'll cover this later in the course)

### Example: Thematic Maps

A classic application is creating thematic maps from satellite imagery. Each point on the earth's surface maps to a high-dimensional feature vector where each dimension represents the intensity at different wavelengths. Different land-use types (forest, water, urban) reflect and emit light differently, so clustering in this feature space can automatically identify different land-use regions.

---

## Representative-based Clustering: Basic Concept

Representative-based (also called partitioning) approaches work by constructing a partition of a database $D$ of $n$ objects into $k$ clusters, minimizing some objective function.

### The Challenge

Exhaustively enumerating all possible partitions into $k$ sets to find the global minimum is computationally infeasible. The number of ways to partition $n$ objects into $k$ clusters grows exponentially. Instead, we use **heuristic methods** that find good (though not necessarily optimal) solutions.

### The General Approach

Representative-based algorithms follow this iterative pattern:

1. **Choose k representatives** for clusters (often randomly)
2. **Improve iteratively**:
   - Assign each object to the cluster it "fits best" in the current clustering
   - Compute new cluster representatives based on these assignments
   - Repeat until the change in the objective function drops below a threshold

### Types of Representatives

The key difference between algorithms is what they use as cluster representatives:

- **k-means**: Each cluster is represented by the **center** (mean/centroid) of the cluster
- **k-medoid**: Each cluster is represented by **one of its actual objects** (a medoid)

This distinction might seem minor, but it has important implications for what distance functions work, how robust the algorithm is to outliers, and what data types it can handle.

---

## K-Means Clustering

K-means is probably the most well-known clustering algorithm. It's simple, efficient, and works well when clusters are roughly spherical and well-separated.

### The Objective

For a given $k$, form $k$ groups so that the **sum of squared distances** between the mean of the groups (centroids) and their elements is minimal.

The idea is that we want clusters to be "tight" - all points should be close to their cluster's center. The squared distance penalizes points that are far from the center more heavily than linear distance would.

### Basic Notions

Objects $\mathbf{x} = (x_1, \ldots, x_d)$ are points in a $d$-dimensional vector space. We need the mean to be defined, which is why k-means works with numerical data.

The **centroid** $\mu_C$ of a cluster $C$ is the mean of all points in that cluster:

$$\mu_C = \frac{1}{|C|} \sum_{\mathbf{x}_j \in C} \mathbf{x}_j$$

The **total distance** (compactness measure) for a cluster $C_i$ is:

$$TD(C_i) = \sum_{\mathbf{x}_j \in C_i} dist(\mathbf{x}_j, \mu_{C_i})^2$$

And for the entire clustering with $k$ clusters:

$$TD = \sum_{i=1}^{k} TD(C_i) = \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in C_i} dist(\mathbf{x}_j, \mu_{C_i})^2$$

The ideal clustering minimizes this objective function. Notice we're using squared Euclidean distance here.

### Lloyd's Algorithm

The standard k-means algorithm is called **Lloyd's algorithm**. Given $k$, it performs these steps:

**Initialization**: Partition the objects into $k$ nonempty subsets (often randomly)

**Iterate until convergence**:

1. **Centroid Update**: Compute the centroids of the clusters of the current partition. The centroid is the center (mean point) of the cluster.

2. **Cluster Assignment**: Assign each object to the cluster with the nearest representative (centroid).

The algorithm repeats until representatives don't change substantially (or the change in the objective function is below a threshold).

### Why This Works

Each step improves (or at least doesn't worsen) the objective function:
- **Centroid update**: For a fixed assignment, the mean minimizes the sum of squared distances (this is a mathematical fact)
- **Cluster assignment**: For fixed centroids, assigning each point to its nearest centroid minimizes the objective

Since the objective is bounded below (it can't be negative), and each step improves it, the algorithm must converge. However, it might converge to a **local optimum** rather than the global optimum - the quality depends heavily on initialization.

### Pseudocode

```
k-means(D, k):
    Initialize: Randomly assign points to k clusters
    repeat
        // Centroid update
        for each cluster C_i:
            μ_i = mean of all points in C_i
        
        // Cluster assignment
        for each point x_j:
            assign x_j to cluster with nearest centroid
    until centroids don't change (or change < threshold)
    return clusters
```

### Example: Iris Dataset

A classic example uses the Iris dataset with 3 species. If we set $k=3$ and run k-means, we typically get clusters that correspond well to the three species. The algorithm starts with random centroids, assigns points to nearest centroids, updates centroids, and repeats until stable.

### Advantages

- **Relatively efficient**: Time complexity is $O(tkn)$ where $n$ is the number of objects, $k$ is the number of clusters, and $t$ is the number of iterations. Typically $k, t \ll n$, so this is quite fast.
- **Easy to implement**: The algorithm is straightforward
- **Works well** when clusters are roughly spherical and well-separated

### Disadvantages

- **Requires mean to be defined**: Only works with numerical data in Euclidean space
- **Need to specify $k$**: We must know (or guess) the number of clusters in advance
- **Sensitive to outliers**: A single outlier can dramatically affect the centroid
- **Convex clusters only**: Clusters are forced to have convex shapes (can't handle non-convex clusters like rings or crescents)
- **Local optima**: Result depends heavily on initialization; often terminates at a local optimum rather than global optimum

### Initialization Strategies

Since initialization matters so much, several strategies exist:

**Multiple random initializations**: Run k-means multiple times with different random starts and pick the best result.

**Bradley-Fayyad method**: 
1. Draw $m$ different small samples from the dataset
2. Cluster each sample to get $m$ estimates of $k$ representatives
3. Cluster the union of samples $m$ times, using each estimate as initialization
4. Use the best of these as initialization for the full dataset

**k-means++**: A smarter initialization that tries to spread out initial centroids:
1. Randomly choose the first center
2. Pick each subsequent center with probability proportional to the squared distance to the nearest already-chosen center
3. This encourages choosing centers that are far apart

This provides a good approximation and helps avoid arbitrarily bad local minima.

---

## Choosing the Parameter k

One of the biggest challenges with k-means is choosing $k$. We can't just try all values and pick the one with minimum $TD$ - that measure decreases monotonically as $k$ increases (more clusters means each cluster is tighter). We need a measure that's independent of $k$.

### The Silhouette Coefficient

The **silhouette coefficient** is a measure of clustering quality that's independent of $k$. It evaluates how well objects are assigned to their clusters.

For an object $o$:
- **$int(o)$**: Average distance between object $o$ and other objects in its cluster $C(o)$
  $$int(o) = \frac{1}{|C(o)|} \sum_{\mathbf{x} \in C(o), \mathbf{x} \neq o} dist(o, \mathbf{x})$$
  
  This measures how well $o$ fits in its cluster - smaller is better.

- **$ext(o)$**: Average distance between object $o$ and objects in its "second closest" cluster (the nearest cluster it's not in)
  $$ext(o) = \min_{C_i \neq C(o)} \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} dist(o, \mathbf{x})$$
  
  This measures how far $o$ is from other clusters - larger is better.

The **silhouette** of object $o$ is:

$$s(o) = \frac{ext(o) - int(o)}{\max(int(o), ext(o))}$$

This normalizes the difference, giving values between -1 and +1:
- **$s(o) = -1$**: Bad assignment - on average closer to members of another cluster
- **$s(o) = 0$**: Ambiguous - equally close to current cluster and another
- **$s(o) = +1$**: Good assignment - much closer to current cluster than any other

The **silhouette coefficient** $s_C$ of a clustering is the average silhouette of all objects. Interpretation:
- $0.7 < s_C \leq 1.0$: Strong structure
- $0.5 < s_C \leq 0.7$: Medium structure  
- $0.25 < s_C \leq 0.5$: Weak structure
- $s_C \leq 0.25$: No structure

We can compute the silhouette coefficient for different values of $k$ and choose the $k$ that gives the highest value.

---

## Kernel K-Means

Standard k-means assumes clusters are separated by **linear boundaries** (lines in 2D, planes in 3D, hyperplanes in higher dimensions). What if clusters have non-linear boundaries?

### The Linearity Problem

K-means creates clusters with linear boundaries. If your data looks like concentric circles or has other non-linear structure, k-means will fail to find the natural clusters.

### The Kernel Trick

**Kernel k-means** uses the kernel trick (familiar from kernel methods) to project points into a different space where they might be linearly separable, then applies k-means in that space.

Recall that a **kernel** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$ is a dot product in some (possibly infinite-dimensional) feature space, but we never need to explicitly compute $\phi(\mathbf{x})$ - we just compute the kernel.

### The Objective

The k-means objective in the feature space is:

$$\min_C \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in C_i} ||\phi(\mathbf{x}_j) - \phi(\mu_i)||^2$$

where $\phi(\mu_i)$ is the centroid in feature space. We can rewrite this entirely in terms of kernels:

$$\sum_{j=1}^{n} K(\mathbf{x}_j, \mathbf{x}_j) - \sum_{i=1}^{k} \frac{1}{n_i} \sum_{\mathbf{x}_a \in C_i} \sum_{\mathbf{x}_b \in C_i} K(\mathbf{x}_a, \mathbf{x}_b)$$

The key insight: we never need to compute $\phi(\mathbf{x})$ explicitly - we can compute everything using just the kernel function.

### Cluster Distance in Feature Space

To assign a point to a cluster, we compute the distance in feature space:

$$||\phi(\mathbf{x}_j) - \phi(\mu_i)||^2 = K(\mathbf{x}_j, \mathbf{x}_j) - \frac{2}{n_i} \sum_{\mathbf{x}_a \in C_i} K(\mathbf{x}_a, \mathbf{x}_j) + \frac{1}{n_i^2} \sum_{\mathbf{x}_a \in C_i} \sum_{\mathbf{x}_b \in C_i} K(\mathbf{x}_a, \mathbf{x}_b)$$

We assign $\mathbf{x}_j$ to the cluster that minimizes this distance.

### Advantages and Disadvantages

**Advantages**:
- Allows detection of clusters with **arbitrary shapes** (depending on the kernel)
- Works with any kernel function (Gaussian/RBF, polynomial, etc.)

**Disadvantages**:
- **Inefficient**: Requires computing an $n \times n$ kernel matrix, giving $O(n^2k)$ complexity instead of $O(nkd)$
- Still need to specify $k$ in advance
- Requires choosing a kernel and its parameters
- Clusters are still forced to have convex shapes in the feature space

### When to Use

Use kernel k-means when you suspect non-linear cluster boundaries but still want a partitioning approach. Common kernels include the Gaussian/RBF kernel for smooth, non-linear boundaries.

---

## K-Medoid Clustering

K-medoid addresses some limitations of k-means by using actual data points as representatives instead of computed centroids.

### Motivation

K-means has several limitations:
- Assumes Euclidean distance (minimizing distance to mean)
- Requires mean to be defined (only works with numerical data)
- Sensitive to outliers

**K-medoid** is more general:
- Works with **any distance function** (not just Euclidean)
- Works in spaces where mean is not defined (e.g., categorical data, graphs)
- More **robust to noise** because medoids are actual data points

### The Objective

Find $k$ representatives (medoids) in the dataset so that the **sum of distances** between representatives and objects assigned to them is minimal.

A **medoid** is a representative object "in the middle" of a cluster - think of it like the median, but for a cluster representative.

### Basic Notions

K-medoid requires:
- Arbitrary objects (not necessarily numerical vectors)
- A distance function (any distance function, not just Euclidean)

For a cluster $C$ with medoid $m_C$, the compactness is:

$$TD(C) = \sum_{\mathbf{x} \in C} dist(\mathbf{x}, m_C)$$

And for the entire clustering:

$$TD = \sum_{i=1}^{k} TD(C_i) = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} dist(\mathbf{x}, m_{C_i})$$

Notice we're using distance (not squared distance) - this is often Manhattan distance ($L_1$ norm) rather than Euclidean ($L_2$).

### PAM Algorithm

**Partitioning Around Medoids (PAM)** is the standard k-medoid algorithm:

1. **Select** $k$ objects arbitrarily as medoids; assign each remaining object to the cluster with the nearest medoid; compute $TD_{current}$.

2. **For each pair** (medoid $M$, non-medoid $N$):
   - Compute $TD_{N \leftrightarrow M}$ (the total distance if we swap $M$ with $N$)

3. **Select** the non-medoid $N$ for which $TD_{N \leftrightarrow M}$ is minimum

4. **If** $TD_{N \leftrightarrow M} < TD_{current}$:
   - Swap $N$ with $M$
   - Set $TD_{current} := TD_{N \leftrightarrow M}$
   - Go back to Step 2

5. **Else**: Stop

The algorithm tries swapping medoids with non-medoids and keeps swaps that improve the objective. This is more expensive than k-means because we need to evaluate many swaps.

### CLARA and CLARANS

Since PAM is expensive ($O(k(n-k)^2)$ per iteration), faster variants exist:

**CLARA (Clustering LARge Applications)**:
- Draws `numlocal` samples of the dataset
- Applies PAM on each sample
- Returns the best set of medoids found

**CLARANS (Clustering Large Applications based on RANdomized Search)**:
- Instead of evaluating all possible swaps, randomly samples at most `maxneighbor` pairs
- Swaps the first pair that improves the objective (greedy)
- Repeats `numlocal` times and returns the best result

Efficiency: `runtime(CLARANS) < runtime(CLARA) < runtime(PAM)`

### Advantages and Disadvantages

**Advantages**:
- Applicable to **arbitrary objects** + distance function (not just numerical vectors)
- **More robust** to noisy data and outliers than k-means
- Works with any distance metric

**Disadvantages**:
- **Inefficient** compared to k-means
- Still need to specify $k$ in advance
- Clusters still forced to have convex shapes
- Results may vary due to randomization (for CLARA/CLARANS)

---

## Expectation-Maximization (EM) Clustering

EM clustering uses a **probabilistic model** where each cluster is represented by a probability distribution, typically a Gaussian. This allows for **overlapping clusters** - points can belong to multiple clusters with different probabilities.

### Basic Notions

Instead of hard assignments (each point belongs to exactly one cluster), EM uses **soft assignments** - each point has a probability of belonging to each cluster.

Each cluster $C_i$ is represented by a **Gaussian distribution**:
- **Center point** $\mu_i$: The mean of the cluster
- **Covariance matrix** $\Sigma_i$: A $d \times d$ matrix describing how points are spread around the center

The **density function** for cluster $C_i$ is:

$$f(\mathbf{x} | \mu_i, \Sigma_i) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_i|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \mu_i)^T \Sigma_i^{-1} (\mathbf{x} - \mu_i)\right)$$

This is the multivariate normal distribution $\mathcal{N}(\mu_i, \Sigma_i)$.

### Gaussian Mixture Model

The overall density for the clustering is a **mixture** of $k$ Gaussians:

$$f(\mathbf{x}) = \sum_{i=1}^{k} f(\mathbf{x} | \mu_i, \Sigma_i) P(C_i)$$

where $\sum_{i=1}^{k} P(C_i) = 1$ (the prior probabilities sum to 1).

This means each point is generated by first choosing a cluster $C_i$ with probability $P(C_i)$, then sampling from that cluster's Gaussian distribution.

### The Problem: Maximum Likelihood Estimation

We want to find the parameters $\theta_i = (\mu_i, \Sigma_i, P(C_i))$ that maximize the likelihood of the data:

$$\arg\max_{(\theta_1, \ldots, \theta_k)} \ln P(D | \theta_1, \ldots, \theta_k)$$

where:

$$\ln P(D | \theta_1, \ldots, \theta_k) = \sum_{j=1}^{n} \ln f(\mathbf{x}_j) = \sum_{j=1}^{n} \ln \left(\sum_{i=1}^{k} f(\mathbf{x}_j | \mu_i, \Sigma_i) P(C_i)\right)$$

This is **hard to maximize directly** because of the sum inside the logarithm. If we knew which cluster each point came from, it would be easy, but we don't.

### The EM Algorithm

EM solves this by iterating between two steps:

**Expectation Step (E-step)**: Given current parameters, compute the probability that each point belongs to each cluster (the posterior probabilities).

**Maximization Step (M-step)**: Given these probabilities, update the parameters to maximize the expected log-likelihood.

### Expectation Step

If we had the parameters $\theta_1, \ldots, \theta_k$, we could compute the probability that point $\mathbf{x}_j$ belongs to cluster $C_i$ using Bayes' theorem:

$$P(C_i | \mathbf{x}_j) = \frac{P(\mathbf{x}_j | C_i) P(C_i)}{P(\mathbf{x}_j)} = \frac{P(\mathbf{x}_j | C_i) P(C_i)}{\sum_{a=1}^{k} P(\mathbf{x}_j | C_a) P(C_a)}$$

We call these probabilities **weights** $w_{ij} = P(C_i | \mathbf{x}_j)$.

The probability $P(\mathbf{x}_j | C_i)$ can be approximated using the density function:
$$P(\mathbf{x}_j | C_i) \approx 2\epsilon \cdot f(\mathbf{x}_j | \mu_i, \Sigma_i)$$

for a small interval $\epsilon$ around the point.

### Maximization Step

Given the weights $w_{ij}$, we update the parameters:

**Center $\mu_i$ of cluster $C_i$**:
$$\mu_i = \frac{\sum_{j=1}^{n} \mathbf{x}_j \cdot w_{ij}}{\sum_{j=1}^{n} w_{ij}}$$

This is a weighted average - points contribute proportionally to how much they belong to this cluster.

**Covariance matrix $\Sigma_i$ of cluster $C_i$**:
$$\Sigma_i = \frac{\sum_{j=1}^{n} w_{ij} (\mathbf{x}_j - \mu_i)(\mathbf{x}_j - \mu_i)^T}{\sum_{j=1}^{n} w_{ij}}$$

This is the weighted covariance - again, points contribute based on their membership probability.

**Prior probability $P(C_i)$**:
$$P(C_i) = \frac{1}{n} \sum_{j=1}^{n} w_{ij} = \frac{1}{n} \sum_{j=1}^{n} P(C_i | \mathbf{x}_j)$$

This is the average probability that points belong to cluster $C_i$.

### The Algorithm

```
EM-Clustering(D, k):
    Generate initial model M' = (C_1', ..., C_k')
    repeat
        // Expectation step
        For each object x_j and cluster C_i:
            Compute w_ij = P(C_i | x_j)
        
        // Maximization step
        For each cluster C_i:
            Compute new μ_i, Σ_i, P(C_i) using weights w_ij
        
        M' ← M
    until ||μ_t - μ_{t-1}|| ≤ ε
    return M
```

### Covariance Matrix Choices

The covariance matrix $\Sigma_i$ can be:

**Full covariance matrix**: 
- Most detailed model
- Requires estimating $O(d^2)$ parameters per cluster
- Often not enough data for reliable estimation
- Computationally costly

**Diagonal covariance matrix**:
- Simplified model assuming all dimensions are independent
- Only requires estimating $d$ parameters per cluster
- More practical when data is limited
- Less flexible but more stable

When the covariance matrix is the identity matrix (diagonal with all 1s), EM clustering becomes equivalent to k-means clustering (with some additional assumptions).

### Advantages and Disadvantages

**Advantages**:
- **Flexible and powerful** probabilistic model
- **Captures overlapping clusters** - points can belong to multiple clusters
- Can model clusters with different shapes and orientations (via covariance matrices)

**Disadvantages**:
- **Converges to local minimum** (like k-means)
- **Computational effort**: $O(n \cdot k \cdot \text{#iterations})$, and number of iterations can be quite high
- Result and runtime strongly depend on:
  - Initial assignment
  - Choice of parameter $k$
- To get a hard partitioning, assign each object to the cluster with highest probability

---

## Key Takeaways

- **Clustering** groups objects so that similar objects are together and dissimilar objects are apart. It's unsupervised - we don't have predefined classes.

- **Representative-based clustering** characterizes each cluster by a representative (centroid or medoid) and iteratively improves the clustering by alternating between assigning points and updating representatives.

- **K-means** uses centroids (means) as representatives. It's efficient and works well for spherical clusters, but requires numerical data, is sensitive to outliers, and needs $k$ specified in advance.

- **K-medoid** uses actual data points as representatives. It's more general (works with any distance function), more robust to outliers, but less efficient than k-means.

- **Kernel k-means** extends k-means to handle non-linear cluster boundaries by using kernels to project data into a feature space, but at the cost of $O(n^2)$ complexity.

- **EM clustering** uses probabilistic models (Gaussian mixtures) with soft assignments. It can capture overlapping clusters and different cluster shapes, but is computationally more expensive and still sensitive to initialization.

- **Silhouette coefficient** provides a way to evaluate clustering quality and choose $k$ that's independent of the number of clusters.

---

## Connections

This lecture builds on:
- **Distance metrics** from Lecture 1 (Euclidean, Manhattan, etc.)
- **Statistical concepts** like mean, variance, and covariance from Lecture 1

This lecture sets up:
- **Density-based clustering** (next lecture) which doesn't require specifying $k$ and can find non-convex clusters
- **Hierarchical clustering** which creates a tree of clusterings
- **Clustering evaluation** methods for assessing quality

The representative-based approaches here assume we know $k$ and create a flat partition. Later methods will relax these assumptions.

---

## References

- Lecture slides: `02-Rep-Clus.pdf`
- DMA book: Chapter 13.3.3 (k-means), Chapter 13.3.4 (EM details, optional), Chapter 5 (kernel methods, optional)
- Lecture notes: Part I, Section 1 (Representative-based clustering)
