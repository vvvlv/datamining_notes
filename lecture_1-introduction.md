# Lecture 1: Introduction

## Overview

This first lecture introduces the fundamental concepts of data mining, establishes how it differs from machine learning, outlines the data mining process, and covers essential statistical tools for exploratory data analysis. These foundations are crucial for understanding the algorithms and techniques we'll explore throughout the course.

---

## What is Data Mining?

Data mining is the process of discovering patterns and models in large amounts of data that are **valid**, **useful**, **unexpected**, and **understandable**. 

Think of it this way: we're drowning in data but starving for knowledge. Data mining helps us extract that knowledge. With automated data collection tools and mature database technology, we now have tremendous amounts of data stored in databases, data warehouses, and other repositories. The challenge is making sense of it all.

### Key Characteristics of Data Mining Patterns

For a discovered pattern to be meaningful, it should be:

- **Valid**: The pattern should hold on new data with some certainty. It's not just a fluke of the specific dataset we're looking at.
- **Useful**: We should be able to act on the pattern. If we can't do anything with it, what's the point?
- **Unexpected**: The pattern should be non-obvious. If it's something everyone already knows, we haven't learned much.
- **Understandable**: Humans should be able to interpret the pattern. A black box that spits out numbers isn't helpful if we can't explain what it means.

### What Data Mining is NOT

It's important to distinguish data mining from related but different activities:

- **Deductive query processing**: This is about retrieving specific information you already know exists. Data mining is about discovering things you didn't know were there.
- **Expert systems or small ML/statistical programs**: These are typically focused on specific, well-defined tasks. Data mining is broader and more exploratory.

### Data Mining vs Machine Learning

This is a common point of confusion. While there's significant overlap, here's the key distinction:

**Machine Learning** is primarily concerned with building models that can make predictions or decisions. It's often supervised (you have labeled training data) and focuses on learning a function that maps inputs to outputs. The emphasis is on generalization: can the model perform well on new, unseen data?

**Data Mining** is broader and more exploratory. It includes:
- Unsupervised learning (clustering, outlier detection)
- Pattern discovery (association rules, frequent patterns)
- Exploratory data analysis
- Finding interesting structures in data without necessarily building predictive models

In practice, data mining often uses machine learning techniques, but the goals are different. Data mining asks "what interesting patterns are in this data?" while machine learning asks "can I build a model that predicts well?"

The course focuses on **unsupervised approaches**: clustering, outlier detection, graph mining, and frequent pattern mining. These are core data mining tasks where we're exploring data structure rather than building predictive models.

---

## The Data Mining Process: KDD

The Knowledge Discovery in Databases (KDD) process provides a framework for how data mining fits into the larger picture of extracting knowledge from data. Here's how it works:

```
Databases/Information Repositories
    ↓
Data Warehouse (Data Cleaning, Data Integration)
    ↓
Task-relevant Data (Selections, Projections, Transformations)
    ↓
Data Mining (Application of algorithms to find patterns)
    ↓
Patterns
    ↓
Knowledge (Visualization, Evaluation)
```

### Steps in the KDD Process

1. **Data Selection**: Choose the relevant data for your task. You don't need everything - just what matters.

2. **Data Cleaning and Integration**: Real-world data is messy. This step involves:
   - Removing errors and inconsistencies
   - Handling missing values
   - Integrating data from multiple sources
   - Standardizing formats

3. **Data Transformation**: Prepare the data for mining:
   - Feature selection (choosing which attributes to use)
   - Dimensionality reduction
   - Normalization or standardization
   - Creating derived attributes

4. **Data Mining**: This is the core step where we apply algorithms to discover patterns. This is what we'll spend most of the course learning about.

5. **Pattern Evaluation**: Not all discovered patterns are interesting or useful. We need to evaluate them based on:
   - Statistical significance
   - Interestingness measures
   - Actionability

6. **Knowledge Presentation**: Visualize and present the findings in a way that humans can understand and use.

### The Central Role of Data Mining

Data mining is a central step in the KDD process, but it's not the whole process. The preprocessing steps (cleaning, integration, transformation) are crucial - garbage in, garbage out. Similarly, the post-processing steps (evaluation, visualization) are essential for making the discovered patterns actually useful.

---

## Basic Data Mining Tasks

The course covers several fundamental data mining tasks:

### Clustering

**Clustering** is an unsupervised learning task where we group objects into subgroups (clusters) based on similarity. The key idea is that objects within a cluster should be similar to each other, and objects in different clusters should be dissimilar.

- **Class labels are unknown**: Unlike classification, we don't have predefined categories.
- **Similarity function**: We need a way to measure how similar two objects are (or how dissimilar - a distance function).
- **Objective**: Maximize intra-cluster similarity and minimize inter-cluster similarity.

Applications include customer segmentation, organizing document collections, analyzing web access patterns, and more. We'll cover representative-based clustering (k-means, k-medoids, EM), density-based clustering (DBSCAN, DENCLUE), and hierarchical clustering.

### Classification

**Classification** is a supervised learning task where we have labeled training data and want to build a model that can predict the class label for new objects.

- **Class labels are known**: We have examples with known categories.
- **Task**: Find models, functions, or rules that describe and distinguish classes, and can predict class membership for new objects.

While classification is important, it's more of a machine learning focus. This course emphasizes unsupervised tasks, though understanding classification helps contrast it with clustering.

### Outlier Detection

**Outliers** are objects that deviate so much from the rest of the dataset that they arouse suspicion they were generated by a different mechanism (Hawkins' definition).

Outlier detection is crucial for:
- Fraud detection (credit card, telecom)
- Quality control
- Medical analysis
- Data cleaning

In this course, we focus on **unsupervised outlier detection**, though it can also be supervised (when you have examples of outliers) or semi-supervised.

### Graph Mining

Graphs model connections between data objects - think social networks, protein interactions, road maps, or web links. **Graph mining** finds patterns in these structures:

- Communities in social networks
- Missing links (link prediction)
- Frequent subgraph patterns
- Central nodes or influential entities

Graphs are inherently high-dimensional and require specialized techniques, which we'll cover in the second part of the course.

### Association Rules and Frequent Pattern Mining

**Association rule mining** finds frequent patterns, associations, correlations, or causal structures among sets of items. The classic example is market basket analysis: which items are frequently bought together?

Rules take the form: "Body ⇒ Head [support, confidence]"

For example, from transaction data like:
- {butter, bread, milk, sugar}
- {butter, flour, milk, sugar}
- {butter, eggs, milk, salt}

We might discover:
- buys(butter) → buys(milk)
- buys(butter) → buys(sugar)

This tells us that customers who buy butter are likely to also buy milk and sugar. Applications include cross-marketing, catalog design, and recommendation systems.

---

## Statistics for Exploratory Data Analysis

Before we can mine data effectively, we need to understand it. Statistics provides the tools for **exploratory data analysis** - characterizing data attributes individually and understanding relationships between them.

### Random Variables and Distributions

A **random variable** $X$ represents a quantity that can take different values with certain probabilities. We distinguish between:

- **Discrete random variables**: Take on a countable set of values (e.g., number of items in a transaction)
- **Continuous random variables**: Take on values in a continuous range (e.g., height, weight, temperature)

The **cumulative distribution function (CDF)** $F(x) = P(X \leq x)$ tells us the probability that the random variable takes a value less than or equal to $x$. The **quantile function** is the inverse: it gives us the minimum value with a given probability $q$.

### Measures of Central Tendency

These measures tell us where the "center" of the data is, though "center" can mean different things:

#### Mean

The **mean** (or expected value) is the average value. For a random variable $X$:

$$\mu = E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

For a sample of $n$ data points, the **sample mean** is:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

The mean is the center of mass of the distribution. However, it's **not robust** - a single outlier can dramatically shift it. If you have one person with an extremely high income in your dataset, the mean income will be pulled upward.

#### Median

The **median** is the middle value when data is sorted. For a sample, if $n$ is odd, it's the middle value; if $n$ is even, it's the average of the two middle values.

The median is **robust** to outliers. That same person with extremely high income won't affect the median much. This makes it useful when you have skewed distributions or suspect outliers.

#### Mode

The **mode** is the most frequently occurring value. For continuous data, we often look at the value where the probability density function is maximum.

The mode is useful for categorical data or when you want to know the "typical" value in a different sense than mean or median.

### Measures of Dispersion

Central tendency tells us where the data is centered, but we also need to know how spread out it is:

#### Variance and Standard Deviation

The **variance** measures how much the values deviate from the mean:

$$\sigma^2 = \text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

For a sample:

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

The **standard deviation** $\sigma$ (or $s$ for samples) is the square root of variance. It has the same units as the original data, making it more interpretable.

Variance and standard deviation tell us how much the data varies. A small standard deviation means values are clustered close to the mean; a large one means they're spread out.

### Bivariate Analysis: Covariance and Correlation

When we have two attributes, we want to understand how they relate to each other.

#### Covariance

**Covariance** measures how two variables vary together:

$$\sigma_{XY} = \text{Cov}(X,Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - \mu_X \mu_Y$$

For a sample:

$$s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

- Positive covariance: When $X$ is above its mean, $Y$ tends to be above its mean too (they vary together).
- Negative covariance: When $X$ is above its mean, $Y$ tends to be below its mean (they vary in opposite directions).
- Zero covariance: No linear relationship.

#### Correlation

**Correlation** is a normalized version of covariance that ranges from -1 to 1:

$$\rho_{XY} = \frac{\sigma_{XY}}{\sigma_X \sigma_Y}$$

For a sample:

$$r_{XY} = \frac{s_{XY}}{s_X s_Y}$$

Correlation tells us the strength and direction of the linear relationship:
- $r = 1$: Perfect positive linear relationship
- $r = -1$: Perfect negative linear relationship  
- $r = 0$: No linear relationship
- Values close to $\pm 1$: Strong relationship
- Values close to 0: Weak relationship

**Important**: Correlation only measures linear relationships. Two variables can have a strong nonlinear relationship but zero correlation.

#### Geometric Interpretation

We can think of data points as vectors in space. Covariance is related to the angle between the centered vectors (vectors with their means subtracted). Correlation is the cosine of this angle. When vectors point in the same direction, correlation is high; when perpendicular, correlation is zero.

#### Covariance Matrix

For multivariate data with $d$ attributes, we organize all pairwise covariances into a **covariance matrix** $\Sigma$:

$$\Sigma = \begin{bmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1d} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{d1} & \sigma_{d2} & \cdots & \sigma_{dd}
\end{bmatrix}$$

The diagonal contains variances ($\sigma_{ii} = \sigma_i^2$), and off-diagonal elements are covariances. The matrix is symmetric ($\sigma_{ij} = \sigma_{ji}$).

### Data Normalization

Different attributes often have different scales. For example, income might be in thousands of dollars while age is in years. This can cause problems in algorithms that use distances (like clustering).

**Normalization** (or standardization) transforms data to have consistent scales. The most common approach is **z-score normalization**:

$$z_i = \frac{x_i - \bar{x}}{s}$$

This transforms data to have mean 0 and standard deviation 1. All attributes are now on the same scale, making distance calculations meaningful.

### Attribute Dependence: Contingency Analysis

For categorical attributes, we use **contingency tables** (also called cross-tabulation) to understand relationships. A contingency table shows the frequency of each combination of attribute values.

We can test for **independence** using statistical tests (like the chi-square test). The **p-value** tells us the probability of observing such a relationship if the attributes were actually independent. A small p-value (typically < 0.05) suggests the attributes are dependent.

### Distances and Similarities

Many data mining algorithms rely on measuring how similar or dissimilar objects are. For numerical data in $d$ dimensions, we have several distance metrics:

#### Minkowski Distance (Lp-norm)

For vectors $\mathbf{x} = (x_1, \ldots, x_d)$ and $\mathbf{y} = (y_1, \ldots, y_d)$:

$$L_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{d} |x_i - y_i|^p \right)^{1/p}$$

Special cases:
- **$p = 2$ (Euclidean distance)**: The straight-line distance, like measuring distance on a map. This is the most common choice.
- **$p = 1$ (Manhattan distance)**: Sum of absolute differences. Like navigating city streets where you can only move horizontally or vertically.
- **$p \to \infty$ (Maximum distance)**: Takes the maximum difference across all dimensions.

For binary vectors (where attributes are 0 or 1), we often use specialized similarity measures like Jaccard similarity, which considers the ratio of shared attributes to total attributes.

**Important**: Distance metrics assume attributes are on similar scales. This is why normalization is crucial before computing distances.

### Discretization

Sometimes we need to convert continuous attributes into discrete ones. **Equal-frequency discretization** divides the data into bins such that each bin contains approximately the same number of data points. This is useful for:
- Making certain algorithms applicable (some work only with discrete data)
- Reducing noise
- Creating categorical attributes for contingency analysis

---

## Meaningfulness of Patterns: Bonferroni's Principle

A critical risk in data mining is discovering patterns that are actually meaningless - just statistical flukes. **Bonferroni's principle** warns us: if you look in more places for interesting patterns than your amount of data will support, you're bound to find spurious patterns.

This is why we need:
- **Statistical significance testing**: Is this pattern likely to occur by chance?
- **Validation on test data**: Does the pattern hold on data we didn't use to discover it?
- **Domain knowledge**: Does the pattern make sense in context?

Not all discovered patterns are interesting. We need both **objective measures** (based on statistics, like support and confidence) and **subjective measures** (based on user beliefs, like unexpectedness, novelty, and actionability).

---

## Key Takeaways

- **Data mining** discovers valid, useful, unexpected, and understandable patterns in large datasets. It's broader and more exploratory than machine learning, with a focus on unsupervised tasks.

- The **KDD process** provides a framework: data must be cleaned, integrated, and transformed before mining; discovered patterns must be evaluated and presented effectively.

- Core data mining tasks include **clustering** (finding groups), **outlier detection** (finding anomalies), **graph mining** (analyzing connections), and **frequent pattern mining** (finding associations).

- **Statistics** provides essential tools for exploratory data analysis: measures of central tendency (mean, median, mode), dispersion (variance, standard deviation), and relationships (covariance, correlation).

- **Data preprocessing** is crucial: normalization ensures attributes are on comparable scales, and proper distance metrics require this.

- **Pattern evaluation** is essential - not all discovered patterns are meaningful. We must guard against spurious findings using statistical tests and validation.

---

## Connections

This lecture establishes the foundation for everything that follows:
- The statistical concepts (mean, variance, correlation) will be used throughout clustering algorithms
- Distance metrics are fundamental to representative-based clustering (next lecture)
- The distinction between supervised and unsupervised learning sets up why we focus on clustering rather than classification
- The emphasis on exploratory analysis and pattern discovery motivates the algorithms we'll study

---

## References

- Lecture slides: `01-Intro.pdf`
- DMA book: Chapters 1, 2, and 3 (Data Matrix, Numeric Attributes, Categorical Attributes)
- Lecture notes: Introduction sections
