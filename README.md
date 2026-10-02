## Introduction

This project implements a scalable similarity-search pipeline for identifying highly similar users in a large Netflix ratings dataset using MinHash and Locality Sensitive Hashing (LSH).

The dataset contains more than 65 million user–movie rating records, covering over 100,000 users and nearly 18,000 movies. Similarity between two users is defined using the Jaccard similarity of the sets of movies they rated; the actual rating values are ignored.

A brute-force comparison of all user pairs would require billions of pairwise similarity calculations, making direct computation impractical. The project therefore uses MinHash to construct compact user signatures that approximately preserve Jaccard similarity, followed by Locality Sensitive Hashing to restrict exact similarity calculations to promising candidate pairs.

Because the user–movie matrix is extremely sparse, the data is represented using a boolean Compressed Sparse Row (CSR) matrix from `scipy.sparse`. Integer indices are stored as 32-bit values and ratings are reduced to binary indicators, substantially reducing memory usage.

MinHash signatures are generated using 80 random permutations of the movie dimension. For each permutation, the first rated movie encountered for each user is recorded, producing an 80-dimensional signature for every user.

The signature matrix is subsequently divided into LSH bands. The number of bands and rows per band is selected automatically by maximizing the theoretical probability of detecting pairs with Jaccard similarity 0.5 under computational constraints. The final configuration uses 20 bands with 4 rows per band.

Within each band, users with matching signature segments are hashed into common buckets and treated as candidate pairs. Bucket sizes are capped to control the quadratic growth in pairwise comparisons.

Candidate verification uses a two-stage filtering procedure:

- signature similarity is first computed from the MinHash signatures;
- exact Jaccard similarity is calculated only for candidates whose signature similarity exceeds the threshold.

This filtering strategy reduces millions of LSH candidates to a much smaller set requiring exact comparison.

Across multiple random seeds, the implementation identifies approximately 150–200 user pairs with Jaccard similarity greater than 0.5 and completes in under 15 minutes on a standard laptop with 8 GB RAM.

The project combines sparse data structures, probabilistic hashing, approximate nearest-neighbor search, parameter optimization, memory management, and exact post-processing to perform large-scale similarity search efficiently.

---

## Summary

**Data Mining – Netflix User Similarity:** Implemented a scalable MinHash and Locality Sensitive Hashing pipeline (`NumPy`, `SciPy`) for Jaccard-based similarity search over 65M+ Netflix rating records, using CSR sparse matrices, optimized LSH banding and bucket construction, and two-stage signature/exact-Jaccard filtering to identify highly similar users within a sub-15-minute runtime on 8GB RAM.