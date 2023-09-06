# DataLearningInRealTimeSystems

### Analyzing 3D Point Cloud Data for Exit Detection Using Hierarchical Clustering and Convex Hull Refinement
This code aims to detect potential exits within 3D point cloud data. Leveraging geometric and clustering techniques, the solution offers robust detection, even in environments with noise or irregularities. 

### Summary:
The project provides a systematic approach to detecting potential exits in 3D point cloud data. Starting from the raw input, the data is loaded from XYZ files, a prevalent format for such spatial datasets. Once loaded, the data undergoes preprocessing where it's centered around its mean, ensuring uniformity. Extreme values that could skew results are also trimmed.

The Convex Hull technique is then applied on the 2D projection of the dataset. Recognizing that the initial convex hull might capture extraneous details, an iterative refinement strategy is used to adjust the hull vertices, aiming to maintain a 90-degree angle between them. This involves shrinking the original hull towards its center, and the points residing between the original and shrunk hull are assumed to potentially signify exits.

For the identification of these exits, hierarchical clustering techniques come into play. In the 2D context, the Agglomerative Hierarchical Clustering technique is deployed, focusing on the variance of distances from cluster centroids combined with cluster density to pinpoint the most probable exit cluster. This methodology is then extended into 3D, applying the same hierarchical clustering logic but adapted for three-dimensional data.

