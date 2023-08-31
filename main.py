# Import necessary libraries
import open3d as o3d
import numpy as np
import tkinter as tk
from scipy.spatial import ConvexHull, Delaunay, distance_matrix
import matplotlib.pyplot as plt
from shapely import MultiPoint
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.patches as patches
import time
from tkinter import filedialog, Button, Label, Frame


# --------------------- File loading and data manipulation functions ---------------------
"""Load point cloud data from an XYZ file"""
def load_xyz(filename):
    data = np.loadtxt(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    return pcd


"""Load the specified map file and run the main program."""
def load_and_run_map(map_file):
    global filename, n_2d_clusters, n_3d_clusters
    filename = map_file
    n_2d_clusters, n_3d_clusters = map_configurations.get(map_file, (20, 20))

    main()


# GUI Creation
def create_gui():
    root = tk.Tk()
    root.title("Map Selector")

    label = Label(root, text="Select a map to load:")
    label.pack(pady=20)

    frame = Frame(root)
    frame.pack(pady=20)

    for map_file in map_configurations:
        button_name = map_file.split('.')[0]  # Extract the name without the extension
        button = Button(frame, text=button_name, command=lambda mf=map_file: load_and_run_map(mf))
        button.pack(side=tk.LEFT, padx=10)

    root.mainloop()


map_configurations = {
        "Map.xyz": (45, 31),
        "Map1.xyz": (10, 10),
        "Map2.xyz": (13, 13),
        "Map4.xyz": (16, 22),
        "Map5.xyz": (12, 8)
    }


"""Retrieve 3D points from the original dataset based on the X and Y coordinates of gap points"""
def retrieve_3d_points_from_2d(gap_points_2d, original_points_3d):
    retrieved_points = []

    for point_2d in gap_points_2d:
        matching_points = original_points_3d[np.where(
            (original_points_3d[:, 0] == point_2d[0]) &
            (original_points_3d[:, 1] == point_2d[1])
        )]
        retrieved_points.extend(matching_points)

    return np.array(retrieved_points)


"""Subtracts the mean of each coordinate from all points to center them around the origin"""
def center_points_around_mean(points):
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point
    return centered_points


"""Trims extreme values from a set of 2D points"""
def trim_extreme_values(points, percentage=5.0):
    if not (0 <= percentage <= 50):
        raise ValueError("Percentage should be between 0 and 50")
    x_values = points[:, 0]
    y_values = points[:, 1]
    x_low_threshold = np.percentile(x_values, percentage)
    x_high_threshold = np.percentile(x_values, 100 - percentage)
    y_low_threshold = np.percentile(y_values, percentage)
    y_high_threshold = np.percentile(y_values, 100 - percentage)
    x_mask = (x_values > x_low_threshold) & (x_values < x_high_threshold)
    y_mask = (y_values > y_low_threshold) & (y_values < y_high_threshold)
    combined_mask = x_mask & y_mask
    trimmed_points = points[combined_mask]
    return trimmed_points


# --------------------- Geometric and Mathematical functions ---------------------
"""Calculate the angle at p2 formed by p1-p2 and p2-p3"""
def calculate_angle(p1, p2, p3):
    vector1 = p1 - p2
    vector2 = p3 - p2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


"""Compute the convex hull of a set of points"""
def computeConvexHull(points):
    hull = ConvexHull(points)
    return hull.vertices


"""Compute the area of a convex hull"""
def compute_area(points):
    hull = ConvexHull(points)
    return hull.volume


"""Returns the points that are inside the convex hull"""
def points_inside_hull(points, hull_vertices):
    delaunay = Delaunay(hull_vertices)
    inside_mask = delaunay.find_simplex(points) >= 0
    inside_points = points[inside_mask]
    return inside_points


"""Extract points that are inside the outer hull but outside the inner hull"""
def points_between_hulls(all_points, hull_outer, hull_inner):
    delaunay_outer = Delaunay(hull_outer)
    delaunay_inner = Delaunay(hull_inner)
    mask_outer = delaunay_outer.find_simplex(all_points) >= 0
    mask_inner = delaunay_inner.find_simplex(all_points) < 0
    return all_points[np.logical_and(mask_outer, mask_inner)]


"""Computes the centroid for each cluster"""
def compute_cluster_centroids(points, labels):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        cluster_points = points[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


"""Refine the shape using guided convex hull functionality"""
def guided_iterative_refinement(points, max_iterations=200, refinement_percentage=0.1):
    for _ in range(max_iterations):
        hull = ConvexHull(points)
        angles = []
        for i in range(len(hull.vertices)):
            prev_vertex = points[hull.vertices[i - 1]]
            curr_vertex = points[hull.vertices[i]]
            next_vertex = points[hull.vertices[(i + 1) % len(hull.vertices)]]
            angle = calculate_angle(prev_vertex, curr_vertex, next_vertex)
            angles.append((angle, hull.vertices[i]))
        angles.sort(key=lambda x: abs(90 - x[0]), reverse=True)
        num_vertices_to_refine = int(refinement_percentage * len(hull.vertices))
        for j in range(num_vertices_to_refine):
            _, problematic_index = angles[j]
            prev_vertex = points[problematic_index - 1]
            curr_vertex = points[problematic_index]
            next_vertex = points[(problematic_index + 1) % len(points)]
            midpoint = (prev_vertex + next_vertex) / 2
            refined_vertex = (curr_vertex + midpoint) / 2
            points[problematic_index] = refined_vertex
    return points


"""Shrink the convex hull of the points towards the center"""
def shrink_hull_towards_center(points, hull_indices, shrink_factor=0.1):
    center = np.mean(points, axis=0)
    hull_points = points[hull_indices]
    shrunk_hull_points = center + shrink_factor * (hull_points - center)
    return shrunk_hull_points


"""Use KMeans clustering and detect exit based on the lowest density"""
def kmeans_clustering_and_exit_detection(points, n_clusters=70):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(points)
    min_density = float("inf")
    exit_cluster = None
    for label in np.unique(labels):
        cluster_points = points[labels == label]
        unique_cluster_points = np.unique(cluster_points, axis=0)
        if len(unique_cluster_points) == 2:
            distance = np.linalg.norm(unique_cluster_points[0] - unique_cluster_points[1])
            cluster_density = len(cluster_points) / distance
        elif len(unique_cluster_points) < 3:
            continue
        else:
            cluster_area = compute_area(cluster_points)
            cluster_density = len(cluster_points) / cluster_area
        if cluster_density < min_density:
            min_density = cluster_density
            exit_cluster = label
    return labels, exit_cluster


"""Use Agglomerative Hierarchical Clustering on 2D points and detect exit based on variance and density"""
def hierarchical_clustering_and_exit_detection(points, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(points)
    max_metric = float("-inf")
    exit_cluster_label = None
    for label in np.unique(labels):
        cluster_points = points[labels == label]
        unique_cluster_points = np.unique(cluster_points, axis=0)
        if len(unique_cluster_points) < 3:
            continue
        centroid = np.mean(cluster_points, axis=0)
        distances_from_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
        variance = np.var(distances_from_centroid)
        hull = ConvexHull(cluster_points)
        cluster_volume = hull.volume
        cluster_density = len(cluster_points) / cluster_volume
        combined_metric = variance * (1 - cluster_density)
        if combined_metric > max_metric:
            max_metric = combined_metric
            exit_cluster_label = label
    return labels, exit_cluster_label


"""Use Agglomerative Hierarchical Clustering on 3D points and detect exit based on variance and density"""
def hierarchical_exit_detection_3d(points, n_clusters=5):
    # Perform Agglomerative Hierarchical Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(points)

    # Find clusters with minimum number of points
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    min_clusters = unique_labels[counts == min_count]

    max_density = float("-inf")
    exit_cluster = None

    # Iterate over the clusters with minimum size
    for label in min_clusters:
        cluster_points = points[labels == label]

        # Calculate variance based on distances from the centroid
        centroid = np.mean(cluster_points, axis=0)
        distances_from_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
        variance = np.var(distances_from_centroid)

        # Compute density using variance and mean
        cluster_density = variance * np.mean(distances_from_centroid)

        # Update the exit cluster if needed
        if cluster_density > max_density:
            max_density = cluster_density
            exit_cluster = cluster_points

    return labels, exit_cluster


"""Computes an approximate solution to the TSP using the nearest neighbor heuristic"""
def nearest_neighbor_path(centroids):
    num_centroids = len(centroids)
    unvisited = set(range(num_centroids))
    current = np.random.choice(num_centroids)
    path = [current]
    unvisited.remove(current)
    dist_matrix = distance_matrix(centroids, centroids)
    while unvisited:
        next_centroid = min(unvisited, key=lambda x: dist_matrix[current][x])
        unvisited.remove(next_centroid)
        path.append(next_centroid)
        current = next_centroid
    return centroids[path]


# --------------------- Visualization functions ---------------------
"""Plot points with Convexhull"""
def plot_points_with_convexhull(original_points, points_filtered, rectangle):
    plt.figure(figsize=(20, 12))
    plt.scatter(original_points[:, 0], original_points[:, 1], c='green', s=4, label='Original Points')
    plt.scatter(points_filtered[:, 0], points_filtered[:, 1], c='magenta', s=4, label='Filtered Points')
    hull_path = np.append(rectangle, [rectangle[0]], axis=0)
    plt.plot(hull_path[:, 0], hull_path[:, 1], 'g--', label='Convex Hull')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original and Filtered Points with Convex Hull')
    plt.grid(True)
    plt.show()


"""Visualize the clusters and highlight the potential exit"""
def visualize_clusters_with_exit(points, labels, exit_cluster):
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    exit_points = None
    for label in unique_labels:
        cluster_points = points[labels == label]
        if label == exit_cluster:
            exit_points = cluster_points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, label=f"Potential Exit (Cluster {label})",
                        c='red', edgecolor='black')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, label=f"Cluster {label}")
    if exit_points is not None:
        exit_center = np.mean(exit_points, axis=0)
        max_distance = np.max(np.linalg.norm(exit_points - exit_center, axis=1))
        circle = patches.Circle(exit_center, max_distance * 1.2, fill=False, color='red', linestyle='dashed',
                                linewidth=2)
        plt.gca().add_patch(circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clusters with Potential Exit Highlighted')
    plt.grid(True)
    plt.show()


"""Visualize clusters with connected centroids and highlight the longest edge between centroids as exit"""
def visualize_clusters_with_connected_centroids(points, labels, exit_cluster):
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    exit_points = None
    centroids = compute_cluster_centroids(points, labels)
    path = nearest_neighbor_path(centroids)
    closed_path = np.vstack([path, path[0]])

    # Compute distances between consecutive centroids in the path and find the edge with the maximum distance.
    distances = np.linalg.norm(closed_path[1:] - closed_path[:-1], axis=1)
    max_distance_index = np.argmax(distances)
    exit_start = closed_path[max_distance_index]
    exit_end = closed_path[max_distance_index + 1]

    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='cyan', marker='X', edgecolor='black', label='Centroids')
    plt.plot(closed_path[:, 0], closed_path[:, 1], 'b--', label='Approximate Shortest Path')
    plt.plot([exit_start[0], exit_end[0]], [exit_start[1], exit_end[1]], 'r-', linewidth=2.5, label="Exit")

    for label in unique_labels:
        cluster_points = points[labels == label]
        if label == exit_cluster:
            exit_points = cluster_points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, label=f"Potential Exit (Cluster {label})",
                        c='red', edgecolor='black')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, label=f"Cluster {label}")

    if exit_points is not None:
        exit_center = np.mean(exit_points, axis=0)
        max_distance_to_center = np.max(np.linalg.norm(exit_points - exit_center, axis=1))
        circle = patches.Circle(exit_center, max_distance_to_center * 1.2, fill=False, color='red', linestyle='dashed',
                                linewidth=2)
        plt.gca().add_patch(circle)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clusters with Centroids, Path, and Exit')
    plt.legend()
    plt.grid(True)
    plt.show()


"""Visualize clusters, points, and main exit using Open3D"""
def visualize_clusters_and_exit_3D(points, labels, exit_cluster):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.array([0.5, 0.5, 0.5])[None, :] * np.ones_like(points)

    unique_labels = np.unique(labels)
    cluster_colors = np.random.rand(len(unique_labels), 3)
    for i, label in enumerate(unique_labels):
        colors[labels == label] = cluster_colors[i]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    exit_pcd = o3d.geometry.PointCloud()
    if exit_cluster is not None:
        exit_pcd.points = o3d.utility.Vector3dVector(exit_cluster)
        exit_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0])[None, :] * np.ones_like(exit_cluster))

    if exit_cluster is not None:
        bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(exit_cluster))
        bounding_box.color = (1, 0, 0)

    if exit_cluster is not None:
        o3d.visualization.draw_geometries([pcd, exit_pcd, bounding_box])
    else:
        o3d.visualization.draw_geometries([pcd, exit_pcd])


# --------------------- Main function ---------------------
def main():

    global filename

    # Load the point cloud data
    pcd = load_xyz(filename)

    # Pre-processing
    centered_points = center_points_around_mean(np.asarray(pcd.points))
    points_after_trimming = trim_extreme_values(centered_points, percentage=3.3)
    points_original = np.asarray(centered_points)[:, :2]
    points_filtered = np.asarray(points_after_trimming)[:, :2]
    points_filtered = guided_iterative_refinement(points_filtered, max_iterations=300, refinement_percentage=0.25)

    # Convex Hull Computation
    start_time_hull = time.time()
    hull_vertices = computeConvexHull(points_filtered)
    end_time_hull = time.time()
    print(f"Time taken for convex hull computation: {end_time_hull - start_time_hull} seconds")
    outside_hull_vertices2 = computeConvexHull(points_original)
    outside_hull_points2 = points_original[outside_hull_vertices2]
    hull_points = points_filtered[hull_vertices]

    # Visualizing the convex hull
    plot_points_with_convexhull(points_original, points_filtered, hull_points)

    # Shrunk Hull Computation
    shrunk_hull = shrink_hull_towards_center(points_filtered, hull_vertices, 0.85)
    outside_hull_vertices = computeConvexHull(points_original)
    outside_hull_points = points_original[outside_hull_vertices]

    # Compute points between hulls
    points_in_gap = points_between_hulls(points_original, outside_hull_points, shrunk_hull)
    points_in_gap = trim_extreme_values(points_in_gap, percentage=1.5)
    mask = np.isin(points_after_trimming[:, :2], points_in_gap).all(axis=1)
    points_in_gap_3D = points_after_trimming[mask]
    points_in_gap_3D = trim_extreme_values(points_in_gap_3D, percentage=1.5)

    # 3D Hierarchical Clustering
    labels, exit_cluster = hierarchical_exit_detection_3d(points_in_gap_3D, n_clusters=n_2d_clusters)
    visualize_clusters_and_exit_3D(points_in_gap_3D, labels, exit_cluster)

    # 2D Hierarchical Clustering
    start_time_kmeans = time.time()
    labels, exit_cluster = hierarchical_clustering_and_exit_detection(points_in_gap, n_clusters=n_3d_clusters)
    end_time_kmeans = time.time()
    print(f"Time taken for hierarchical clustering: {end_time_kmeans - start_time_kmeans} seconds")

    # Visualize the clusters
    visualize_clusters_with_exit(points_in_gap, labels, exit_cluster)
    visualize_clusters_with_connected_centroids(points_in_gap, labels, exit_cluster)


if __name__ == "__main__":
    create_gui()
