import numpy as np
import open3d as o3d



# read demo point cloud provided by Open3D
pcd_point_cloud_path = 'C:/Users/pchristou/OneDrive - FOS Development Corp/Documents/Personal Documents/School Shit/Homework/RP_HW2/fragment.pcd'
pcd = o3d.io.read_point_cloud(pcd_point_cloud_path)
# function to visualize the point cloud
o3d.visualization.draw_geometries([pcd],
                                zoom=1,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])


def fit_plane(points):
    # Fit plane Ax + By + Cz + D = 0
    p1, p2, p3 = points
    normal = np.cross(p2 - p1, p3 - p1)
    A, B, C = normal
    D = -np.dot(normal, p1)
    return A, B, C, D

def distance_to_plane(A, B, C, D, point):
    x, y, z = point
    return abs(A*x + B*y + C*z + D) / np.sqrt(A**2 + B**2 + C**2)

def check_inlier(A, B, C, D, cloud_points, distance_threshold):
    # Count the number of points within the threshold
    inlier_count = 0
    inlier_indices = []

    for index, point in enumerate(cloud_points):
        if distance_to_plane(A, B, C, D, point) <= distance_threshold:
            inlier_count += 1
            inlier_indices.append(index)

    return inlier_count, inlier_indices


def ADAPTIVE_RANSAC(pcd, sample_size, p):
    N = float('inf')
    sample_count = 0
    distance_threshold = 0.025
    best_plane = None
    total_inliers = 0
    total_inlier_indices = []
    
    while N > sample_count:
        # Choose a sample
        cloud_points = np.asarray(pcd.points)
        random_indices = np.random.choice(len(cloud_points), sample_size, replace=False)
        random_points = cloud_points[random_indices] 
        A, B, C, D = fit_plane(random_points)
        inliers, inlier_indices = check_inlier(A, B, C, D, cloud_points, distance_threshold)
                
        if inliers > total_inliers:
            total_inliers = inliers
            total_inlier_indices = inlier_indices
            best_plane = [A, B, C, D]
          
        total_points = cloud_points.shape[0]
                        
        # Calculate e
        e = 1 - inliers / total_points
      
        # Update N using the formula with p = 0.99
        N = np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** sample_size)).astype(int)
        
        # Increment sample_count
        print(sample_count)
        sample_count += 1
    
    # Convert inlier indices to indices in the original point cloud
    inlier_indices_original = np.array(list(range(len(pcd.points))))[total_inlier_indices]

    # Color the inliers red in the original point cloud
    colors = np.asarray(pcd.colors)
    colors[inlier_indices_original] = [1.0, 0.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Visualize the point cloud with colored inliers
    o3d.visualization.draw_geometries([pcd],
                                  zoom=1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


# Read the demo point cloud
sample_size = 3
p = 0.99
best_plane, inlier_data_index = ADAPTIVE_RANSAC(pcd, sample_size, p)