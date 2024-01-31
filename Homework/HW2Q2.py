import open3d as o3d
import copy
import numpy as np
from sklearn.decomposition import PCA

# Write your code here
def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])

def find_nearest_neighbors(source, target, num_neighbors):
    # Find the closest neighbor for each anchor point through KDTree
    point_cloud_tree = o3d.geometry.KDTreeFlann(source)
    # Find nearest target_point neighbor index
    nearest_neighbors = []
    for point in target.points:
        [_, idx, _] = point_cloud_tree.search_knn_vector_3d(point, num_neighbors)
        nearest_neighbors.append(source.points[idx[0]])
    return np.asarray(nearest_neighbors)


def ICP(source, target):
    source.paint_uniform_color([0.5, 0.5, 0.5])
    target.paint_uniform_color([0, 0, 1])
    
    target_points = np.asarray(target.points)

    # Initial transformation matrix given in the ICP tutoral from Open3D
    transform_matrix = np.asarray([[0.862, 0.011, -0.507, 0.5], 
                                   [-0.139, 0.967, -0.215, 0.7], 
                                   [0.487, 0.255, 0.835, -1.4], 
                                   [0.0, 0.0, 0.0, 1.0]])
    source = source.transform(transform_matrix)

    curr_iteration = 0
    cost_change_threshold = 0.001
    curr_cost = 1000
    prev_cost = 10000

    while (True):
        # 1. Find nearest neighbors
        new_source_points = find_nearest_neighbors(source, target, 1)

        # 2. Find point cloud centroids and their repositions
        source_centroid = np.mean(new_source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        source_new_position = np.zeros_like(new_source_points)
        target_new_position = np.zeros_like(target_points)
        source_new_position = np.asarray([new_source_points[idx] - source_centroid for idx in range(len(new_source_points))])
        target_new_position = np.asarray([target_points[idx] - target_centroid for idx in range(len(target_points))])

        # 3. Find correspondence between source and target point clouds
        cov_mat = np.matmul(target_new_position.transpose(),source_new_position)

        U, X, V = np.linalg.svd(cov_mat)
        R = np.matmul(U,V)
        t = target_centroid - np.matmul(R,source_centroid)
        t = np.reshape(t, (1,3))
        curr_cost = np.linalg.norm(target_new_position - (np.matmul(R,source_new_position.T)).T)
        print("Curr_cost=", curr_cost)
        if ((prev_cost - curr_cost) > cost_change_threshold):
            prev_cost = curr_cost
            transform_matrix = np.hstack((R, t.T))
            transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
            # If cost_change is acceptable, update source with new transformation matrix
            source = source.transform(transform_matrix)
            curr_iteration += 1
        else:
            break
    print("\nIteration=", curr_iteration)
    # Visualize final iteration and print out final variables
    draw_registration_result(source, target, transform_matrix)
    return transform_matrix

### PART A ###

source = o3d.io.read_point_cloud('C:/Users/pchristou/OneDrive - FOS Development Corp/Documents/Personal Documents/School Shit/Homework/RP_HW2/cloud_bin_0.pcd')
target = o3d.io.read_point_cloud('C:/Users/pchristou/OneDrive - FOS Development Corp/Documents/Personal Documents/School Shit/Homework/RP_HW2/cloud_bin_1.pcd')
PART_A = ICP(source, target)

### PART B ###

source = o3d.io.read_point_cloud('C:/Users/pchristou/OneDrive - FOS Development Corp/Documents/Personal Documents/School Shit/Homework/RP_HW2/kitti_frame1.pcd')
target = o3d.io.read_point_cloud('C:/Users/pchristou/OneDrive - FOS Development Corp/Documents/Personal Documents/School Shit/Homework/RP_HW2/kitti_frame2.pcd')
print(source.points, target.points)
PART_B = ICP(source, target)