import numpy as np
import open3d as o3d

def load_data(file_name):
    data = np.loadtxt(file_name)
    return data

def generate_planar_points(data, distance=0.1):
    points = []
    for row in data:
        position, normal = row[:3], row[3:]
        planar_point = position + normal * distance
        points.append(planar_point)
    return np.array(points)

def fit_plane_ransac(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    return plane_model, inliers

def visualize_points_and_plane(points, plane_model):
    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(points)

    plane_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_points, 0.01)
    plane_mesh.compute_vertex_normals()
    plane_mesh.paint_uniform_color([0, 1, 0])  # 绿色

    o3d.visualization.draw_geometries([pcd_points, plane_mesh])

def main():
    file_name = "PointsNormals.txt"
    data = load_data(file_name)

    # 生成片状点
    planar_points = generate_planar_points(data)

    # 使用RANSAC拟合平面
    plane_model, inliers = fit_plane_ransac(planar_points)

    # 输出平面模型参数：a, b, c, d
    print(f"平面模型参数：{plane_model}")

    # 使用Open3D显示片状点和拟合的平面
    visualize_points_and_plane(planar_points, plane_model)

if __name__ == "__main__":
    main()
