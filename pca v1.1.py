import numpy as np
from sklearn.decomposition import IncrementalPCA
import open3d as o3d

def load_data(file_name):
    data = np.loadtxt(file_name)
    return data

def generate_planar_points(data, distance=0.2):
    points = []
    for row in data:
        position, normal = row[:3], row[3:]
        planar_point = position + normal * distance
        points.append(planar_point)
    return np.array(points)

def ipca_reduction(data, dimensions, batch_size):
    ipca = IncrementalPCA(n_components=dimensions, batch_size=batch_size)
    reduced_data = ipca.fit_transform(data)
    return reduced_data, ipca

def fit_plane(ipca, point):
    reduced_point = ipca.transform(point.reshape(1, -1))
    fitted_point = ipca.inverse_transform(reduced_point)
    return fitted_point[0]  # 从2D数组中返回第一个元素，消除多余的维度

def visualize_points(fitted_points):
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(fitted_points)
    pcd_original.paint_uniform_color([0, 1, 0])  # 红色

    # pcd_fitted = o3d.geometry.PointCloud()
    # pcd_fitted.points = o3d.utility.Vector3dVector(fitted_points)
    # pcd_fitted.paint_uniform_color([0, 1, 0])  # 绿色

    # o3d.visualization.draw_geometries([pcd_original, pcd_fitted])
    # 设置渲染选项
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_original)
    render_option = vis.get_render_option()
    render_option.point_size = 3  # 设置点大小

    vis.run()
    vis.destroy_window()

def main():
    file_name = "PointsNormals.txt"
    data = load_data(file_name)

    # 生成片状点
    planar_points = generate_planar_points(data)
    print(planar_points.shape)
    # IPCA降维
    reduced_data, ipca = ipca_reduction(planar_points, 3, batch_size=500)

    # 平面拟合
    fitted_points = np.array([fit_plane(ipca, p) for p in planar_points])
    print(fitted_points.shape)

    # 三维可视化
    visualize_points(fitted_points)

if __name__ == "__main__":
    main()
