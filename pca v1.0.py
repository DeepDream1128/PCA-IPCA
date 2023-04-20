import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import open3d as o3d

# 加载数据
def load_data(file_name):
    data = np.loadtxt(file_name)
    return data

# PCA降维
def pca_reduction(data, dimensions):
    pca = PCA(n_components=dimensions)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# 二维散点图
def plot_2d_scatter(data):
    plt.scatter(data[:, 1], data[:, 2])
    plt.xlabel("1")
    plt.ylabel("2")
    plt.title("2D Graph")
    plt.show()

# 三维散点图
def plot_3d_scatter(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    num_points = len(data)
    uniform_color = [0.5, 0.5, 0.5]  # 灰色，您可以选择其他颜色
    pcd.colors = o3d.utility.Vector3dVector([uniform_color] * num_points)
    # 设置渲染选项
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 3  # 设置点大小

    vis.run()
    vis.destroy_window()

def main():
    file_name = "./PointsNormals.txt"
    data = load_data(file_name)

    # 2D散点图
    reduced_data_6d = pca_reduction(data, 6)
    plot_2d_scatter(reduced_data_6d)

    # 3D散点图
    reduced_data_6d = pca_reduction(data, 6)
    reduced_data_3d = reduced_data_6d[:, [0, 1, 3]]  # 选择1, 2, 4主元
    plot_3d_scatter(reduced_data_3d)

if __name__ == "__main__":
    main()
