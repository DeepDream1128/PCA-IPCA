import numpy as np
from sklearn.decomposition import IncrementalPCA
import open3d as o3d
# 读取 PCD 文件
pcd = o3d.io.read_point_cloud("./global_map.pcd")
voxel_size = 0.1  # 可调整的体素大小
downsampled_pcd = pcd.voxel_down_sample(voxel_size)
points = np.asarray(downsampled_pcd.points)

# 应用 IPCA，选取前两个主成分
ipca = IncrementalPCA(n_components=2)
projected_points = ipca.fit_transform(points)

# 计算平面的法向量和点
normal = ipca.components_[2]
centroid = np.mean(points, axis=0)

# 创建平面的几何形状
plane = o3d.geometry.TriangleMesh.create_plane(center=centroid, normal=normal, width=20, height=20)

# 打印点云信息
print(pcd)

# 可视化点云
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(plane)
render_option = vis.get_render_option()
render_option.point_size = 1  # 设置点大小
vis.run()
vis.destroy_window()