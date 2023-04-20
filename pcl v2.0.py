import open3d as o3d
import numpy as np

# 读取 PCD 文件
pcd = o3d.io.read_point_cloud("./global_map.pcd")
voxel_size = 0.05  # 可调整的体素大小
downsampled_pcd = pcd.voxel_down_sample(voxel_size)
# 使用 RANSAC 进行平面拟合
plane_model, inliers = downsampled_pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

# 提取平面上的点云
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0, 0])

# 提取平面外的点云
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# 可视化点云和拟合的平面
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# 打印点云信息
print(pcd)

# 可视化点云
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(inlier_cloud)
render_option = vis.get_render_option()
render_option.point_size = 1  # 设置点大小
vis.run()
vis.destroy_window()