import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_test_data(
    data_path: Optional[str] = None,  # 测试数据文件路径，None时使用默认路径
    device="cuda",  # 计算设备，默认使用CUDA GPU
    scene_crop: Tuple[float, float, float, float, float, float] = (-2, -2, -2, 2, 2, 2),  # 场景裁剪边界 (xmin, ymin, zmin, xmax, ymax, zmax)
    scene_grid: int = 1,  # 场景网格重复次数，用于模拟大规模场景
):
    """
    加载3D高斯点云的测试数据
    
    该函数用于加载预处理的测试数据集，主要功能包括：
    1. 从NPZ文件中加载原始的点云数据和相机参数
    2. 根据指定边界对场景进行空间裁剪
    3. 通过网格重复扩展场景规模（用于大规模场景测试）
    4. 为高斯点云生成必要的几何和外观属性
    5. 返回完整的训练/测试所需数据
    
    @param data_path: 数据文件路径，支持NPZ格式。None时使用默认的garden测试数据
    @param device: 张量所在的计算设备，通常为"cuda"或"cpu"
    @param scene_crop: 场景裁剪的3D边界框，格式为(xmin, ymin, zmin, xmax, ymax, zmax)
    @param scene_grid: 场景在X-Y平面上的网格重复次数，必须为奇数
    
    @return: 包含高斯点云属性和相机参数的元组
        - means: 高斯点的3D位置 [N, 3]
        - quats: 高斯点的旋转四元数 [N, 4]
        - scales: 高斯点的尺度参数 [N, 3]  
        - opacities: 高斯点的不透明度 [N]
        - colors: 高斯点的颜色 [N, 3]
        - viewmats: 相机视图矩阵 [C, 4, 4]
        - Ks: 相机内参矩阵 [C, 3, 3]
        - width: 图像宽度
        - height: 图像高度
    """
    
    # ===== 参数验证 =====
    # 确保场景网格重复次数为奇数，这样可以保证原始场景在网格中心
    assert scene_grid % 2 == 1, "scene_grid must be odd"

    # ===== 数据文件路径处理 =====
    if data_path is None:
        # 如果未指定路径，使用默认的测试数据文件
        # 路径相对于当前文件所在目录，指向assets/test_garden.npz
        data_path = os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz")
    
    # ===== 加载原始数据 =====
    # 从NPZ文件中加载预处理的测试数据
    data = np.load(data_path)
    
    # 提取图像尺寸信息
    height, width = data["height"].item(), data["width"].item()
    
    # 加载相机参数并转换为PyTorch张量
    viewmats = torch.from_numpy(data["viewmats"]).float().to(device)  # [C, 4, 4] 相机视图变换矩阵
    Ks = torch.from_numpy(data["Ks"]).float().to(device)  # [C, 3, 3] 相机内参矩阵
    
    # 加载点云的几何和外观信息
    means = torch.from_numpy(data["means3d"]).float().to(device)  # [N, 3] 3D点位置
    colors = torch.from_numpy(data["colors"] / 255.0).float().to(device)  # [N, 3] RGB颜色，归一化到[0,1]
    
    # 获取相机数量
    C = len(viewmats)

    # ===== 场景空间裁剪 =====
    # crop - 根据指定的3D边界框对点云进行裁剪
    aabb = torch.tensor(scene_crop, device=device)  # 轴对齐边界框 [xmin, ymin, zmin, xmax, ymax, zmax]
    edges = aabb[3:] - aabb[:3]  # 计算边界框的尺寸 [width, height, depth]
    
    # 筛选在边界框内的点
    # ((means >= aabb[:3]) & (means <= aabb[3:])) 检查每个点的每个坐标是否在范围内
    # .all(dim=-1) 确保所有三个坐标都满足条件
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]  # 获取满足条件的点的索引
    
    # 只保留裁剪后的点和对应的颜色
    means, colors = means[sel], colors[sel]

    # ===== 场景网格化扩展 =====
    # repeat the scene into a grid (to mimic a large-scale setting)
    # 通过网格重复来模拟大规模场景，用于测试算法的可扩展性
    repeats = scene_grid
    
    # 创建网格坐标，以原点为中心的对称网格
    # 例如scene_grid=3时，网格坐标为 [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
    gridx, gridy = torch.meshgrid(
        [
            # 从 -(repeats//2) 到 repeats//2，总共repeats个网格点
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",  # 使用行列索引模式
    )
    
    # 组合X-Y网格坐标，Z坐标保持为0（在同一水平面重复）
    # grid shape: [repeats^2, 3]
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    
    # 将原始场景复制到每个网格位置
    # means[None, :, :] shape: [1, N, 3]
    # grid[:, None, :] shape: [repeats^2, 1, 3]  
    # edges[None, None, :] shape: [1, 1, 3]
    # 结果：每个网格位置都有一个完整的场景副本，按边界框尺寸进行偏移
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    means = means.reshape(-1, 3)  # 展平为 [repeats^2 * N, 3]
    
    # 相应地重复颜色数据
    colors = colors.repeat(repeats**2, 1)  # [repeats^2 * N, 3]

    # ===== 创建高斯点云属性 =====
    # create gaussian attributes - 为每个点生成高斯点云所需的几何属性
    N = len(means)  # 扩展后的总点数
    
    # 随机生成高斯椭球的尺度参数，使用较小的随机值确保初始尺度合理
    scales = torch.rand((N, 3), device=device) * 0.02  # [N, 3] 每个轴向的尺度
    
    # 随机生成旋转四元数并归一化
    # 四元数必须是单位四元数才能表示有效的旋转
    quats = F.normalize(torch.randn((N, 4), device=device), dim=-1)  # [N, 4]
    
    # 随机生成不透明度值，范围在[0, 1]
    opacities = torch.rand((N,), device=device)  # [N]

    # ===== 返回完整数据集 =====
    # 返回高斯点云训练/测试所需的所有数据
    return means, quats, scales, opacities, colors, viewmats, Ks, width, height
    
    # 返回值说明：
    # - means: 高斯点中心位置，经过裁剪和网格扩展
    # - quats: 旋转四元数，随机初始化
    # - scales: 尺度参数，随机初始化为小值
    # - opacities: 不透明度，随机初始化
    # - colors: 颜色信息，来自原始数据
    # - viewmats: 相机视图矩阵，来自原始数据
    # - Ks: 相机内参，来自原始数据
    # - width, height: 图像尺寸，来自原始数据
