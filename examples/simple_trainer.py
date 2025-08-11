import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    """
    训练配置类：存储3D高斯点云训练的所有配置参数
    
    该类包含了数据加载、模型初始化、训练超参数、优化器设置、
    评估参数等训练过程中需要的所有配置选项。
    """
    
    # ===== 基础设置 =====
    # 是否禁用可视化查看器
    disable_viewer: bool = False
    # 检查点文件路径列表，如果提供则跳过训练直接进行评估
    ckpt: Optional[List[str]] = None
    # 压缩策略名称，目前支持"png"格式
    compression: Optional[Literal["png"]] = None
    # 渲染轨迹路径类型，可选"interp"(插值)等
    render_traj_path: str = "interp"

    # ===== 数据相关配置 =====
    # Mip-NeRF 360数据集路径
    data_dir: str = "data/360_v2/garden"
    # 数据下采样因子，用于减少输入图像分辨率
    data_factor: int = 4
    # 结果保存目录
    result_dir: str = "results/garden"
    # 每N张图像中有一张测试图像
    test_every: int = 8
    # 训练时随机裁剪尺寸（实验性功能）
    patch_size: Optional[int] = None
    # 场景尺寸相关参数的全局缩放因子
    global_scale: float = 1.0
    # 是否标准化世界坐标空间
    normalize_world_space: bool = True
    # 相机模型类型：针孔、正交或鱼眼
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # ===== 服务器配置 =====
    # 查看器服务器端口号
    port: int = 8080

    # ===== 训练配置 =====
    # 训练批大小，学习率会自动缩放
    batch_size: int = 1
    # 训练步数的全局缩放因子
    steps_scaler: float = 1.0

    # ===== 训练步数设置 =====
    # 最大训练步数
    max_steps: int = 30_000
    # 模型评估步数列表
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # 模型保存步数列表
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # 是否保存ply格式文件（文件可能很大）
    save_ply: bool = False
    # 保存ply格式模型的步数列表
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # 是否禁用训练和评估期间的视频生成
    disable_video: bool = False

    # ===== 模型初始化配置 =====
    # 初始化策略：SfM(运动恢复结构)或随机初始化
    init_type: str = "sfm"
    # 初始高斯点数量（使用SfM时忽略）
    init_num_pts: int = 100_000
    # 高斯点初始范围，作为相机范围的倍数（使用SfM时忽略）
    init_extent: float = 3.0
    # 球谐函数的阶数，控制颜色表示的复杂度
    sh_degree: int = 3
    # 每隔这么多步启用下一个球谐阶数
    sh_degree_interval: int = 1000
    # 高斯点初始不透明度
    init_opa: float = 0.1
    # 高斯点初始尺度
    init_scale: float = 1.0
    # SSIM损失的权重
    ssim_lambda: float = 0.2

    # ===== 渲染配置 =====
    # 近平面裁剪距离
    near_plane: float = 0.01
    # 远平面裁剪距离
    far_plane: float = 1e10

    # ===== 密集化策略配置 =====
    # 高斯点密集化策略，默认或MCMC策略
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # 是否使用打包模式进行光栅化（内存使用更少但稍慢）
    packed: bool = False
    # 是否使用稀疏梯度优化（实验性功能）
    sparse_grad: bool = False
    # 是否使用来自Taming 3DGS的可见Adam优化器（实验性功能）
    visible_adam: bool = False
    # 光栅化中的抗锯齿，可能会略微影响定量指标
    antialiased: bool = False

    # ===== 训练技巧 =====
    # 使用随机背景进行训练以抑制透明度
    random_bkgd: bool = False

    # ===== 学习率设置 =====
    # 3D点位置的学习率
    means_lr: float = 1.6e-4
    # 高斯尺度因子的学习率
    scales_lr: float = 5e-3
    # alpha混合权重（不透明度）的学习率
    opacities_lr: float = 5e-2
    # 方向（四元数）的学习率
    quats_lr: float = 1e-3
    # 球谐带0（亮度）的学习率
    sh0_lr: float = 2.5e-3
    # 高阶球谐（细节）的学习率
    shN_lr: float = 2.5e-3 / 20

    # ===== 正则化参数 =====
    # 不透明度正则化权重
    opacity_reg: float = 0.0
    # 尺度正则化权重
    scale_reg: float = 0.0

    # ===== 相机姿态优化 =====
    # 是否启用相机姿态优化
    pose_opt: bool = False
    # 相机姿态优化的学习率
    pose_opt_lr: float = 1e-5
    # 相机姿态优化的正则化权重衰减
    pose_opt_reg: float = 1e-6
    # 向相机外参添加噪声（仅用于测试相机姿态优化）
    pose_noise: float = 0.0

    # ===== 外观优化（实验性功能） =====
    # 是否启用外观优化
    app_opt: bool = False
    # 外观嵌入维度
    app_embed_dim: int = 16
    # 外观优化的学习率
    app_opt_lr: float = 1e-3
    # 外观优化的正则化权重衰减
    app_opt_reg: float = 1e-6

    # ===== 双边网格（实验性功能） =====
    # 是否使用双边网格
    use_bilateral_grid: bool = False
    # 双边网格的形状 (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # ===== 深度损失（实验性功能） =====
    # 是否启用深度损失
    depth_loss: bool = False
    # 深度损失的权重
    depth_lambda: float = 1e-2

    # ===== 日志记录 =====
    # 每隔这么多步向tensorboard输出信息
    tb_every: int = 100
    # 是否将训练图像保存到tensorboard
    tb_save_image: bool = False

    # ===== 感知损失网络 =====
    # LPIPS网络类型：VGG或AlexNet
    lpips_net: Literal["vgg", "alex"] = "alex"

    # ===== 3DGUT相关（实验性功能） =====
    # 是否使用无迹变换(Unscented Transform)
    with_ut: bool = False
    # 是否使用3D评估
    with_eval3d: bool = False

    # ===== 融合双边网格 =====
    # 是否使用融合的双边网格实现
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        """
        根据缩放因子调整所有步数相关的参数
        
        该方法用于在分布式训练或不同训练设置下按比例调整训练步数，
        确保训练计划的一致性。
        
        @param factor: 缩放因子，用于调整步数参数
        """
        # 调整评估步数列表
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        # 调整模型保存步数列表
        self.save_steps = [int(i * factor) for i in self.save_steps]
        # 调整PLY文件保存步数列表
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        # 调整最大训练步数
        self.max_steps = int(self.max_steps * factor)
        # 调整球谐阶数增长间隔
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        # 获取当前使用的密集化策略
        strategy = self.strategy
        # 根据不同策略类型调整相应的步数参数
        if isinstance(strategy, DefaultStrategy):
            # 默认策略：调整细化开始/结束步数和间隔
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            # MCMC策略：调整细化开始/结束步数和间隔
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            # 如果遇到未知策略类型，抛出错误
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,  # 数据解析器，包含SfM点云数据
    init_type: str = "sfm",  # 初始化类型："sfm"使用运动恢复结构数据，"random"随机初始化
    init_num_pts: int = 100_000,  # 随机初始化时的点数量
    init_extent: float = 3.0,  # 随机初始化时点云的空间范围倍数
    init_opacity: float = 0.1,  # 高斯点的初始不透明度
    init_scale: float = 1.0,  # 初始尺度的缩放因子
    means_lr: float = 1.6e-4,  # 点位置的学习率
    scales_lr: float = 5e-3,  # 尺度参数的学习率
    opacities_lr: float = 5e-2,  # 不透明度的学习率
    quats_lr: float = 1e-3,  # 四元数旋转的学习率
    sh0_lr: float = 2.5e-3,  # 球谐函数0阶（环境光）的学习率
    shN_lr: float = 2.5e-3 / 20,  # 球谐函数高阶（方向光）的学习率
    scene_scale: float = 1.0,  # 场景的全局缩放因子
    sh_degree: int = 3,  # 球谐函数的最大阶数，控制颜色表示的复杂度
    sparse_grad: bool = False,  # 是否使用稀疏梯度优化
    visible_adam: bool = False,  # 是否使用可见性感知的Adam优化器
    batch_size: int = 1,  # 训练批大小
    feature_dim: Optional[int] = None,  # 特征维度，None时使用球谐颜色，否则使用特征表示
    device: str = "cuda",  # 计算设备
    world_rank: int = 0,  # 分布式训练中当前进程的排名
    world_size: int = 1,  # 分布式训练的总进程数
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    创建3D高斯点云参数并初始化对应的优化器
    
    该函数是3D高斯点云渲染的核心初始化函数，负责：
    1. 根据不同策略初始化点云的位置和颜色
    2. 计算每个高斯点的初始尺度
    3. 初始化旋转（四元数）和不透明度
    4. 处理球谐函数颜色表示或特征表示
    5. 为不同参数创建专门的优化器
    6. 支持分布式训练的参数分配
    
    Args:
        parser: 包含SfM重建数据的解析器对象
        init_type: 初始化策略，"sfm"使用SfM点云，"random"随机生成
        init_num_pts: 随机初始化时生成的点数量
        init_extent: 随机初始化时点云分布的空间范围
        init_opacity: 所有高斯点的初始不透明度值
        init_scale: 初始尺度的缩放倍数
        means_lr: 点位置参数的学习率
        scales_lr: 尺度参数的学习率  
        opacities_lr: 不透明度参数的学习率
        quats_lr: 四元数旋转参数的学习率
        sh0_lr: 球谐函数0阶系数的学习率
        shN_lr: 球谐函数高阶系数的学习率
        scene_scale: 场景尺度，影响位置学习率
        sh_degree: 球谐函数最大阶数，控制颜色复杂度
        sparse_grad: 是否启用稀疏梯度优化以节省内存
        visible_adam: 是否使用只更新可见点的Adam优化器
        batch_size: 训练批大小，影响学习率缩放
        feature_dim: 特征维度，None时使用球谐颜色表示
        device: 张量所在的计算设备
        world_rank: 当前进程在分布式训练中的排名
        world_size: 分布式训练的总进程数
        
    Returns:
        Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]: 
            - 包含所有可训练参数的参数字典
            - 为每个参数组创建的优化器字典
    """
    
    # ===== 第一步：根据初始化类型设置点位置和颜色 =====
    if init_type == "sfm":
        # 使用SfM（运动恢复结构）的3D点作为初始位置
        points = torch.from_numpy(parser.points).float()
        # 将RGB颜色值从[0,255]归一化到[0,1]
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        # 随机初始化：在[-1,1]^3立方体内随机采样点，然后缩放到指定范围
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        # 随机生成RGB颜色值
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # ===== 第二步：计算初始高斯尺度 =====
    # 使用k近邻算法找到每个点最近的3个邻居点的平均距离作为初始尺度
    # 这确保了高斯点的大小与局部点密度相适应
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,] k=4但排除自身，取后3个
    dist_avg = torch.sqrt(dist2_avg)  # 计算平均距离
    # 将距离转换为对数空间的尺度参数，并复制到3个维度（x,y,z方向）
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # ===== 第三步：分布式训练的数据分配 =====
    # 将高斯点平均分配到不同的GPU进程中，每个进程处理一部分点
    # 使用轮询方式：进程0处理点0,world_size,2*world_size...，进程1处理点1,world_size+1...
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    # ===== 第四步：初始化其他高斯参数 =====
    N = points.shape[0]  # 当前进程处理的高斯点数量
    # 随机初始化四元数表示的旋转，四元数需要4个分量
    quats = torch.rand((N, 4))  # [N, 4]
    # 初始化不透明度：使用logit函数将[0,1]的不透明度映射到实数空间便于优化
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    # ===== 第五步：构建参数列表 =====
    # 每个元组包含：(参数名称, 参数值, 学习率)
    params = [
        # 高斯点的3D位置，学习率需要根据场景尺度调整
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        # 高斯椭球的尺度参数（log空间）
        ("scales", torch.nn.Parameter(scales), scales_lr),
        # 高斯椭球的旋转四元数
        ("quats", torch.nn.Parameter(quats), quats_lr),
        # 高斯点的不透明度（logit空间）
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    # ===== 第六步：处理颜色表示 =====
    if feature_dim is None:
        # 使用球谐函数表示颜色：可以表示视角相关的颜色变化
        # 球谐系数数量 = (sh_degree + 1)^2，每个系数有RGB三个通道
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        # 将RGB颜色转换为球谐函数的0阶系数（环境光分量）
        colors[:, 0, :] = rgb_to_sh(rgbs)
        # 分别为0阶和高阶球谐系数设置不同的学习率
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))  # 0阶系数
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))  # 高阶系数
    else:
        # 使用特征表示：适用于外观优化等高级功能
        # 为每个高斯点分配随机特征向量
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        # 同时保留基础颜色，使用logit空间便于优化
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    # ===== 第七步：创建参数字典并移动到指定设备 =====
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    # ===== 第八步：创建优化器 =====
    # 根据批大小缩放学习率，这是分布式训练的标准做法
    # 参考：https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # 注意：这种缩放不会使训练完全等价，详见 https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size  # 有效批大小 = 单GPU批大小 × GPU数量
    
    # 根据配置选择优化器类型
    optimizer_class = None
    if sparse_grad:
        # 稀疏梯度优化器：适用于大规模场景，只更新有梯度的参数
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        # 可见性感知优化器：只更新当前视角下可见的高斯点
        optimizer_class = SelectiveAdam
    else:
        # 标准Adam优化器：适用于大多数情况
        optimizer_class = torch.optim.Adam
    
    # 为每个参数组创建独立的优化器，允许使用不同的学习率
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            # eps值也需要根据批大小调整以保持数值稳定性
            eps=1e-15 / math.sqrt(BS),
            # 调整momentum参数beta以适应不同的批大小
            # TODO: 当BS > 10时需要检查beta逻辑，因为betas[0]可能变为零
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params  # 遍历所有参数组
    }
    
    # ===== 返回结果 =====
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        """
        Runner类的初始化方法：设置3D高斯点云训练和测试的完整环境
        
        该方法负责初始化训练过程中需要的所有组件，包括：
        - 数据加载和预处理
        - 模型参数和优化器初始化
        - 各种训练策略和技巧的设置
        - 损失函数和评估指标的配置
        - 可视化和日志记录系统的建立
        
        @param local_rank: 当前GPU在单机内的本地排名（0到GPU数量-1）
        @param world_rank: 当前进程在全局分布式训练中的排名
        @param world_size: 分布式训练的总进程数量
        @param cfg: 包含所有训练配置参数的Config对象
        """
        # ===== 基础环境设置 =====
        # 设置随机种子，确保不同GPU上的随机性不同但可重现
        # 每个GPU使用不同的种子（42 + local_rank）避免完全相同的随机化
        set_random_seed(42 + local_rank)

        # 保存配置和分布式训练相关的参数
        self.cfg = cfg  # 训练配置对象
        self.world_rank = world_rank  # 全局进程排名，用于分布式同步
        self.local_rank = local_rank  # 本地GPU排名，用于设备选择
        self.world_size = world_size  # 总进程数，用于数据分片和梯度聚合
        self.device = f"cuda:{local_rank}"  # 当前进程使用的GPU设备

        # ===== 创建输出目录结构 =====
        # 创建主结果目录，exist_ok=True表示目录存在时不报错
        os.makedirs(cfg.result_dir, exist_ok=True)

        # 设置各种输出子目录，用于保存不同类型的结果
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"  # 检查点目录：保存模型权重
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"  # 统计目录：保存训练和评估指标
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"  # 渲染目录：保存渲染结果图像
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"  # PLY目录：保存点云文件
        os.makedirs(self.ply_dir, exist_ok=True)

        # ===== 初始化Tensorboard日志记录器 =====
        # SummaryWriter用于记录训练过程中的损失、指标等信息，便于可视化分析
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # ===== 数据加载和预处理 =====
        # 训练数据应包含用于初始化的3D点和对应的颜色信息
        
        # 创建数据解析器：解析COLMAP格式的SfM重建结果
        self.parser = Parser(
            data_dir=cfg.data_dir,  # 数据集目录路径
            factor=cfg.data_factor,  # 图像下采样因子，用于减少计算量
            normalize=cfg.normalize_world_space,  # 是否标准化世界坐标系
            test_every=cfg.test_every,  # 每N张图像中选择一张作为测试图像
        )
        
        # 创建训练数据集
        self.trainset = Dataset(
            self.parser,
            split="train",  # 使用训练集分割
            patch_size=cfg.patch_size,  # 随机裁剪的块大小（实验性功能）
            load_depths=cfg.depth_loss,  # 是否加载深度信息（用于深度损失）
        )
        
        # 创建验证数据集，用于评估模型性能
        self.valset = Dataset(self.parser, split="val")
        
        # 计算场景尺度：用于调整学习率和其他尺度相关的参数
        # 1.1是一个经验值，稍微放大场景范围以确保完整覆盖
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # ===== 模型初始化 =====
        # 根据是否启用外观优化来决定特征维度
        # 外观优化需要额外的特征向量来建模不同图像间的外观变化
        feature_dim = 32 if cfg.app_opt else None
        
        # 创建3D高斯点云参数和对应的优化器
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,  # 数据解析器，提供初始点云数据
            init_type=cfg.init_type,  # 初始化类型：SfM或随机
            init_num_pts=cfg.init_num_pts,  # 随机初始化时的点数量
            init_extent=cfg.init_extent,  # 随机初始化时的空间范围
            init_opacity=cfg.init_opa,  # 初始不透明度
            init_scale=cfg.init_scale,  # 初始尺度缩放因子
            means_lr=cfg.means_lr,  # 位置参数学习率
            scales_lr=cfg.scales_lr,  # 尺度参数学习率
            opacities_lr=cfg.opacities_lr,  # 不透明度学习率
            quats_lr=cfg.quats_lr,  # 旋转四元数学习率
            sh0_lr=cfg.sh0_lr,  # 球谐0阶系数学习率
            shN_lr=cfg.shN_lr,  # 球谐高阶系数学习率
            scene_scale=self.scene_scale,  # 场景尺度
            sh_degree=cfg.sh_degree,  # 球谐函数最大阶数
            sparse_grad=cfg.sparse_grad,  # 是否使用稀疏梯度
            visible_adam=cfg.visible_adam,  # 是否使用可见性感知优化器
            batch_size=cfg.batch_size,  # 批大小
            feature_dim=feature_dim,  # 特征维度
            device=self.device,  # 计算设备
            world_rank=world_rank,  # 全局进程排名
            world_size=world_size,  # 总进程数
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # ===== 密集化策略初始化 =====
        # 密集化策略负责在训练过程中动态添加、删除和分裂高斯点
        # 检查策略配置的合理性，确保参数和优化器兼容
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        # 根据不同的密集化策略初始化对应的状态
        if isinstance(self.cfg.strategy, DefaultStrategy):
            # 默认策略：基于原始3DGS论文的密集化启发式方法
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale  # 传入场景尺度用于阈值计算
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            # MCMC策略：基于马尔科夫链蒙特卡洛的密集化方法
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            # 未知策略类型，抛出错误
            assert_never(self.cfg.strategy)

        # ===== 压缩策略初始化 =====
        # 压缩方法用于减少模型存储空间，目前支持PNG格式压缩
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                # PNG压缩：使用图像压缩技术压缩高斯参数
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        # ===== 相机姿态优化设置 =====
        # 相机姿态优化允许在训练过程中微调相机的外参，提高重建质量
        self.pose_optimizers = []
        if cfg.pose_opt:
            # 创建相机姿态调整模块，为每张训练图像学习姿态偏移
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            # 零初始化：开始时不对相机姿态进行调整
            self.pose_adjust.zero_init()
            
            # 为相机姿态参数创建专门的优化器
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),  # 学习率根据批大小缩放
                    weight_decay=cfg.pose_opt_reg,  # 正则化权重，防止过度调整
                )
            ]
            
            # 分布式训练时需要包装模型
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        # 相机姿态噪声添加（用于测试姿态优化的鲁棒性）
        if cfg.pose_noise > 0.0:
            # 创建姿态扰动模块，向相机外参添加随机噪声
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            # 随机初始化噪声，强度由pose_noise控制
            self.pose_perturb.random_init(cfg.pose_noise)
            
            # 分布式训练时需要包装模型
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # ===== 外观优化设置（实验性功能） =====
        # 外观优化用于建模不同图像间的光照、曝光等外观变化
        self.app_optimizers = []
        if cfg.app_opt:
            # 确保在外观优化模式下使用了特征表示
            assert feature_dim is not None
            
            # 创建外观优化模块：将图像ID映射到外观嵌入，再结合特征生成颜色
            self.app_module = AppearanceOptModule(
                len(self.trainset),  # 训练图像数量
                feature_dim,  # 高斯点特征维度
                cfg.app_embed_dim,  # 外观嵌入维度
                cfg.sh_degree  # 球谐函数阶数
            ).to(self.device)
            
            # 将最后一层初始化为零，确保初始时外观调整为零
            # 这样模型开始时使用基础颜色，逐渐学习外观变化
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            
            # 为外观模块的不同部分创建独立的优化器
            self.app_optimizers = [
                # 外观嵌入优化器：学习每张图像的外观特征
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,  # 更高的学习率
                    weight_decay=cfg.app_opt_reg,  # 正则化防止过拟合
                ),
                # 颜色头优化器：学习从特征到颜色的映射
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            
            # 分布式训练时需要包装模型
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # ===== 双边网格设置（实验性功能） =====
        # 双边网格用于实现图像后处理效果，如色彩校正
        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            # 创建双边网格：为每张训练图像学习一个3D查找表
            self.bil_grids = BilateralGrid(
                len(self.trainset),  # 训练图像数量
                grid_X=cfg.bilateral_grid_shape[0],  # X方向网格分辨率
                grid_Y=cfg.bilateral_grid_shape[1],  # Y方向网格分辨率
                grid_W=cfg.bilateral_grid_shape[2],  # 亮度方向网格分辨率
            ).to(self.device)
            
            # 为双边网格创建优化器
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),  # 固定的学习率策略
                    eps=1e-15,  # 数值稳定性参数
                ),
            ]

        # ===== 损失函数和评估指标初始化 =====
        # 初始化各种用于训练监督和模型评估的损失函数和指标
        
        # SSIM（结构相似性指数）：评估图像结构相似性，data_range=1.0表示像素值在[0,1]范围
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # PSNR（峰值信噪比）：评估图像重建质量的经典指标
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        # LPIPS（感知图像块相似性）：基于深度神经网络的感知损失函数
        if cfg.lpips_net == "alex":
            # 使用AlexNet作为特征提取网络，normalize=True表示对特征进行标准化
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # 使用VGG网络，这与3DGS官方实现保持一致
            # normalize=False表示不对VGG特征进行标准化
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # ===== 可视化查看器设置 =====
        # 实时可视化查看器允许在训练过程中交互式查看渲染结果
        if not self.cfg.disable_viewer:
            # 创建Viser服务器：提供Web界面的后端服务
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            
            # 创建Gsplat专用的查看器：集成了3D高斯点云的特定渲染功能
            self.viewer = GsplatViewer(
                server=self.server,  # Web服务器实例
                render_fn=self._viewer_render_fn,  # 自定义渲染函数
                output_dir=Path(cfg.result_dir),  # 输出目录
                mode="training",  # 训练模式：提供训练特定的交互功能
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,  # 相机到世界坐标的变换矩阵 [C, 4, 4]
        Ks: Tensor,  # 相机内参矩阵 [C, 3, 3]
        width: int,  # 渲染图像宽度
        height: int,  # 渲染图像高度
        masks: Optional[Tensor] = None,  # 可选的掩码，用于屏蔽某些区域 [C, H, W]
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,  # 光栅化模式
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,  # 相机模型类型
        **kwargs,  # 其他参数，如sh_degree、render_mode等
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        3D高斯点云光栅化渲染核心函数
        
        该函数是3D高斯点云渲染的核心，负责将3D空间中的高斯点投影到2D图像平面上，
        生成逼真的渲染图像。主要流程包括：
        1. 提取和预处理高斯点的几何和外观参数
        2. 处理不同的颜色表示方式（球谐函数或外观优化）
        3. 配置渲染参数和相机模型
        4. 调用底层CUDA实现的光栅化内核
        5. 应用掩码并返回渲染结果
        
        @param camtoworlds: 相机到世界坐标系的变换矩阵，用于确定观察视角
        @param Ks: 相机内参矩阵，包含焦距、主点等投影参数
        @param width: 目标渲染图像的像素宽度
        @param height: 目标渲染图像的像素高度
        @param masks: 可选的布尔掩码，True表示保留像素，False表示屏蔽
        @param rasterize_mode: 光栅化模式，"classic"为标准模式，"antialiased"为抗锯齿模式
        @param camera_model: 相机投影模型，支持针孔、正交、鱼眼等
        @param **kwargs: 额外参数，如球谐阶数、渲染模式、图像ID等
        
        @return Tuple[Tensor, Tensor, Dict]: 
            - render_colors: 渲染的颜色图像 [C, H, W, 3/4]
            - render_alphas: alpha通道（透明度）[C, H, W, 1]
            - info: 包含渲染统计信息的字典，如可见高斯点数量等
        """
        
        # ===== 第一步：提取和预处理高斯点参数 =====
        # 从参数字典中提取各个高斯点的几何和外观属性
        
        # 高斯点的3D中心位置坐标
        means = self.splats["means"]  # [N, 3] N为高斯点数量，3为XYZ坐标
        
        # 高斯点的旋转四元数（不需要手动归一化，光栅化函数内部会处理）
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4] 旧版本需要手动归一化
        # rasterization does normalization internally - 现在光栅化函数内部自动归一化
        quats = self.splats["quats"]  # [N, 4] 四元数表示的旋转，格式为[w, x, y, z]
        
        # 高斯椭球的尺度参数：从对数空间转换到实数空间
        # 训练时在对数空间优化以确保尺度为正值，渲染时需要指数变换
        scales = torch.exp(self.splats["scales"])  # [N, 3] 每个高斯点在XYZ三个轴向的尺度
        
        # 高斯点的不透明度：从logit空间转换到[0,1]概率空间
        # 训练时在logit空间优化避免数值不稳定，渲染时需要sigmoid变换
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,] 每个点的不透明度值

        # ===== 第二步：处理颜色表示 =====
        # 从kwargs中提取图像ID，用于外观优化或其他需要图像特定信息的功能
        image_ids = kwargs.pop("image_ids", None)
        
        if self.cfg.app_opt:
            # 外观优化模式：使用神经网络根据观察方向和图像ID动态生成颜色
            colors = self.app_module(
                features=self.splats["features"],  # 高斯点的特征向量 [N, feature_dim]
                embed_ids=image_ids,  # 图像ID，用于查找对应的外观嵌入
                # 计算观察方向：从高斯点中心指向相机中心的向量
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],  # [C, N, 3]
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),  # 球谐函数阶数
            )
            # 将外观调整添加到基础颜色上，实现精细的外观控制
            colors = colors + self.splats["colors"]  # [C, N, 3]
            # 通过sigmoid激活确保颜色值在[0,1]范围内
            colors = torch.sigmoid(colors)
        else:
            # 标准球谐函数颜色表示：将0阶和高阶球谐系数连接
            # sh0: 环境光分量（常数项），shN: 方向光分量（视角相关）
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
            # K = (sh_degree + 1)^2 为球谐系数总数

        # ===== 第三步：设置渲染配置参数 =====
        # 如果未指定光栅化模式，根据配置自动选择
        if rasterize_mode is None:
            # 抗锯齿模式可以减少渲染伪影，但计算开销稍大
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        # 如果未指定相机模型，使用配置中的默认模型
        if camera_model is None:
            camera_model = self.cfg.camera_model

        # ===== 第四步：调用底层光栅化函数 =====
        # 这是整个渲染过程的核心，调用CUDA加速的光栅化内核
        render_colors, render_alphas, info = rasterization(
            # 高斯点的几何参数
            means=means,  # 3D位置
            quats=quats,  # 旋转四元数
            scales=scales,  # 椭球尺度
            opacities=opacities,  # 不透明度
            colors=colors,  # 颜色表示（球谐或RGB）
            
            # 相机参数：需要世界到相机的变换矩阵（camtoworlds的逆矩阵）
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4] 世界到相机变换
            Ks=Ks,  # [C, 3, 3] 相机内参矩阵
            
            # 图像分辨率
            width=width,
            height=height,
            
            # 内存和计算优化选项
            packed=self.cfg.packed,  # 打包模式：减少内存使用但稍慢
            
            # 梯度计算相关配置
            absgrad=(
                # 只有DefaultStrategy需要计算绝对梯度用于密集化决策
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,  # 稀疏梯度：只为可见高斯点计算梯度
            
            # 渲染质量和模式设置
            rasterize_mode=rasterize_mode,  # 光栅化算法模式
            distributed=self.world_size > 1,  # 是否为分布式训练
            camera_model=self.cfg.camera_model,  # 相机投影模型
            
            # 3DGUT相关的实验性功能
            with_ut=self.cfg.with_ut,  # 是否使用无迹变换
            with_eval3d=self.cfg.with_eval3d,  # 是否启用3D评估
            
            # 其他传递的参数（如render_mode、near_plane、far_plane等）
            **kwargs,
        )

        # ===== 第五步：后处理和返回结果 =====
        # 如果提供了掩码，将被掩码的区域设置为黑色（RGB=0）
        if masks is not None:
            # ~masks 将True/False反转，False的区域（需要屏蔽的区域）被设为0
            render_colors[~masks] = 0

        # 返回渲染结果
        # render_colors: 渲染的颜色图像，可能包含RGB或RGBD通道
        # render_alphas: alpha通道，表示每个像素的累积透明度
        # info: 渲染统计信息，包含可见高斯点ID、半径等调试信息
        return render_colors, render_alphas, info

    def train(self):
        """
        3D高斯点云训练主函数
        
        该方法实现了完整的3D高斯点云训练流程，包括：
        1. 训练环境初始化和配置保存
        2. 学习率调度器设置
        3. 数据加载和预处理
        4. 训练循环中的前向传播、损失计算、反向传播
        5. 高斯点密集化策略的执行
        6. 模型检查点保存和评估
        7. 实时可视化和监控
        """
        # ===== 基础配置和变量初始化 =====
        cfg = self.cfg  # 训练配置对象
        device = self.device  # 当前GPU设备
        world_rank = self.world_rank  # 分布式训练中的全局排名
        world_size = self.world_size  # 分布式训练的总进程数

        # ===== 保存训练配置 =====
        # 只在主进程（rank 0）保存配置文件，避免多进程同时写入冲突
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                # 将配置对象的所有属性转换为字典并保存为YAML格式
                yaml.dump(vars(cfg), f)

        # ===== 训练参数设置 =====
        max_steps = cfg.max_steps  # 最大训练步数
        init_step = 0  # 起始步数（如果从检查点恢复训练，这里会被修改）

        # ===== 学习率调度器设置 =====
        # 为不同类型的参数设置专门的学习率衰减策略
        schedulers = [
            # 高斯点位置参数的学习率调度：指数衰减到初始值的1%
            # gamma值计算：0.01^(1/max_steps) 确保在max_steps步后学习率衰减到1%
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        
        # 相机姿态优化的学习率调度（如果启用）
        if cfg.pose_opt:
            # 姿态优化也使用指数衰减策略，与位置参数保持一致
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        
        # 双边网格的学习率调度（如果启用）
        if cfg.use_bilateral_grid:
            # 双边网格使用更复杂的调度策略：线性预热 + 指数衰减
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        # 第一阶段：线性预热1000步，从1%逐渐增加到100%
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,  # 起始学习率为设定值的1%
                            total_iters=1000,   # 预热1000步
                        ),
                        # 第二阶段：指数衰减，与其他参数保持一致
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        # ===== 数据加载器设置 =====
        # 创建训练数据加载器，使用多进程加载以提高效率
        trainloader = torch.utils.data.DataLoader(
            self.trainset,                    # 训练数据集
            batch_size=cfg.batch_size,        # 批大小
            shuffle=True,                     # 随机打乱数据顺序
            num_workers=4,                    # 使用4个工作进程进行数据加载
            persistent_workers=True,          # 保持工作进程持续运行，减少创建开销
            pin_memory=True,                  # 将数据固定在内存中，加速GPU传输
        )
        # 创建数据加载器迭代器，支持无限循环获取数据
        trainloader_iter = iter(trainloader)

        # ===== 训练循环初始化 =====
        global_tic = time.time()  # 记录训练开始时间，用于计算总训练时间
        # 创建进度条，显示训练进度和相关指标
        pbar = tqdm.tqdm(range(init_step, max_steps))
        
        # ===== 主训练循环 =====
        for step in pbar:
            # ===== 可视化查看器同步 =====
            # 如果启用了实时查看器，需要处理暂停/继续逻辑
            if not cfg.disable_viewer:
                # 当查看器处于暂停状态时，等待恢复
                while self.viewer.state == "paused":
                    time.sleep(0.01)  # 短暂休眠避免忙等待
                # 获取查看器锁，确保渲染和训练同步
                self.viewer.lock.acquire()
                tic = time.time()  # 记录当前步骤开始时间

            # ===== 数据获取和预处理 =====
            try:
                # 尝试获取下一批训练数据
                data = next(trainloader_iter)
            except StopIteration:
                # 数据集遍历完毕，重新创建迭代器开始新的epoch
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            # 将数据移动到GPU并进行预处理
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4] 相机到世界坐标变换
            Ks = data["K"].to(device)  # [1, 3, 3] 相机内参矩阵
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3] 图像像素，归一化到[0,1]
            
            # 计算当前步骤处理的射线总数，用于性能统计
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            
            # 获取图像ID，用于外观优化等功能
            image_ids = data["image_id"].to(device)
            # 获取掩码（如果存在），用于屏蔽某些区域不参与训练
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            
            # 深度损失相关数据（如果启用）
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2] 2D点坐标
                depths_gt = data["depths"].to(device)  # [1, M] 对应的真实深度值

            # 获取图像尺寸
            height, width = pixels.shape[1:3]

            # ===== 相机姿态处理 =====
            # 添加姿态噪声（用于测试姿态优化的鲁棒性）
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            # 应用姿态优化调整（如果启用）
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # ===== 球谐函数阶数调度 =====
            # 逐步增加球谐函数的阶数，从简单到复杂逐步学习颜色表示
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # ===== 前向传播 =====
            # 调用光栅化函数渲染当前视角的图像
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,      # 相机姿态
                Ks=Ks,                        # 相机内参
                width=width,                  # 图像宽度
                height=height,                # 图像高度
                sh_degree=sh_degree_to_use,   # 当前使用的球谐阶数
                near_plane=cfg.near_plane,    # 近裁剪平面
                far_plane=cfg.far_plane,      # 远裁剪平面
                image_ids=image_ids,          # 图像ID
                # 渲染模式：如果启用深度损失则同时渲染RGB和深度，否则只渲染RGB
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,                  # 掩码
            )
            
            # 分离颜色和深度通道
            if renders.shape[-1] == 4:
                # 4通道：RGB + 深度
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                # 3通道：仅RGB
                colors, depths = renders, None

            # ===== 双边网格后处理（如果启用） =====
            if cfg.use_bilateral_grid:
                # 创建归一化的网格坐标 [0, 1]
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",  # 使用行列索引顺序
                )
                # 组合XY坐标并添加批次维度
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                # 通过双边网格进行颜色校正
                colors = slice(
                    self.bil_grids,                                    # 双边网格模块
                    grid_xy.expand(colors.shape[0], -1, -1, -1),      # 扩展到批次大小
                    colors,                                            # 输入颜色
                    image_ids.unsqueeze(-1),                          # 图像ID
                )["rgb"]

            # ===== 随机背景处理 =====
            # 使用随机背景颜色增强训练，有助于学习准确的不透明度
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)  # 生成随机RGB背景色
                # Alpha混合：前景色 + 背景色 * (1 - alpha)
                colors = colors + bkgd * (1.0 - alphas)

            # ===== 密集化策略前处理 =====
            # 在反向传播前执行密集化策略的预处理步骤
            self.cfg.strategy.step_pre_backward(
                params=self.splats,           # 高斯点参数
                optimizers=self.optimizers,   # 优化器
                state=self.strategy_state,    # 策略状态
                step=step,                    # 当前步数
                info=info,                    # 渲染信息
            )

            # ===== 损失计算 =====
            # L1损失：衡量渲染图像和真实图像的像素级差异
            l1loss = F.l1_loss(colors, pixels)
            
            # SSIM损失：衡量结构相似性，需要调整维度顺序为[B, C, H, W]
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            
            # 综合损失：L1损失和SSIM损失的加权组合
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            
            # 深度损失（如果启用）
            if cfg.depth_loss:
                # 将2D点坐标从像素空间[0, W/H]归一化到[-1, 1]用于grid_sample
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,   # X坐标归一化
                        points[:, :, 1] / (height - 1) * 2 - 1,  # Y坐标归一化
                    ],
                    dim=-1,
                )  # [1, M, 2] 归一化到[-1, 1]范围
                
                # 为grid_sample添加必要的维度
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                
                # 从深度图中采样对应点的深度值
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M] 移除多余维度
                
                # 在视差空间计算损失（比直接深度损失更稳定）
                # 视差 = 1/深度，对于深度为0的点视差也为0
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M] 真实视差
                
                # 计算视差L1损失，乘以场景尺度进行归一化
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                # 将深度损失加入总损失
                loss += depthloss * cfg.depth_lambda
            
            # 双边网格的总变差正则化（如果启用）
            if cfg.use_bilateral_grid:
                # 总变差损失鼓励网格平滑，避免过度锐化
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # ===== 正则化项 =====
            # 不透明度正则化：鼓励高斯点保持适度的不透明度
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            
            # 尺度正则化：防止高斯点变得过大
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            # ===== 反向传播 =====
            loss.backward()

            # ===== 训练进度显示 =====
            # 构建进度条描述信息
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # 如果同时启用姿态优化和噪声，监控姿态误差
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # ===== Tensorboard日志记录 =====
            # 定期记录训练指标到Tensorboard（仅主进程）
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3  # GPU内存使用量(GB)
                # 记录各种训练指标
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                
                # 记录可选的损失项
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                
                # 保存训练图像（如果启用）
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # ===== 检查点保存 =====
            # 在指定步数保存模型检查点
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                # 收集训练统计信息
                stats = {
                    "mem": mem,                                    # 内存使用量
                    "ellipse_time": time.time() - global_tic,     # 总训练时间
                    "num_GS": len(self.splats["means"]),          # 高斯点数量
                }
                print("Step: ", step, stats)
                
                # 保存统计信息到JSON文件
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                
                # 准备检查点数据
                data = {"step": step, "splats": self.splats.state_dict()}
                
                # 保存姿态优化模块（如果启用）
                if cfg.pose_opt:
                    if world_size > 1:
                        # 分布式训练时需要访问.module属性
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                
                # 保存外观优化模块（如果启用）
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                
                # 保存检查点文件
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            
            # ===== PLY文件保存 =====
            # 保存高斯点云为PLY格式文件（用于可视化）
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # 外观优化模式：将外观烘焙到颜色中
                    # 在原点方向评估外观模块，获得平均颜色
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,  # 不使用特定图像的外观
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),  # 零方向向量
                        sh_degree=sh_degree_to_use,
                    )
                    # 添加基础颜色并应用sigmoid激活
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    # 转换为球谐表示
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)  # 空的高阶项
                else:
                    # 标准模式：直接使用球谐系数
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                # 提取高斯点参数
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                
                # 导出为PLY文件
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # ===== 稀疏梯度处理 =====
            # 将密集梯度转换为稀疏张量以节省内存和计算
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]  # 可见高斯点的索引
                
                # 遍历所有参数，将梯度转换为稀疏格式
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue  # 跳过没有梯度或已经是稀疏的参数
                    
                    # 创建稀疏COO张量，只存储可见高斯点的梯度
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],        # [1, nnz] 索引
                        values=grad[gaussian_ids],          # [nnz, ...] 梯度值
                        size=self.splats[k].size(),         # [N, ...] 完整张量大小
                        is_coalesced=len(Ks) == 1,         # 是否合并重复索引
                    )

            # ===== 可见性掩码计算（可见Adam优化器） =====
            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    # 打包模式：根据渲染信息创建可见性掩码
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    # 非打包模式：根据半径判断可见性
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # ===== 优化器更新 =====
            # 更新高斯点参数
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    # 可见Adam：只更新可见的高斯点
                    optimizer.step(visibility_mask)
                else:
                    # 标准优化：更新所有参数
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # 清零梯度，释放内存
            
            # 更新姿态优化器
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # 更新外观优化器
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # 更新双边网格优化器
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # 更新所有学习率调度器
            for scheduler in schedulers:
                scheduler.step()

            # ===== 密集化策略后处理 =====
            # 在优化器更新后执行密集化策略的后处理步骤
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],  # 传入当前学习率
                )
            else:
                assert_never(self.cfg.strategy)

            # ===== 模型评估 =====
            # 在指定步数进行完整的验证集评估
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)           # 评估验证集
                self.render_traj(step)    # 渲染轨迹视频

            # ===== 模型压缩 =====
            # 在评估步数运行压缩算法
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            # ===== 可视化查看器更新 =====
            if not cfg.disable_viewer:
                # 释放查看器锁
                self.viewer.lock.release()
                
                # 计算性能指标
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                
                # 更新查看器状态显示
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                
                # 更新场景显示
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """
        模型评估函数：在验证集上评估3D高斯点云模型的渲染质量
        
        该方法执行完整的模型评估流程，包括：
        1. 在验证集上进行推理渲染
        2. 计算多种图像质量指标（PSNR、SSIM、LPIPS）
        3. 保存对比图像和统计结果
        4. 记录性能指标到日志和Tensorboard
        
        @torch.no_grad() 装饰器的作用：
        - 禁用自动梯度计算，节省内存和计算资源
        - 防止在评估过程中意外修改模型参数
        - 加速推理过程，因为不需要构建计算图
        
        @param step: 当前训练步数，用于标识评估时机
        @param stage: 评估阶段标识，默认为"val"（验证），也可以是"test"等
        """
        print("Running evaluation...")
        
        # ===== 基础配置获取 =====
        cfg = self.cfg  # 训练配置对象
        device = self.device  # 当前GPU设备
        world_rank = self.world_rank  # 分布式训练中的进程排名
        world_size = self.world_size  # 分布式训练的总进程数

        # ===== 验证数据加载器设置 =====
        # 创建验证集数据加载器，配置与训练时不同：
        valloader = torch.utils.data.DataLoader(
            self.valset,           # 验证数据集
            batch_size=1,          # 评估时使用批大小1，逐张图像处理
            shuffle=False,         # 不打乱顺序，确保结果可重现
            num_workers=1          # 使用单线程加载，避免多进程开销
        )
        
        # ===== 性能统计变量初始化 =====
        ellipse_time = 0  # 累计渲染时间，用于计算平均每张图像的渲染速度
        metrics = defaultdict(list)  # 存储各种评估指标的列表，自动创建空列表
        
        # ===== 验证集推理循环 =====
        for i, data in enumerate(valloader):
            # ===== 数据预处理 =====
            # 将验证数据移动到GPU并进行预处理
            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4] 相机到世界坐标变换矩阵
            Ks = data["K"].to(device)  # [1, 3, 3] 相机内参矩阵
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3] 真实图像，归一化到[0,1]
            masks = data["mask"].to(device) if "mask" in data else None  # 可选掩码
            height, width = pixels.shape[1:3]  # 获取图像尺寸

            # ===== 渲染时间测量开始 =====
            # 同步CUDA操作，确保时间测量准确
            torch.cuda.synchronize()
            tic = time.time()  # 记录渲染开始时间
            
            # ===== 模型推理渲染 =====
            # 使用当前训练的模型渲染该视角的图像
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,    # 相机姿态
                Ks=Ks,                      # 相机内参
                width=width,                # 图像宽度
                height=height,              # 图像高度
                sh_degree=cfg.sh_degree,    # 使用完整的球谐阶数（评估时不做渐进）
                near_plane=cfg.near_plane,  # 近裁剪平面
                far_plane=cfg.far_plane,    # 远裁剪平面
                masks=masks,                # 掩码
            )  # [1, H, W, 3] 渲染得到的RGB图像
            
            # ===== 渲染时间测量结束 =====
            torch.cuda.synchronize()  # 再次同步，等待GPU计算完成
            ellipse_time += max(time.time() - tic, 1e-10)  # 累计渲染时间，避免除零

            # ===== 图像后处理 =====
            # 将渲染结果限制在[0,1]范围内，防止数值溢出影响指标计算
            colors = torch.clamp(colors, 0.0, 1.0)
            # 准备对比图像：[真实图像, 渲染图像] 水平拼接
            canvas_list = [pixels, colors]

            # ===== 结果保存和指标计算（仅主进程） =====
            if world_rank == 0:
                # ===== 保存对比图像 =====
                # 将GPU张量转换为CPU numpy数组，便于图像保存
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()  # 水平拼接并移除批次维度
                canvas = (canvas * 255).astype(np.uint8)  # 转换为8位整数像素值
                # 保存对比图像，文件名包含阶段、步数、图像索引
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                # ===== 图像质量指标计算 =====
                # 调整张量维度顺序以适配torchmetrics库的要求：[B, C, H, W]
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W] 真实图像
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W] 渲染图像
                
                # 计算三个核心图像质量指标
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))    # 峰值信噪比：衡量像素级相似性
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))    # 结构相似性：衡量结构特征保持
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))  # 感知相似性：衡量人眼感知质量
                
                # ===== 双边网格颜色校正指标（如果启用） =====
                if cfg.use_bilateral_grid:
                    # 对渲染图像进行颜色校正，使其更接近真实图像的色调
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    
                    # 计算颜色校正后的图像质量指标
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))    # 校正后PSNR
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))    # 校正后SSIM
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))  # 校正后LPIPS

        # ===== 评估结果统计和输出（仅主进程） =====
        if world_rank == 0:
            # ===== 计算平均指标 =====
            # 计算每张图像的平均渲染时间
            ellipse_time /= len(valloader)

            # 将每个指标的列表转换为张量，计算平均值并转为Python标量
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            
            # 添加性能统计信息
            stats.update(
                {
                    "ellipse_time": ellipse_time,              # 平均每张图像渲染时间（秒）
                    "num_GS": len(self.splats["means"]),      # 当前高斯点总数
                }
            )
            
            # ===== 结果打印输出 =====
            if cfg.use_bilateral_grid:
                # 双边网格模式：显示原始和颜色校正后的指标
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                # 标准模式：只显示基础指标
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            
            # ===== 结果持久化保存 =====
            # 保存统计结果到JSON文件，便于后续分析
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            
            # 记录统计结果到Tensorboard，便于可视化监控
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()  # 强制写入Tensorboard日志

    @torch.no_grad()
    def render_traj(self, step: int):
        """
        轨迹渲染函数：生成沿指定相机轨迹的渲染视频
        
        该方法用于在训练过程中或训练结束后生成演示视频，展示3D高斯点云模型
        在不同视角下的渲染效果。支持多种轨迹类型：插值、椭圆、螺旋等。
        
        @torch.no_grad() 装饰器的作用：
        - 禁用梯度计算，节省内存和计算资源
        - 这是推理过程，不需要更新模型参数
        
        @param step: 当前训练步数，用于视频文件命名
        """
        # ===== 检查是否禁用视频生成 =====
        if self.cfg.disable_video:
            return  # 如果配置中禁用了视频生成，直接返回
        
        print("Running trajectory rendering...")
        
        # ===== 基础配置获取 =====
        cfg = self.cfg  # 训练配置对象
        device = self.device  # 当前GPU设备

        # ===== 获取基础相机轨迹 =====
        # 从解析器中获取所有训练相机的姿态，去掉首尾各5个相机避免边界效应
        # 这样可以确保插值轨迹更加平滑和稳定
        camtoworlds_all = self.parser.camtoworlds[5:-5]  # [N-10, 4, 4] 相机到世界坐标变换矩阵
        
        # ===== 根据配置生成不同类型的相机轨迹 =====
        if cfg.render_traj_path == "interp":
            # 插值轨迹：在现有相机位置之间进行平滑插值
            # 这种方式能够重现训练时的视角变化，适合展示模型对训练视角的重建质量
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1  # 插值因子为1，表示在每两个相机之间插入1个新视角
            )  # [N', 3, 4] 返回插值后的相机轨迹（注意：这里是3x4形式，缺少齐次坐标行）
            
        elif cfg.render_traj_path == "ellipse":
            # 椭圆轨迹：生成围绕场景的椭圆形飞行路径
            # 适合全方位展示3D场景，提供cinematic的观看体验
            height = camtoworlds_all[:, 2, 3].mean()  # 计算所有相机的平均高度（Z坐标）
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height  # 在指定高度生成椭圆轨迹
            )  # [N', 3, 4] 椭圆轨迹的相机姿态
            
        elif cfg.render_traj_path == "spiral":
            # 螺旋轨迹：生成螺旋上升或下降的相机路径
            # 提供更加动态和立体的视角变化，适合复杂场景的展示
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,  # 基础相机轨迹
                bounds=self.parser.bounds * self.scene_scale,  # 场景边界，调整螺旋范围
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],  # 螺旋半径缩放因子
            )  # [N', 3, 4] 螺旋轨迹的相机姿态
            
        else:
            # 不支持的轨迹类型，抛出错误
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        # ===== 补全齐次坐标 =====
        # 轨迹生成函数返回的是3x4变换矩阵，需要补充最后一行[0, 0, 0, 1]成为4x4齐次变换矩阵
        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,  # 原始的3x4变换矩阵
                # 为每个相机姿态添加齐次坐标行[0, 0, 0, 1]
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]),  # 齐次坐标行
                    len(camtoworlds_all),  # 重复N'次，对应每个相机姿态
                    axis=0  # 在批次维度上重复
                ),
            ],
            axis=1,  # 在矩阵的行维度上拼接
        )  # [N', 4, 4] 完整的齐次变换矩阵

        # ===== 数据类型转换和设备迁移 =====
        # 将numpy数组转换为PyTorch张量并移动到GPU
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        
        # 获取相机内参矩阵（使用第一个相机的内参，假设所有相机内参相同）
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        
        # 获取图像尺寸（使用第一张图像的尺寸）
        width, height = list(self.parser.imsize_dict.values())[0]

        # ===== 视频保存设置 =====
        # 创建视频保存目录
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        
        # 创建视频写入器，设置输出路径和帧率
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        
        # ===== 轨迹渲染循环 =====
        # 遍历轨迹上的每个相机位置，逐帧渲染
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            # ===== 准备单帧渲染的相机参数 =====
            # 提取当前帧的相机姿态，保持批次维度
            camtoworlds = camtoworlds_all[i : i + 1]  # [1, 4, 4]
            Ks = K[None]  # [1, 3, 3] 为内参矩阵添加批次维度

            # ===== 渲染当前视角 =====
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,    # 当前相机姿态
                Ks=Ks,                      # 相机内参
                width=width,                # 图像宽度
                height=height,              # 图像高度
                sh_degree=cfg.sh_degree,    # 使用完整的球谐阶数
                near_plane=cfg.near_plane,  # 近裁剪平面
                far_plane=cfg.far_plane,    # 远裁剪平面
                render_mode="RGB+ED",       # 渲染模式：RGB颜色 + 期望深度
            )  # [1, H, W, 4] 渲染结果包含RGB和深度通道
            
            # ===== 图像后处理 =====
            # 提取RGB颜色通道并限制在[0,1]范围内
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            
            # 提取深度通道
            depths = renders[..., 3:4]  # [1, H, W, 1]
            
            # 深度归一化：将深度值映射到[0,1]范围，便于可视化
            # 使用min-max归一化，使最近点为0（黑色），最远点为1（白色）
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            
            # 准备画布：将颜色和深度图并排显示
            # 深度图复制到3个通道以匹配RGB格式
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # ===== 图像格式转换和保存 =====
            # 水平拼接颜色图和深度图
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()  # [H, W*2, 3]
            
            # 转换为8位整数像素值
            canvas = (canvas * 255).astype(np.uint8)
            
            # 将当前帧添加到视频中
            writer.append_data(canvas)
        
        # ===== 完成视频生成 =====
        writer.close()  # 关闭视频写入器，确保文件正确保存
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """
        模型压缩运行函数：对3D高斯点云模型进行压缩并评估压缩效果
        
        该方法实现了完整的模型压缩和评估流程，主要用于：
        1. 将训练好的高斯点云模型进行压缩以减少存储空间
        2. 测试压缩算法对模型质量的影响
        3. 为模型部署提供更紧凑的表示形式
        
        @torch.no_grad() 装饰器的作用：
        - 禁用自动梯度计算，节省内存开销
        - 压缩和评估过程不需要梯度信息
        - 避免意外修改模型参数
        
        @param step: 当前训练步数，用于标识压缩时机和结果文件命名
        """
        print("Running compression...")
        
        # ===== 获取分布式训练信息 =====
        world_rank = self.world_rank  # 当前进程在分布式训练中的排名
        
        # ===== 创建压缩结果存储目录 =====
        # 为每个进程创建独立的压缩目录，避免多进程同时写入冲突
        # 目录结构：{result_dir}/compression/rank{rank_id}
        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)  # 创建目录，如果已存在则不报错

        # ===== 执行模型压缩 =====
        # 调用压缩方法将当前的高斯点云模型进行压缩
        # self.compression_method 在初始化时根据配置创建（如PngCompression等）
        # 压缩过程会将模型参数转换为更紧凑的表示形式并保存到指定目录
        self.compression_method.compress(compress_dir, self.splats)
        
        # 压缩方法的作用：
        # - 将高斯点的各种参数（位置、尺度、旋转、不透明度、颜色等）进行有损或无损压缩
        # - 使用专门的压缩算法（如PNG图像压缩）减少数据存储大小
        # - 保存压缩后的数据到文件系统中

        # ===== 评估压缩效果 =====
        print("Evaluating compression quality...")
        
        # 第一步：解压缩模型
        # 从压缩文件中恢复模型参数，模拟实际使用时的解压过程
        splats_c = self.compression_method.decompress(compress_dir)
        
        # 解压缩过程的作用：
        # - 从压缩文件中读取数据并还原为张量格式
        # - 可能存在量化误差或其他压缩引入的精度损失
        # - 返回的数据格式与原始模型参数保持一致
        
        # 第二步：替换当前模型参数
        # 将解压缩后的参数重新加载到模型中，替换原始的未压缩参数
        for k in splats_c.keys():
            # 遍历所有参数类型（means, scales, quats, opacities, sh0, shN等）
            # 将解压缩的参数数据复制到原始参数的data属性中
            # .to(self.device) 确保数据在正确的GPU设备上
            self.splats[k].data = splats_c[k].to(self.device)
        
        # 参数替换的意义：
        # - 使用压缩后再解压的参数进行评估，真实反映压缩对质量的影响
        # - 保持原有的参数结构和优化器状态
        # - 只更新参数值，不改变模型架构
        
        # 第三步：运行压缩模型评估
        # 使用压缩后的模型在验证集上进行评估，测量压缩对渲染质量的影响
        self.eval(step=step, stage="compress")
        
        # 评估过程包括：
        # - 使用压缩后的模型渲染验证集图像
        # - 计算PSNR、SSIM、LPIPS等图像质量指标
        # - 将评估结果保存为"compress"阶段的数据
        # - 结果文件命名格式：compress_step{step:04d}.json
        
        # ===== 压缩效果分析 =====
        # 通过对比压缩前后的评估指标，可以分析：
        # 1. 压缩率（文件大小减少比例）
        # 2. 质量损失（PSNR/SSIM/LPIPS的变化）
        # 3. 压缩算法的性能权衡
        # 4. 不同压缩参数设置的效果

    print("Compression and evaluation completed.")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    """
    主函数：3D高斯点云训练和评估的程序入口
    
    该函数是整个程序的核心入口，负责协调训练和评估流程，包括：
    1. 分布式训练配置的处理
    2. 训练器(Runner)实例的创建和初始化
    3. 根据配置决定执行训练或评估模式
    4. 检查点加载和模型恢复
    5. 可视化查看器的管理
    
    @param local_rank: 当前GPU在单机内的本地排名（0到GPU数量-1）
    @param world_rank: 当前进程在全局分布式训练中的排名
    @param world_size: 分布式训练的总进程数量
    @param cfg: 包含所有训练和评估配置的Config对象
    """
    
    # ===== 分布式训练配置检查 =====
    # 在分布式训练模式下，自动禁用可视化查看器以避免冲突
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True  # 强制禁用查看器
        # 只在主进程输出提示信息，避免重复打印
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")
    
    # 分布式训练禁用查看器的原因：
    # 1. 多个进程同时尝试启动Web服务器会导致端口冲突
    # 2. 查看器界面只需要一个进程提供即可
    # 3. 减少多进程间的同步复杂度

    # ===== 创建训练器实例 =====
    # Runner类是整个训练和评估流程的管理器
    # 它会初始化数据加载器、模型、优化器、评估指标等所有组件
    runner = Runner(local_rank, world_rank, world_size, cfg)

    # ===== 模式选择：评估模式 vs 训练模式 =====
    if cfg.ckpt is not None:
        # ===== 评估模式：从检查点加载模型并进行评估 =====
        # 当提供了检查点文件时，跳过训练直接进行模型评估
        
        print("Loading checkpoints for evaluation...")
        
        # 加载所有指定的检查点文件
        # cfg.ckpt 是检查点文件路径列表，支持加载多个检查点（用于模型集成）
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        # weights_only=True: 只加载模型权重，提高安全性和加载速度
        # map_location=runner.device: 将模型直接加载到目标设备上
        
        # 合并多个检查点的参数（如果提供多个检查点）
        # 这种设计支持分布式训练中每个进程保存自己负责的高斯点
        for k in runner.splats.keys():
            # 遍历所有参数类型（means, scales, quats, opacities等）
            # 将来自不同检查点的同类参数在第0维度上拼接
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        
        # 参数合并的意义：
        # - 分布式训练时每个进程只保存部分高斯点
        # - 评估时需要将所有进程的高斯点合并成完整模型
        # - torch.cat在dim=0上拼接，恢复完整的点云数据
        
        # 获取训练步数（用于结果文件命名）
        step = ckpts[0]["step"]  # 使用第一个检查点的步数作为标识
        
        # ===== 执行各种评估任务 =====
        
        # 1. 标准模型评估：在验证集上计算图像质量指标
        runner.eval(step=step)
        
        # 2. 轨迹渲染：生成演示视频展示不同视角的渲染效果
        runner.render_traj(step=step)
        
        # 3. 模型压缩（如果启用）：测试压缩算法对模型质量的影响
        if cfg.compression is not None:
            runner.run_compression(step=step)
        
        print("Evaluation completed.")
        
    else:
        # ===== 训练模式：从头开始训练模型 =====
        # 当没有提供检查点时，执行完整的训练流程
        
        print("Starting training from scratch...")
        runner.train()
        
        # 训练过程包括：
        # - 数据加载和预处理
        # - 前向传播和损失计算
        # - 反向传播和参数更新
        # - 密集化策略执行
        # - 定期评估和检查点保存
        # - 实时可视化监控

    # ===== 可视化查看器管理 =====
    # 如果启用了查看器，保持其运行状态供用户交互
    if not cfg.disable_viewer:
        # 完成查看器的初始化设置
        runner.viewer.complete()
        
        # 提示用户查看器已就绪
        print("Viewer running... Ctrl+C to exit.")
        
        # 保持程序运行，让用户可以通过Web界面查看结果
        # 使用长时间sleep而不是无限循环，便于响应键盘中断
        time.sleep(1000000)  # 睡眠约11.5天，实际上用户会通过Ctrl+C退出
        
        # 查看器的功能：
        # - 实时显示训练进度和渲染结果
        # - 提供交互式的相机控制
        # - 支持不同渲染模式的切换
        # - 显示性能统计信息

    print("Program completed.")


if __name__ == "__main__":
    """
    程序入口：处理命令行参数和配置，启动3D高斯点云训练程序
    
    该入口块负责：
    1. 提供使用说明和示例命令
    2. 定义不同的训练配置选项
    3. 处理命令行参数解析
    4. 根据配置动态导入必要的依赖包
    5. 验证配置的合理性
    6. 启动主程序
    """
    
    # ===== 使用说明文档 =====
    """
    Usage:

    ```bash
    # Single GPU training
    # 单GPU训练示例：使用9号GPU，采用默认配置
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    # 分布式训练示例：使用4个GPU（0,1,2,3），由于有效批大小变为4倍，所以训练步数缩减为1/4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """
    # 使用说明解释：
    # - CUDA_VISIBLE_DEVICES: 指定使用的GPU设备
    # - python -m examples.simple_trainer: 以模块方式运行程序
    # - default/mcmc: 选择预定义的配置模板
    # - --steps_scaler: 调整训练步数的缩放因子

    # ===== 预定义配置选项 =====
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    # 可选择的配置对象，每个都是(命令行描述, 配置对象)的元组
    configs = {
        "default": (
            # 默认配置：使用原始论文中的密集化启发式方法进行高斯点云训练
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),  # 使用默认策略，启用详细输出
            ),
        ),
        "mcmc": (
            # MCMC配置：使用MCMC论文中的密集化方法进行训练
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                # MCMC策略的特殊参数设置
                init_opa=0.5,           # 更高的初始不透明度
                init_scale=0.1,         # 更小的初始尺度
                opacity_reg=0.01,       # 不透明度正则化权重
                scale_reg=0.01,         # 尺度正则化权重
                strategy=MCMCStrategy(verbose=True),  # 使用MCMC策略
            ),
        ),
    }
    
    # ===== 命令行参数解析 =====
    # 使用tyro库创建可重写的配置命令行接口
    # 用户可以选择预定义配置，也可以通过命令行参数覆盖任何配置项
    cfg = tyro.extras.overridable_config_cli(configs)
    
    # tyro.extras.overridable_config_cli的功能：
    # 1. 根据configs字典创建命令行选择器
    # 2. 允许用户通过--参数名的方式覆盖配置
    # 3. 自动生成帮助信息和类型检查
    # 4. 支持嵌套配置对象的参数覆盖
    
    # ===== 训练步数调整 =====
    # 根据步数缩放因子调整所有与步数相关的参数
    cfg.adjust_steps(cfg.steps_scaler)
    
    # 步数调整的意义：
    # - 分布式训练时有效批大小增加，需要相应减少训练步数
    # - 保持训练收敛性和效果的一致性
    # - 自动调整评估、保存等关键步数节点

    # ===== 条件依赖导入：双边网格功能 =====
    # Import BilateralGrid and related functions based on configuration
    # 根据配置条件导入双边网格相关函数，避免不必要的依赖
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            # 使用融合版本的双边网格实现（性能更优）
            cfg.use_bilateral_grid = True  # 确保标志位正确设置
            from fused_bilagrid import (
                BilateralGrid,          # 双边网格主类
                color_correct,          # 颜色校正函数
                slice,                  # 网格切片函数
                total_variation_loss,   # 总变差损失函数
            )
        else:
            # 使用标准版本的双边网格实现
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
    
    # 条件导入的优点：
    # 1. 只在需要时导入，减少启动时间
    # 2. 避免可选依赖缺失导致的错误
    # 3. 支持不同实现版本之间的切换

    # ===== 条件依赖导入：PNG压缩功能 =====
    # try import extra dependencies
    # 尝试导入额外的依赖包，用于PNG压缩功能
    if cfg.compression == "png":
        try:
            import plas      # PLAS压缩库
            import torchpq   # PyTorch量化库
        except ImportError:
            # 如果依赖缺失，提供详细的安装指导
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )
    
    # 依赖检查的重要性：
    # 1. PNG压缩是可选功能，不应该影响基础训练
    # 2. 提供明确的安装指导，帮助用户解决依赖问题
    # 3. 使用try-except确保程序的健壮性

    # ===== 配置一致性检查 =====
    # 验证无迹变换(Unscented Transform)相关配置的一致性
    if cfg.with_ut:
        # 如果启用无迹变换，必须同时启用3D评估
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."
    
    # 配置验证的意义：
    # 1. 确保相关功能的配置组合是有效的
    # 2. 在程序启动时就发现配置问题，而不是在训练过程中
    # 3. 提供清晰的错误提示，帮助用户正确配置

    # ===== 启动主程序 =====
    # 使用gsplat提供的cli函数启动分布式训练或单GPU训练
    cli(main, cfg, verbose=True)
    
    # cli函数的功能：
    # 1. 自动检测可用的GPU数量
    # 2. 根据GPU数量决定是否启用分布式训练
    # 3. 为每个GPU进程分配local_rank和world_rank
    # 4. 处理分布式训练的初始化和同步
    # 5. 调用main函数开始实际的训练或评估
    
    # 参数说明：
    # - main: 要执行的主函数
    # - cfg: 完整的配置对象
    # - verbose=True: 启用详细输出，显示分布式训练信息
