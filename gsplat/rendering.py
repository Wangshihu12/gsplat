import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    RollingShutterType,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    fully_fused_projection,
    fully_fused_projection_2dgs,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    rasterize_to_pixels_eval3d,
    spherical_harmonics,
)
from .distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)
from .utils import depth_to_normal, get_projection_matrix


def rasterization(
    means: Tensor,  # [..., N, 3] 高斯点的3D中心位置
    quats: Tensor,  # [..., N, 4] 高斯点的旋转四元数（wxyz格式），无需归一化
    scales: Tensor,  # [..., N, 3] 高斯点的尺度参数
    opacities: Tensor,  # [..., N] 高斯点的不透明度
    colors: Tensor,  # [..., (C,) N, D] 或 [..., (C,) N, K, 3] 颜色值或球谐系数
    viewmats: Tensor,  # [..., C, 4, 4] 世界到相机的变换矩阵
    Ks: Tensor,  # [..., C, 3, 3] 相机内参矩阵
    width: int,  # 渲染图像宽度
    height: int,  # 渲染图像高度
    near_plane: float = 0.01,  # 近裁剪平面距离
    far_plane: float = 1e10,  # 远裁剪平面距离
    radius_clip: float = 0.0,  # 2D半径裁剪阈值，用于加速大规模场景
    eps2d: float = 0.3,  # 2D协方差矩阵的数值稳定性参数
    sh_degree: Optional[int] = None,  # 球谐函数阶数，None时colors为直接颜色值
    packed: bool = True,  # 是否使用打包模式（内存高效但可能稍慢）
    tile_size: int = 16,  # 光栅化使用的瓦片大小
    backgrounds: Optional[Tensor] = None,  # [..., C, D] 背景颜色
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",  # 渲染模式
    sparse_grad: bool = False,  # 是否使用COO稀疏梯度布局
    absgrad: bool = False,  # 是否计算2D投影点的绝对梯度
    rasterize_mode: Literal["classic", "antialiased"] = "classic",  # 光栅化模式
    channel_chunk: int = 32,  # 通道分块大小，用于大通道数的渲染
    distributed: bool = False,  # 是否使用多GPU分布式渲染
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",  # 相机模型
    segmented: bool = False,  # 是否使用分段基数排序
    covars: Optional[Tensor] = None,  # [..., N, 3, 3] 可选的协方差矩阵
    with_ut: bool = False,  # 是否使用无迹变换进行投影
    with_eval3d: bool = False,  # 是否在3D世界空间计算高斯响应
    # 相机畸变参数
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] 或 [..., C, 4] 径向畸变系数
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2] 切向畸变系数
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4] 薄棱镜畸变系数
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,  # F-Theta相机畸变参数
    # 卷帘快门参数
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,  # 卷帘快门类型
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4] 卷帘快门的第二视图矩阵
) -> Tuple[Tensor, Tensor, Dict]:
    """
    3D高斯点云光栅化渲染核心函数
    
    将一组3D高斯点(N个)光栅化到一批图像平面(C个)上。这是3D高斯点云渲染的
    最核心函数，支持众多高级功能：多GPU分布式渲染、批量渲染、N维特征支持、
    深度渲染、内存-速度权衡、稀疏梯度、抗锯齿渲染、相机畸变等。
    
    主要功能包括：
    1. 将3D高斯点投影到2D图像平面
    2. 处理球谐函数颜色表示或直接颜色值
    3. 支持多种渲染模式（RGB、深度、组合）
    4. 优化的瓦片化光栅化算法
    5. 分布式多GPU训练支持
    6. 各种高级渲染技术
    
    @param means: 高斯点的3D中心位置坐标
    @param quats: 高斯椭球的旋转四元数，使用wxyz约定
    @param scales: 高斯椭球在三个轴向的尺度参数
    @param opacities: 高斯点的不透明度值
    @param colors: 颜色信息，可以是直接RGB值或球谐系数
    @param viewmats: 相机的世界到相机坐标系变换矩阵
    @param Ks: 相机内参矩阵
    @param width, height: 目标渲染图像的像素尺寸
    
    其他参数详见函数签名注释...
    
    @return: 三元组 (render_colors, render_alphas, meta)
        - render_colors: 渲染的颜色图像 [..., C, height, width, X]
        - render_alphas: 渲染的alpha通道 [..., C, height, width, 1]
        - meta: 包含中间计算结果的字典
    """
    
    # ===== 初始化和参数解析 =====
    meta = {}  # 存储中间计算结果的字典

    # 解析张量的批次维度信息
    batch_dims = means.shape[:-2]  # 除最后两个维度外的所有维度
    num_batch_dims = len(batch_dims)  # 批次维度数量
    B = math.prod(batch_dims)  # 批次总数
    N = means.shape[-2]  # 高斯点总数
    C = viewmats.shape[-3]  # 相机总数  
    I = B * C  # 总图像数量
    device = means.device  # 计算设备
    
    # ===== 输入参数验证 =====
    # 验证means张量的形状
    assert means.shape == batch_dims + (N, 3), means.shape
    
    if covars is None:
        # 如果没有提供协方差矩阵，则需要四元数和尺度参数
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
    else:
        # 如果提供了协方差矩阵，则忽略四元数和尺度参数
        assert covars.shape == batch_dims + (N, 3, 3), covars.shape
        quats, scales = None, None
        # 将3x3协方差矩阵转换为6D上三角向量表示（内存高效）
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    
    # 验证其他参数的形状
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape  
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    # ===== 分布式训练辅助函数 =====
    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        """
        重新整形分布式数据的视图
        将来自不同rank的数据按相机重新组织
        """
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    # ===== 颜色数据验证 =====
    if sh_degree is None:
        # 颜色作为后激活值，形状应为 [..., N, D] 或 [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
        if distributed:
            assert (
                colors.dim() == num_batch_dims + 2
            ), "Distributed mode only supports per-Gaussian colors."
    else:
        # 颜色作为球谐系数，形状应为 [..., N, K, 3] 或 [..., C, N, K, 3]
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        # 验证球谐阶数和系数数量的一致性
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert (
                colors.dim() == num_batch_dims + 3
            ), "Distributed mode only supports per-Gaussian colors."

    # ===== 功能兼容性检查 =====
    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # 畸变和卷帘快门需要无迹变换支持
    if (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
        or ftheta_coeffs is not None
        or rolling_shutter != RollingShutterType.GLOBAL
    ):
        assert (
            with_ut
        ), "Distortion and rolling shutter are only supported with `with_ut=True`."

    # 卷帘快门参数验证
    if rolling_shutter != RollingShutterType.GLOBAL:
        assert (
            viewmats_rs is not None
        ), "Rolling shutter requires to provide viewmats_rs."
    else:
        assert (
            viewmats_rs is None
        ), "viewmats_rs should be None for global rolling shutter."

    # UT和eval3d功能验证
    if with_ut or with_eval3d:
        assert (quats is not None) and (
            scales is not None
        ), "UT and eval3d requires to provide quats and scales."
        assert packed is False, "Packed mode is not supported with UT."
        assert sparse_grad is False, "Sparse grad is not supported with UT."

    # ===== 多GPU分布式训练处理 =====
    # 实现论文《On Scaling Up 3D Gaussian Splatting Training》中提出的多GPU策略
    # 在分布式模式下，将投影计算分布在高斯点上，将光栅化计算分布在相机上
    if distributed:
        assert batch_dims == (), "Distributed mode does not support batch dimensions"
        world_rank = torch.distributed.get_rank()  # 当前进程排名
        world_size = torch.distributed.get_world_size()  # 总进程数

        # 收集各个rank中的高斯点数量
        N_world = all_gather_int32(world_size, N, device=device)

        # 确保所有rank的相机数量相同
        C_world = [C] * world_size
        # 从所有rank收集相机参数用于投影计算
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])
        if viewmats_rs is not None:
            (viewmats_rs,) = all_gather_tensor_list(world_size, [viewmats_rs])

        # 更新C从本地相机数量到全局相机数量
        C = len(viewmats)

    # ===== 3D到2D投影计算 =====
    if with_ut:
        # 使用无迹变换的融合投影（支持畸变和卷帘快门）
        proj_results = fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,  # 用不透明度计算更紧的半径边界
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=(rasterize_mode == "antialiased"),  # 抗锯齿需要补偿因子
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
        )
    else:
        # 标准融合投影：直接传入{quats, scales}比预计算协方差更快
        proj_results = fully_fused_projection(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            packed=packed,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sparse_grad=sparse_grad,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            opacities=opacities,  # 用不透明度计算更紧的半径边界
        )

    # ===== 投影结果处理 =====
    if packed:
        # 打包模式：结果被打包为形状 [nnz, ...] 的稀疏张量，所有元素都有效
        (
            batch_ids,      # 批次ID
            camera_ids,     # 相机ID
            gaussian_ids,   # 高斯点ID
            radii,          # 2D投影半径
            means2d,        # 2D投影中心
            depths,         # 深度值
            conics,         # 2D二次曲线参数
            compensations,  # 抗锯齿补偿因子
        ) = proj_results
        # 根据索引提取对应的不透明度
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]  # [nnz]
        image_ids = batch_ids * C + camera_ids  # 计算全局图像ID
    else:
        # 非打包模式：结果形状为 [..., C, N, ...]，只有半径>0的元素有效
        radii, means2d, depths, conics, compensations = proj_results
        # 广播不透明度到所有相机
        opacities = torch.broadcast_to(
            opacities[..., None, :], batch_dims + (C, N)
        )  # [..., C, N]
        batch_ids, camera_ids, gaussian_ids = None, None, None
        image_ids = None

    # ===== 应用抗锯齿补偿 =====
    if compensations is not None:
        # 将补偿因子应用到不透明度上，实现抗锯齿效果
        opacities = opacities * compensations

    # ===== 更新元数据字典 =====
    meta.update(
        {
            # 全局批次和相机ID
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            # 本地高斯点ID
            "gaussian_ids": gaussian_ids,
            "radii": radii,           # 2D投影半径
            "means2d": means2d,       # 2D投影中心
            "depths": depths,         # 深度值
            "conics": conics,         # 二次曲线参数
            "opacities": opacities,   # 处理后的不透明度
        }
    )

    # ===== 颜色数据处理 =====
    # 将颜色转换为 [..., C, N, D] 或 [..., nnz, D] 格式传入光栅化函数
    if sh_degree is None:
        # 颜色是后激活值，形状为 [..., N, D] 或 [..., C, N, D]
        if packed:
            if colors.dim() == num_batch_dims + 2:
                # 将 [..., N, D] 转换为 [nnz, D]
                colors = colors.view(B, N, -1)[batch_ids, gaussian_ids]
            else:
                # 将 [..., C, N, D] 转换为 [nnz, D]
                colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
        else:
            if colors.dim() == num_batch_dims + 2:
                # 将 [..., N, D] 转换为 [..., C, N, D]
                colors = torch.broadcast_to(
                    colors[..., None, :, :], batch_dims + (C, N, -1)
                )
            else:
                # colors已经是 [..., C, N, D] 格式
                pass
    else:
        # 颜色是球谐系数，需要计算观察方向相关的颜色
        # 计算相机位置（世界坐标）
        campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        if viewmats_rs is not None:
            # 卷帘快门情况下，使用两个时刻相机位置的平均值
            campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
            campos = 0.5 * (campos + campos_rs)  # [..., C, 3]
            
        if packed:
            # 计算从高斯点到相机的观察方向向量
            dirs = (
                means.view(B, N, 3)[batch_ids, gaussian_ids]
                - campos.view(B, C, 3)[batch_ids, camera_ids]
            )  # [nnz, 3]
            masks = (radii > 0).all(dim=-1)  # [nnz] 有效性掩码
            
            if colors.dim() == num_batch_dims + 3:
                # 将 [..., N, K, 3] 转换为 [nnz, K, 3]
                shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]  # [nnz, K, 3]
            else:
                # 将 [..., C, N, K, 3] 转换为 [nnz, K, 3]
                shs = colors.view(B, C, N, -1, 3)[
                    batch_ids, camera_ids, gaussian_ids
                ]  # [nnz, K, 3]
            # 计算球谐函数颜色
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
        else:
            # 计算从高斯点到相机的观察方向向量
            dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
            masks = (radii > 0).all(dim=-1)  # [..., C, N] 有效性掩码
            
            if colors.dim() == num_batch_dims + 3:
                # 将 [..., N, K, 3] 转换为 [..., C, N, K, 3]
                shs = torch.broadcast_to(
                    colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
                )
            else:
                # colors已经是 [..., C, N, K, 3] 格式
                shs = colors
            # 计算球谐函数颜色
            colors = spherical_harmonics(
                sh_degree, dirs, shs, masks=masks
            )  # [..., C, N, 3]
        
        # 使其与Inria CUDA后端保持一致：添加偏移并截断负值
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # ===== 分布式数据交换 =====
    # 在分布式模式下，需要将高斯点分散到目标rank，基于它们对哪些相机可见
    # 这些信息在投影阶段已经计算出来了
    if distributed:
        if packed:
            # 计算需要发送到每个rank的元素数量
            cnts = torch.bincount(camera_ids, minlength=C)  # 所有相机
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # 跨所有rank进行全互连通信。此步骤后，每个rank将拥有
            # 渲染自己图像所需的所有高斯点
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # 在发送数据之前，需要将camera_ids从全局转换为本地
            # 即投影阶段产生的camera_ids是全球范围的，需要转换为每个rank本地的
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # 将gaussian_ids从本地转换为全局
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # 跨所有rank进行全互连通信
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # 将C从全局相机数量改为本地相机数量
            C = C_world[world_rank]

        else:
            # 非打包模式的分布式处理
            # 将C从全局相机数量改为本地相机数量
            C = C_world[world_rank]

            # 跨所有rank进行全互连通信
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # ===== 根据渲染模式处理颜色和深度 =====
    if render_mode in ["RGB+D", "RGB+ED"]:
        # RGB+深度模式：将深度作为额外通道添加到颜色中
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            # 为背景添加零深度通道
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode in ["D", "ED"]:
        # 纯深度模式：只渲染深度
        colors = depths[..., None]
        if backgrounds is not None:
            # 背景深度设为零
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)
    else:  # RGB
        # RGB模式：使用原始颜色
        pass

    # ===== 瓦片相交计算 =====
    # 计算图像瓦片的尺寸
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    
    # 识别与每个高斯点相交的瓦片
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,        # 2D投影中心
        radii,          # 投影半径
        depths,         # 深度值
        tile_size,      # 瓦片大小
        tile_width,     # 瓦片宽度数量
        tile_height,    # 瓦片高度数量
        segmented=segmented,  # 是否使用分段排序
        packed=packed,        # 是否打包模式
        n_images=I,          # 图像总数
        image_ids=image_ids, # 图像ID
        gaussian_ids=gaussian_ids,  # 高斯点ID
    )
    
    # 编码相交偏移量，用于快速瓦片访问
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # ===== 更新元数据 =====
    meta.update(
        {
            "tile_width": tile_width,           # 瓦片宽度数量
            "tile_height": tile_height,         # 瓦片高度数量
            "tiles_per_gauss": tiles_per_gauss, # 每个高斯点的瓦片数
            "isect_ids": isect_ids,             # 相交ID
            "flatten_ids": flatten_ids,         # 扁平化ID
            "isect_offsets": isect_offsets,     # 相交偏移量
            "width": width,                     # 图像宽度
            "height": height,                   # 图像高度
            "tile_size": tile_size,             # 瓦片大小
            "n_batches": B,                     # 批次数量
            "n_cameras": C,                     # 相机数量
        }
    )

    # ===== 像素光栅化 =====
    # 如果颜色通道数超过chunk大小，则分块处理
    if colors.shape[-1] > channel_chunk:
        # 分块处理大通道数的渲染
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        
        for i in range(n_chunks):
            # 提取当前chunk的颜色和背景
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            
            if with_eval3d:
                # 3D世界空间评估模式
                render_colors_, render_alphas_ = rasterize_to_pixels_eval3d(
                    means,      # 3D高斯点位置
                    quats,      # 旋转四元数
                    scales,     # 尺度参数
                    colors_chunk,    # 当前chunk颜色
                    opacities,       # 不透明度
                    viewmats,        # 视图矩阵
                    Ks,              # 内参矩阵
                    width, height,   # 图像尺寸
                    tile_size,       # 瓦片大小
                    isect_offsets,   # 相交偏移
                    flatten_ids,     # 扁平化ID
                    backgrounds=backgrounds_chunk,  # 背景
                    camera_model=camera_model,      # 相机模型
                    radial_coeffs=radial_coeffs,    # 径向畸变
                    tangential_coeffs=tangential_coeffs,  # 切向畸变
                    thin_prism_coeffs=thin_prism_coeffs,  # 薄棱镜畸变
                    ftheta_coeffs=ftheta_coeffs,          # F-Theta畸变
                    rolling_shutter=rolling_shutter,      # 卷帘快门
                    viewmats_rs=viewmats_rs,              # 卷帘快门视图矩阵
                )
            else:
                # 标准2D光栅化模式
                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,         # 2D投影位置
                    conics,          # 二次曲线参数
                    colors_chunk,    # 当前chunk颜色
                    opacities,       # 不透明度
                    width, height,   # 图像尺寸
                    tile_size,       # 瓦片大小
                    isect_offsets,   # 相交偏移
                    flatten_ids,     # 扁平化ID
                    backgrounds=backgrounds_chunk,  # 背景
                    packed=packed,   # 打包模式
                    absgrad=absgrad, # 绝对梯度
                )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
            
        # 合并所有chunk的结果
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # alpha通道相同，丢弃重复
    else:
        # 单次处理所有通道
        if with_eval3d:
            # 3D世界空间评估模式
            render_colors, render_alphas = rasterize_to_pixels_eval3d(
                means, quats, scales, colors, opacities,
                viewmats, Ks, width, height, tile_size,
                isect_offsets, flatten_ids,
                backgrounds=backgrounds,
                camera_model=camera_model,
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                thin_prism_coeffs=thin_prism_coeffs,
                ftheta_coeffs=ftheta_coeffs,
                rolling_shutter=rolling_shutter,
                viewmats_rs=viewmats_rs,
            )
        else:
            # 标准2D光栅化模式
            render_colors, render_alphas = rasterize_to_pixels(
                means2d, conics, colors, opacities,
                width, height, tile_size,
                isect_offsets, flatten_ids,
                backgrounds=backgrounds,
                packed=packed,
                absgrad=absgrad,
            )
    
    # ===== 期望深度计算 =====
    if render_mode in ["ED", "RGB+ED"]:
        # 对于期望深度模式，需要将累积深度标准化为期望深度
        # 期望深度 = 累积深度 / 累积alpha
        render_colors = torch.cat(
            [
                render_colors[..., :-1],  # 保持其他通道不变
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),  # 标准化深度
            ],
            dim=-1,
        )

    # ===== 返回渲染结果 =====
    return render_colors, render_alphas, meta
    
    # 返回值说明：
    # - render_colors: 渲染的颜色/深度图像，形状为 [..., C, height, width, X]
    #   其中X取决于render_mode：RGB时为D，D/ED时为1，RGB+D/RGB+ED时为D+1
    # - render_alphas: 渲染的alpha通道，形状为 [..., C, height, width, 1]
    # - meta: 包含中间计算结果的字典，用于调试和进一步处理


def _rasterization(
    means: Tensor,  # [..., N, 3] 高斯点的3D中心位置
    quats: Tensor,  # [..., N, 4] 高斯点的旋转四元数（wxyz格式）
    scales: Tensor,  # [..., N, 3] 高斯点的尺度参数
    opacities: Tensor,  # [..., N] 高斯点的不透明度
    colors: Tensor,  # [..., (C,) N, D] 或 [..., (C,) N, K, 3] 颜色值或球谐系数
    viewmats: Tensor,  # [..., C, 4, 4] 世界到相机的变换矩阵
    Ks: Tensor,  # [..., C, 3, 3] 相机内参矩阵
    width: int,  # 渲染图像宽度（像素）
    height: int,  # 渲染图像高度（像素）
    near_plane: float = 0.01,  # 近裁剪平面距离
    far_plane: float = 1e10,  # 远裁剪平面距离
    eps2d: float = 0.3,  # 2D协方差矩阵特征值的数值稳定性参数
    sh_degree: Optional[int] = None,  # 球谐函数阶数，None时colors为直接颜色值
    tile_size: int = 16,  # 光栅化瓦片大小（像素）
    backgrounds: Optional[Tensor] = None,  # [..., C, D] 背景颜色
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",  # 渲染模式
    rasterize_mode: Literal["classic", "antialiased"] = "classic",  # 光栅化模式
    channel_chunk: int = 32,  # 通道分块大小，用于处理高维特征
    batch_per_iter: int = 100,  # 每次迭代处理的批次大小（内存管理）
) -> Tuple[Tensor, Tensor, Dict]:
    """
    基于PyTorch自动求导的3D高斯点云光栅化函数
    
    这是rasterization()函数的一个变体版本，主要特点：
    1. 利用PyTorch的autograd机制进行反向传播，而不是完全自定义的CUDA后端
    2. 整个可微分计算图基于PyTorch (和nerfacc)，可以使用标准的autograd
    3. 实现更简单，但性能可能不如完全优化的CUDA版本
    4. 适用于研究和原型开发，便于调试和修改
    
    注意事项：
    - 仍然依赖gsplat的CUDA后端进行某些计算，但主要的可微分流程使用PyTorch
    - 需要安装最新版本的nerfacc：pip install git+https://github.com/nerfstudio-project/nerfacc
    - 相比rasterization()，不支持某些参数如`packed`、`sparse_grad`和`absgrad`
    
    @param means: 高斯点的3D世界坐标中心位置
    @param quats: 高斯椭球的旋转四元数，用于定义椭球的方向
    @param scales: 高斯椭球在三个主轴方向的尺度
    @param opacities: 高斯点的不透明度，控制透明度混合
    @param colors: 颜色信息，可以是RGB值或球谐系数
    @param viewmats: 相机视图矩阵（世界到相机坐标变换）
    @param Ks: 相机内参矩阵，包含焦距、主点等投影参数
    @param width, height: 目标渲染图像的像素分辨率
    @param near_plane, far_plane: 视锥体的近远裁剪平面
    @param eps2d: 用于数值稳定性的小值，防止投影后的高斯过小
    @param sh_degree: 球谐函数的最大阶数，控制颜色的角度复杂性
    @param tile_size: 图像瓦片的边长，用于并行化渲染
    @param backgrounds: 背景颜色，用于alpha混合
    @param render_mode: 渲染内容类型（RGB、深度或组合）
    @param rasterize_mode: 光栅化算法类型（经典或抗锯齿）
    @param channel_chunk: 特征通道分块大小，用于内存管理
    @param batch_per_iter: 批处理大小，控制内存使用
    
    @return: 三元组 (render_colors, render_alphas, meta)
    """
    
    # ===== 导入必要的PyTorch实现函数 =====
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,      # 融合的3D到2D投影函数
        _quat_scale_to_covar_preci,  # 四元数和尺度转协方差矩阵函数
        _rasterize_to_pixels,        # 像素光栅化函数
    )

    # ===== 解析批次维度和基本参数 =====
    batch_dims = means.shape[:-2]  # 获取批次维度，排除最后的[N, 3]
    num_batch_dims = len(batch_dims)  # 批次维度的数量
    B = math.prod(batch_dims)  # 总批次数量
    N = means.shape[-2]  # 高斯点总数
    C = viewmats.shape[-3]  # 相机总数
    I = B * C  # 总图像数量（批次×相机）
    device = means.device  # 获取张量所在的设备
    
    # ===== 输入参数验证 =====
    # 验证所有输入张量的形状是否符合预期
    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    # ===== 颜色数据格式验证 =====
    if sh_degree is None:
        # 颜色作为后激活值处理，应为 [..., N, D] 或 [..., C, N, D] 形状
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # 颜色作为球谐系数处理，应为 [..., N, K, 3] 或 [..., C, N, K, 3] 形状
        # 支持激活部分球谐频带（partial SH bands）
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        # 验证球谐阶数与系数数量的一致性
        # (sh_degree + 1)² 应该不超过提供的系数数量
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # ===== 3D高斯点投影到2D =====
    # Project Gaussians to 2D.
    # The results are with shape [..., C, N, ...]. Only the elements with radii > 0 are valid.
    
    # 第一步：将四元数和尺度参数转换为协方差矩阵
    # 这是高斯椭球在3D空间中形状的数学表示
    covars, _ = _quat_scale_to_covar_preci(
        quats,      # 旋转四元数
        scales,     # 尺度参数
        True,       # 计算协方差矩阵
        False,      # 不计算精度矩阵
        triu=False  # 返回完整矩阵而非上三角
    )
    
    # 第二步：执行融合的3D到2D投影变换
    # 这个函数将3D高斯椭球投影到2D图像平面上
    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,      # 3D高斯点中心位置
        covars,     # 3D协方差矩阵
        viewmats,   # 相机视图变换矩阵
        Ks,         # 相机内参矩阵
        width,      # 图像宽度
        height,     # 图像高度
        eps2d=eps2d,        # 数值稳定性参数
        near_plane=near_plane,  # 近裁剪平面
        far_plane=far_plane,    # 远裁剪平面
        calc_compensations=(rasterize_mode == "antialiased"),  # 是否计算抗锯齿补偿
    )
    
    # 投影结果解释：
    # - radii: 2D投影后的椭圆半径 [..., C, N, 2]
    # - means2d: 2D投影中心点坐标 [..., C, N, 2]
    # - depths: 高斯点的深度值 [..., C, N]
    # - conics: 2D二次曲线参数，用于快速椭圆测试 [..., C, N, 3]
    # - compensations: 抗锯齿补偿因子（可选）[..., C, N]
    
    # 将不透明度广播到所有相机视角
    opacities = torch.broadcast_to(
        opacities[..., None, :], batch_dims + (C, N)
    )  # [..., C, N]
    
    # 在这个实现中，不使用打包模式，所以这些ID都设为None
    batch_ids, camera_ids, gaussian_ids = None, None, None
    image_ids = None

    # ===== 应用抗锯齿补偿 =====
    if compensations is not None:
        # 将补偿因子应用到不透明度上
        # 这实现了Mip-Splatting论文中的视角相关补偿
        # 补偿公式：ρ = sqrt(Det(Σ) / Det(Σ + εI))
        opacities = opacities * compensations

    # ===== 瓦片相交计算 =====
    # Identify intersecting tiles - 识别与每个高斯点相交的图像瓦片
    
    # 计算图像需要的瓦片数量
    tile_width = math.ceil(width / float(tile_size))   # 水平方向瓦片数
    tile_height = math.ceil(height / float(tile_size)) # 垂直方向瓦片数
    
    # 计算每个高斯点与哪些瓦片相交
    # 这是光栅化算法的关键步骤，用于空间剔除和并行化
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,        # 2D投影中心点
        radii,          # 投影半径
        depths,         # 深度值（用于深度排序）
        tile_size,      # 瓦片大小
        tile_width,     # 瓦片宽度数量
        tile_height,    # 瓦片高度数量
        packed=False,   # 不使用打包模式（与主函数不同）
        n_images=I,     # 总图像数量
        image_ids=image_ids,      # 图像ID（此处为None）
        gaussian_ids=gaussian_ids, # 高斯点ID（此处为None）
    )
    
    # 编码相交偏移量，用于高效的瓦片访问
    # 这个数据结构允许快速找到每个瓦片中的高斯点
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # ===== 颜色数据预处理 =====
    # Turn colors into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    # 将颜色转换为适合传入像素光栅化函数的格式
    
    if sh_degree is None:
        # 颜色是后激活值，形状为 [..., N, D] 或 [..., C, N, D]
        if colors.dim() == num_batch_dims + 2:
            # 将 [..., N, D] 广播为 [..., C, N, D]
            # 这意味着所有相机使用相同的颜色
            colors = torch.broadcast_to(
                colors[..., None, :, :], batch_dims + (C, N, -1)
            )
        else:
            # colors已经是 [..., C, N, D] 格式，无需转换
            pass
    else:
        # 颜色是球谐系数，需要根据观察方向计算实际颜色
        # Colors are SH coefficients, with shape [..., N, K, 3] or [..., C, N, K, 3]
        
        # 计算相机在世界坐标系中的位置
        camtoworlds = torch.inverse(viewmats)  # [..., C, 4, 4]
        
        # 计算从高斯点指向相机的观察方向向量
        # 这个方向决定了球谐函数的计算结果
        dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]  # [..., C, N, 3]
        
        # 创建有效性掩码，只有半径大于0的高斯点才参与渲染
        masks = (radii > 0).all(dim=-1)  # [..., C, N]
        
        if colors.dim() == num_batch_dims + 3:
            # 将 [..., N, K, 3] 广播为 [..., C, N, K, 3]
            # 所有相机使用相同的球谐系数
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors已经是 [..., C, N, K, 3] 格式
            shs = colors
        
        # 使用球谐函数计算视角相关的颜色
        # 这允许高斯点在不同观察角度下显示不同的颜色
        colors = spherical_harmonics(
            sh_degree,  # 球谐函数阶数
            dirs,       # 观察方向
            shs,        # 球谐系数
            masks=masks # 有效性掩码
        )  # [..., C, N, 3]
        
        # 使其与Inria的CUDA后端保持一致
        # 添加0.5的偏移并截断负值，这是为了匹配原始实现的数值行为
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # ===== 根据渲染模式处理颜色和深度 =====
    if render_mode in ["RGB+D", "RGB+ED"]:
        # RGB+深度模式：将深度作为额外通道添加到颜色后面
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            # 为背景添加零深度通道
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode in ["D", "ED"]:
        # 纯深度模式：只渲染深度信息
        colors = depths[..., None]
        if backgrounds is not None:
            # 背景深度设为零
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)
    else:  # RGB
        # RGB模式：保持原始颜色不变
        pass

    # ===== 像素光栅化处理 =====
    if colors.shape[-1] > channel_chunk:
        # 如果颜色通道数超过分块大小，进行分块处理以控制内存使用
        # slice into chunks - 分块处理
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        
        for i in range(n_chunks):
            # 提取当前块的颜色数据
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            # 提取对应的背景数据（如果有）
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            
            # 对当前块进行像素光栅化
            render_colors_, render_alphas_ = _rasterize_to_pixels(
                means2d,         # 2D投影中心点
                conics,          # 二次曲线参数
                colors_chunk,    # 当前块的颜色数据
                opacities,       # 不透明度
                width,           # 图像宽度
                height,          # 图像高度
                tile_size,       # 瓦片大小
                isect_offsets,   # 相交偏移量
                flatten_ids,     # 扁平化ID
                backgrounds=backgrounds_chunk,  # 背景颜色
                batch_per_iter=batch_per_iter, # 批处理大小
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        
        # 合并所有块的渲染结果
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # alpha通道相同，丢弃重复的
    else:
        # 通道数不超过分块大小，一次性处理所有通道
        render_colors, render_alphas = _rasterize_to_pixels(
            means2d,         # 2D投影中心点
            conics,          # 二次曲线参数
            colors,          # 完整的颜色数据
            opacities,       # 不透明度
            width,           # 图像宽度
            height,          # 图像高度
            tile_size,       # 瓦片大小
            isect_offsets,   # 相交偏移量
            flatten_ids,     # 扁平化ID
            backgrounds=backgrounds,        # 背景颜色
            batch_per_iter=batch_per_iter, # 批处理大小
        )

    # ===== 期望深度标准化 =====
    if render_mode in ["ED", "RGB+ED"]:
        # 对于期望深度模式，需要将累积深度标准化
        # 期望深度 = 累积深度 / 累积不透明度
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],  # 保持除最后一个通道外的所有通道
                # 最后一个通道（深度）除以alpha值进行标准化，避免除零
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    # ===== 构建元数据字典 =====
    # 收集所有中间计算结果，用于调试、分析或进一步处理
    meta = {
        "batch_ids": batch_ids,               # 批次ID（此实现中为None）
        "camera_ids": camera_ids,             # 相机ID（此实现中为None）
        "gaussian_ids": gaussian_ids,         # 高斯点ID（此实现中为None）
        "radii": radii,                       # 2D投影半径
        "means2d": means2d,                   # 2D投影中心点
        "depths": depths,                     # 深度值
        "conics": conics,                     # 二次曲线参数
        "opacities": opacities,               # 处理后的不透明度
        "tile_width": tile_width,             # 瓦片宽度数量
        "tile_height": tile_height,           # 瓦片高度数量
        "tiles_per_gauss": tiles_per_gauss,   # 每个高斯点的瓦片数
        "isect_ids": isect_ids,               # 相交ID
        "flatten_ids": flatten_ids,           # 扁平化ID
        "isect_offsets": isect_offsets,       # 相交偏移量
        "width": width,                       # 图像宽度
        "height": height,                     # 图像高度
        "tile_size": tile_size,               # 瓦片大小
        "n_batches": B,                       # 批次数量
        "n_cameras": C,                       # 相机数量
    }
    
    # ===== 返回渲染结果 =====
    return render_colors, render_alphas, meta
    
    # 返回值说明：
    # - render_colors: 渲染的颜色/深度图像 [..., C, height, width, X]
    # - render_alphas: 渲染的alpha通道 [..., C, height, width, 1]  
    # - meta: 包含所有中间计算结果的字典，便于调试和分析


# def rasterization_legacy_wrapper(
#     means: Tensor,  # [N, 3]
#     quats: Tensor,  # [N, 4]
#     scales: Tensor,  # [N, 3]
#     opacities: Tensor,  # [N]
#     colors: Tensor,  # [N, D] or [N, K, 3]
#     viewmats: Tensor,  # [C, 4, 4]
#     Ks: Tensor,  # [C, 3, 3]
#     width: int,
#     height: int,
#     near_plane: float = 0.01,
#     eps2d: float = 0.3,
#     sh_degree: Optional[int] = None,
#     tile_size: int = 16,
#     backgrounds: Optional[Tensor] = None,
#     **kwargs,
# ) -> Tuple[Tensor, Tensor, Dict]:
#     """Wrapper for old version gsplat.

#     .. warning::
#         This function exists for comparison purpose only. So we skip collecting
#         the intermidiate variables, and only return an empty dict.

#     """
#     from gsplat.cuda_legacy._wrapper import (
#         project_gaussians,
#         rasterize_gaussians,
#         spherical_harmonics,
#     )

#     assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
#     C = len(viewmats)

#     render_colors, render_alphas = [], []
#     for cid in range(C):
#         fx, fy = Ks[cid, 0, 0], Ks[cid, 1, 1]
#         cx, cy = Ks[cid, 0, 2], Ks[cid, 1, 2]
#         viewmat = viewmats[cid]

#         means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
#             means3d=means,
#             scales=scales,
#             glob_scale=1.0,
#             quats=quats,
#             viewmat=viewmat,
#             fx=fx,
#             fy=fy,
#             cx=cx,
#             cy=cy,
#             img_height=height,
#             img_width=width,
#             block_width=tile_size,
#             clip_thresh=near_plane,
#         )

#         if colors.dim() == 3:
#             c2w = viewmat.inverse()
#             viewdirs = means - c2w[:3, 3]
#             # viewdirs = F.normalize(viewdirs, dim=-1).detach()
#             if sh_degree is None:
#                 sh_degree = int(math.sqrt(colors.shape[1]) - 1)
#             colors = spherical_harmonics(sh_degree, viewdirs, colors)  # [N, 3]

#         background = (
#             backgrounds[cid]
#             if backgrounds is not None
#             else torch.zeros(colors.shape[-1], device=means.device)
#         )

#         render_colors_, render_alphas_ = rasterize_gaussians(
#             xys=means2d,
#             depths=depths,
#             radii=radii,
#             conics=conics,
#             num_tiles_hit=num_tiles_hit,
#             colors=colors,
#             opacity=opacities[..., None],
#             img_height=height,
#             img_width=width,
#             block_width=tile_size,
#             background=background,
#             return_alpha=True,
#         )
#         render_colors.append(render_colors_)
#         render_alphas.append(render_alphas_[..., None])
#     render_colors = torch.stack(render_colors, dim=0)
#     render_alphas = torch.stack(render_alphas, dim=0)
#     return render_colors, render_alphas, {}


def rasterization_inria_wrapper(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., N, D] or [..., N, K, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Wrapper for Inria's rasterization backend.

    .. warning::
        This function exists for comparison purpose only. Only rendered image is
        returned.

    .. warning::
        Inria's CUDA backend has its own LICENSE, so this function should be used with
        the respect to the original LICENSE at:
        https://github.com/graphdeco-inria/diff-gaussian-rasterization

    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    N = means.shape[-2]
    B = math.prod(batch_dims)
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # flatten all batch dimensions
    means = means.reshape(B, N, 3)
    quats = quats.reshape(B, N, 4)
    scales = scales.reshape(B, N, 3)
    opacities = opacities.reshape(B, N)
    viewmats = viewmats.reshape(B, C, 4, 4)
    Ks = Ks.reshape(B, C, 3, 3)
    if colors.dim() == num_batch_dims + 2:
        colors = colors.reshape(B, N, -1)
    elif colors.dim() == num_batch_dims + 3:
        colors = colors.reshape(B, C, N, -1)

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]

    render_colors = []
    for bid in range(B):
        for cid in range(C):
            FoVx = 2 * math.atan(width / (2 * Ks[bid, cid, 0, 0].item()))
            FoVy = 2 * math.atan(height / (2 * Ks[bid, cid, 1, 1].item()))
            tanfovx = math.tan(FoVx * 0.5)
            tanfovy = math.tan(FoVy * 0.5)

            world_view_transform = viewmats[bid, cid].transpose(0, 1)
            projection_matrix = get_projection_matrix(
                znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
            ).transpose(0, 1)
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            background = (
                backgrounds[bid, cid]
                if backgrounds is not None
                else torch.zeros(3, device=device)
            )

            raster_settings = GaussianRasterizationSettings(
                image_height=height,
                image_width=width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=1.0,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=0 if sh_degree is None else sh_degree,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means2D = torch.zeros_like(means, requires_grad=True, device=device)

            render_colors_ = []
            for i in range(0, channels, 3):
                _colors = colors[bid, ..., i : i + 3]
                if _colors.shape[-1] < 3:
                    pad = torch.zeros(
                        _colors.shape[:-1], 3 - _colors.shape[-1], device=device
                    )
                    _colors = torch.cat([_colors, pad], dim=-1)
                _render_colors_, radii = rasterizer(
                    means3D=means[bid],
                    means2D=means2D[bid],
                    shs=_colors if colors.dim() == 4 else None,
                    colors_precomp=_colors if colors.dim() == 3 else None,
                    opacities=opacities[..., None],
                    scales=scales[bid],
                    rotations=quats[bid],
                    cov3D_precomp=None,
                )
                if _colors.shape[-1] < 3:
                    _render_colors_ = _render_colors_[..., : _colors.shape[-1]]
                render_colors_.append(_render_colors_)
            render_colors_ = torch.cat(render_colors_, dim=-1)

            render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
            render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)
    render_colors = render_colors.reshape(batch_dims + (height, width, channels))
    return render_colors, None, {}


###### 2DGS ######
def rasterization_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: The colors of the Gaussians. [..., (C,) N, D] or [..., (C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [..., C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **render_normals**: The rendered normals. [..., C, height, width, 3].

        **surf_normals**: surface normal from depth. [..., C, height, width, 3]

        **render_distort**: The rendered distortions. [..., C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    if distloss:
        assert render_mode in [
            "D",
            "ED",
            "RGB+D",
            "RGB+ED",
        ], f"distloss requires depth rendering, render_mode should be D, ED, RGB+D, RGB+ED, but got {render_mode}"

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = torch.broadcast_to(
            opacities[..., None, :], batch_dims + (C, N)
        )  # [..., C, N]
        camera_ids, gaussian_ids = None, None
        image_ids = None

    densify = torch.zeros_like(
        means2d, dtype=means.dtype, requires_grad=True, device="cuda"
    )
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # TODO: SH also suport N-D.
    # Compute the per-view colors
    # if not (
    #     colors.dim() == num_batch_dims + 3 and sh_degree is None
    # ):  # silently support [..., C, N, D] color.
    #     colors = (
    #         colors.view(B, N, -1)[batch_ids, gaussian_ids]
    #         if packed
    #         else colors[..., None, :, :].expand((-1,) * num_batch_dims + (C, -1, -1))
    #     )  # [nnz, D] or [..., C, N, 3]
    # else:
    #     if packed:
    #         colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids, :]
    if sh_degree is not None:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        if colors.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors is already [..., C, N, K, 3]
            shs = colors
        colors = spherical_harmonics(
            sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1)
        )  # [nnz, D] or [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)

        if backgrounds is not None:
            backgrounds = torch.cat(
                (backgrounds, torch.zeros_like(backgrounds[..., :1])), dim=-1
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
    else:  # RGB
        pass

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode in ["RGB+ED", "RGB+D"]:
        # render_depths = render_colors[..., -1:]
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = depth_to_normal(
            depth_for_normal, torch.linalg.inv(viewmats), Ks
        ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = torch.einsum(
        "...ij,...hwj->...hwi", torch.linalg.inv(viewmats)[..., :3, :3], render_normals
    )

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


def rasterization_2dgs_inria_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    depth_ratio: int = 0,
    **kwargs,
) -> Tuple[Tuple, Dict]:
    """Wrapper for 2DGS's rasterization backend which is based on Inria's backend.

    Install the 2DGS rasterization backend from
        https://github.com/hbb1/diff-surfel-rasterization

    Credit to Jeffrey Hu https://github.com/jefequien

    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]
    scales = scales[:, :2]  # [N, 2]

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=device)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = torch.zeros_like(means, requires_grad=True, device=device)

        render_colors_ = []
        for i in range(0, channels, 3):
            _colors = colors[..., i : i + 3]
            if _colors.shape[-1] < 3:
                pad = torch.zeros(
                    _colors.shape[0], 3 - _colors.shape[-1], device=device
                )
                _colors = torch.cat([_colors, pad], dim=-1)
            _render_colors_, radii, allmap = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=_colors if colors.dim() == 3 else None,
                colors_precomp=_colors if colors.dim() == 2 else None,
                opacities=opacities[:, None],
                scales=scales,
                rotations=quats,
                cov3D_precomp=None,
            )
            if _colors.shape[-1] < 3:
                _render_colors_ = _render_colors_[:, :, : _colors.shape[-1]]
            render_colors_.append(_render_colors_)
        render_colors_ = torch.cat(render_colors_, dim=-1)

        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)

    # additional maps
    allmap = allmap.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    render_depth_expected = allmap[..., 0:1]
    render_alphas = allmap[..., 1:2]
    render_normal = allmap[..., 2:5]
    render_depth_median = allmap[..., 5:6]
    render_dist = allmap[..., 6:7]

    render_normal = render_normal @ (world_view_transform[:3, :3].T)
    render_depth_expected = render_depth_expected / render_alphas
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # render_depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
    render_depth = (
        render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
    )

    normals_surf = depth_to_normal(render_depth, torch.linalg.inv(viewmats), Ks)
    normals_surf = normals_surf * (render_alphas).detach()

    render_colors = torch.cat([render_colors, render_depth], dim=-1)

    meta = {
        "normals_rend": render_normal,
        "normals_surf": normals_surf,
        "render_distloss": render_dist,
        "means2d": means2D,
        "width": width,
        "height": height,
        "radii": radii.unsqueeze(0),
        "n_cameras": C,
        "gaussian_ids": None,
    }
    return (render_colors, render_alphas), meta
