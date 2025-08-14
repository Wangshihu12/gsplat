import glob
import os
import os.path as osp
import pathlib
import platform
import sys

from setuptools import find_packages, setup

# 从版本文件中读取版本号，避免重复定义
__version__ = None
exec(open("gsplat/version.py", "r").read())

# 项目仓库地址
URL = "https://github.com/nerfstudio-project/gsplat"

# 环境变量配置，控制编译选项
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"  # 是否禁用CUDA编译
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"    # 是否包含调试符号
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"          # 是否包含行信息
MAX_JOBS = os.getenv("MAX_JOBS")                        # 并行编译任务数
need_to_unset_max_jobs = False

# 如果未设置MAX_JOBS环境变量，则设置默认值为10
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def get_ext():
    """
    获取构建扩展的配置。
    
    返回值：
        BuildExtension: 配置好的构建扩展对象，禁用Python ABI后缀并启用ninja构建系统
    """
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    """
    构建CUDA扩展模块的配置。
    
    返回值：
        list: 包含CUDAExtension对象的列表，用于编译CUDA代码
    """
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    # 设置扩展源码目录和获取所有源文件
    extensions_dir = osp.join("gsplat", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    # 初始化宏定义和取消定义列表
    undef_macros = []
    define_macros = []

    # 配置C++编译参数
    extra_compile_args = {"cxx": ["-O3"]}  # 启用O3优化
    if not os.name == "nt":  # 非Windows系统
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]  # 忽略符号比较警告
    
    # 配置链接参数，WITH_SYMBOLS决定是否保留调试符号
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    # 检查OpenMP并行支持并配置相关编译参数
    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"  # macOS不使用OpenMP
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]  # Windows OpenMP标志
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]  # Linux/Unix OpenMP标志
    else:
        print("Compiling without OpenMP...")

    # macOS ARM64架构特殊编译配置
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    # 配置NVCC编译器参数
    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]  # 优化和C++17标准
    
    # 如果启用行信息，添加对应标志
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    
    # ROCm/HIP支持配置（AMD GPU）
    if torch.version.hip:
        # 为旧版本PyTorch定义USE_ROCM宏
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        # NVIDIA GPU配置
        nvcc_flags += ["--expt-relaxed-constexpr"]

    # 抑制GLM/Torch的冗长警告信息
    nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags
    
    # Windows特殊编译配置
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += [
            "-DWIN32_LEAN_AND_MEAN",        # 减少Windows头文件包含
            "-allow-unsupported-compiler",   # 允许不受支持的编译器
        ]

    # 设置包含目录路径
    current_dir = pathlib.Path(__file__).parent.resolve()
    glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")  # GLM数学库路径
    include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

    # 创建CUDA扩展对象
    extension = CUDAExtension(
        "gsplat.csrc",                    # 扩展模块名称
        sources,                          # 源文件列表
        include_dirs=include_dirs,        # 头文件搜索路径
        define_macros=define_macros,      # 预定义宏
        undef_macros=undef_macros,        # 取消定义的宏
        extra_compile_args=extra_compile_args,  # 额外编译参数
        extra_link_args=extra_link_args,  # 额外链接参数
    )
    return [extension]


# 设置包的配置信息
setup(
    name="gsplat",                        # 包名称
    version=__version__,                  # 版本号
    description=" Python package for differentiable rasterization of gaussians",  # 包描述
    keywords="gaussian, splatting, cuda", # 关键词
    url=URL,                             # 项目主页
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",  # 下载地址
    python_requires=">=3.7",            # Python版本要求
    install_requires=[                   # 依赖包列表
        "ninja",                         # 构建系统
        "numpy",                         # 数值计算库
        "jaxtyping",                     # 类型注解库
        "rich>=12",                      # 富文本显示库
        "torch",                         # PyTorch深度学习框架
        "typing_extensions; python_version<'3.8'",  # 类型扩展（Python 3.8以下需要）
    ],
    extras_require={                     # 可选依赖
        # 开发依赖，通过 `pip install gsplat[dev]` 安装
        "dev": [
            "black[jupyter]==22.3.0",    # 代码格式化工具
            "isort==5.10.1",             # 导入排序工具
            "pylint==2.13.4",            # 代码检查工具
            "pytest==7.1.2",             # 测试框架
            "pytest-xdist==2.5.0",       # 并行测试插件
            "typeguard>=2.13.3",         # 运行时类型检查
            "pyyaml==6.0",               # YAML解析库
            "build",                     # 构建工具
            "twine",                     # PyPI上传工具
        ],
    },
    # 根据BUILD_NO_CUDA决定是否包含CUDA扩展
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    # 根据BUILD_NO_CUDA决定是否使用自定义构建命令
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),            # 自动查找包
    # 包含包数据文件
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

# 如果之前设置了MAX_JOBS环境变量，则清除它
if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
