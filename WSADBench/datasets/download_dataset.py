import argparse
import os
import subprocess
import glob
from modelscope.hub.snapshot_download import snapshot_download
import re

# 只有以下数据集需要校验分卷数 (Manifest)
PART_COUNT = {
    'CV_by_I3D/UCF_Crime': 5,
    'CV_by_I3D/XD-violence': 8,
    'CV_by_MViT_32/UCF_Crime': 2,
    'CV_by_MViT_32/XD-violence': 3,
    'CV_by_SlowFast/UCF_Crime': 5,
    'CV_by_SlowFast/XD-violence': 9,
    'CV_by_SlowFast_R50/UCF_Crime': 5,
    'CV_by_SlowFast_R50/XD-violence': 9,
    'CV_by_X3DM/UCF_Crime': 3,
    'CV_by_X3DM/XD-violence': 5,
}
# 定义每个文件夹下包含哪些数据集 (为了支持按文件夹全量串行下载)


def run_cmd(cmd):
    """执行 Shell 命令并检查错误"""
    subprocess.run(cmd, shell=True, check=True)


def process_single_dataset(dataset_prefix, target_dir):
    """
    处理单个数据集：下载 -> 校验 -> 合并 -> 解压 -> 删除
    dataset_prefix 示例: 'CV_by_MViT_32/XD-violence'
    """
    print(f"\n" + "=" * 60)
    print(f"🚀 开始处理数据集: {dataset_prefix}")
    print("=" * 60)
    # 获取相对路径目录和基础名称
    sub_dir = os.path.join(target_dir, os.path.dirname(dataset_prefix))
    base_name = os.path.basename(dataset_prefix)
    # =========================================================
    # 0. 提前检查阶段：如果解压后的文件夹已经存在，直接跳过！
    # =========================================================
    extracted_folder_path = os.path.join(sub_dir, base_name)
    if os.path.exists(extracted_folder_path):
        print(f"⏩ 检测到最终数据目录已存在: {extracted_folder_path}")
        print(f"✅ 该数据集之前已下载并解压完成，无需重复处理，直接跳过！")
        return

    # 1. 下载阶段 (ModelScope 会自动跳过已存在且完整的文件)
    pattern = f"{dataset_prefix}.*"
    try:
        dataset_dir = snapshot_download(
            repo_id='mac4mac/WSADBench-Datasets',
            repo_type='dataset',
            local_dir=target_dir,
            allow_patterns=[pattern]
        )
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    # 2. 查找下载的文件
    # 获取相对路径目录，例如 target_dir/CV_by_MViT_32
    sub_dir = os.path.join(target_dir, os.path.dirname(dataset_prefix))
    base_name = os.path.basename(dataset_prefix)

    # 查找相关的压缩包或分卷
    search_pattern = os.path.join(sub_dir, f"{base_name}.*")
    downloaded_files = glob.glob(search_pattern)

    if not downloaded_files:
        print(f"⚠️ 警告: 未找到匹配 {pattern} 的文件，可能云端不存在该文件。")
        return

    # =========================================================
    # 核心修复：使用正则匹配 .tar00, .tar.gz01 等数字结尾的分卷文件
    # =========================================================
    part_files = sorted([f for f in downloaded_files if re.search(r'\.(tar|tar\.gz)\d+$', f)])

    archive_path = ""

    # 3. 校验与合并阶段
    if part_files:
        expected_count = PART_COUNT.get(dataset_prefix)
        if expected_count is None:
            print(f"❌ 错误: 发现分卷文件，但在 PART_COUNT 字典中未配置 {dataset_prefix} 的预期数量！")
            return

        actual_count = len(part_files)
        print(f"🔍 校验分卷: 预期 {expected_count} 个，实际找到 {actual_count} 个。")

        if actual_count != expected_count:
            print(f"❌ 严重错误: {dataset_prefix} 分卷不完整！请重新运行脚本尝试断点续传。")
            return

        # 核心修复：提取合并后的目标文件名 (去掉末尾的纯数字后缀)
        archive_path = re.sub(r'\d+$', '', part_files[0])

        # 如果合并后的文件还不存在，则执行合并
        if not os.path.exists(archive_path):
            print(f"🔄 正在合并 {actual_count} 个分卷为: {os.path.basename(archive_path)} ...")
            parts_str = " ".join(part_files)
            run_cmd(f"cat {parts_str} > {archive_path}")
        else:
            print(f"⏩ 合并后的文件已存在，跳过合并。")

        # 删除分卷碎片省空间
        print("🗑️ 清理数字后缀的分卷碎片...")
        run_cmd(f"rm -f {' '.join(part_files)}")

    else:
        # 单一压缩包
        archive_path = downloaded_files[0]

    # 4. 解压阶段
    if not archive_path.endswith(('.tar', '.tar.gz')):
        print(f"⏩ {archive_path} 不是压缩包，跳过解压。")
        return

    print(f"📦 正在解压: {os.path.basename(archive_path)} ...")
    try:
        # 提取到所在的子目录中
        run_cmd(f"tar -xf {archive_path} -C {sub_dir}")
        print(f"✅ 解压成功！")

        # 5. 删除原始压缩包
        os.remove(archive_path)
        print(f"🗑️ 已删除原压缩包: {os.path.basename(archive_path)}，释放空间。")

    except Exception as e:
        print(f"❌ 解压失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键流式下载与解压 WSADBench 数据集")
    parser.add_argument(
        "--datasets",
        nargs='+',
        required=True,
        help="指定要下载的数据集前缀列表。例如: CV_by_MViT_32/shanghaitech CV_by_MViT_32/XD-violence"
    )
    args = parser.parse_args()

    # 获取脚本所在目录 (强制锁定在当前目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir.endswith('WSADBench/datasets'):
        print("⚠️ 警告: 建议将此脚本放置在 WSADBench/datasets/ 目录下运行。")

    target_dir = script_dir
    print(f"📂 工作目录设为: {target_dir}")

    # 展开用户输入的参数：如果是文件夹名，就展开为具体的数据集列表
    expanded_datasets = []
    for user_input in args.datasets:
        # 去掉输入末尾可能带有的斜杠，防止匹配失败
        clean_input = user_input.rstrip('/')
        if clean_input in ['CV_by_MViT_32', 'CV_by_SlowFast', 'CV_by_SlowFast_R50', 'CV_by_X3DM', 'CV_by_I3D']:  # Classical_bags_inexact.tar.gz

            expanded_datasets.extend([f"{clean_input}/{ds_name}" for ds_name in ['UCF_Crime', 'XD-violence',  'TAD', 'shanghaitech']])
            print(f"📂 识别到文件夹标识: {clean_input}，已自动展开为 4 个数据集。")
        else:
            expanded_datasets.append(clean_input)

    # 循环遍历展开后的数据集，严格串行：下载 -> 合并 -> 解压 -> 删除
    for ds_prefix in expanded_datasets:
        process_single_dataset(ds_prefix, target_dir)

    print("\n🎉 所有请求的数据集已处理完毕！")