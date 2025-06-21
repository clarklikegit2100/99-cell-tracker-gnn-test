#!/usr/bin/env python3
# build_gt_dict.py
# 依赖库: tifffile, imagecodecs

import os
import argparse
import json
import numpy as np
import tifffile

def build_gt_dict(gt_dir, seg_dir, num_frames=100):
    """
    从GT和SEG文件夹生成(frame, cell_id)到track_id的映射字典。
    gt_dir: GT图像文件夹路径，应包含命名为 man_trackXXX.tif 的帧图像。
    seg_dir: SEG图像文件夹路径，应包含命名为 man_segXXX.tif 的帧图像。
    num_frames: 要处理的帧数。
    返回: 字典 {(frame, cell_id): track_id,...}。
    """
    gt_dict = {}  # 用于存储映射结果
    total_matches = 0  # 计数总匹配对数

    # 遍历每一帧
    for frame in range(1, num_frames+1):
        # 构建当前帧的文件名，按照三位数字格式
        gt_filename = os.path.join(gt_dir, f"man_track{frame:03d}.tif")
        seg_filename = os.path.join(seg_dir, f"man_seg{frame:03d}.tif")

        # 若文件不存在则跳过该帧
        if not os.path.isfile(gt_filename) or not os.path.isfile(seg_filename):
            print(f"帧{frame:03d}的GT或SEG文件缺失，跳过该帧。")
            continue

        # 读取GT和SEG图像为numpy数组
        gt_image = tifffile.imread(gt_filename)
        seg_image = tifffile.imread(seg_filename)

        # 获取GT图像中出现的所有非零track_id
        track_ids = np.unique(gt_image)
        track_ids = track_ids[track_ids != 0]  # 去除0（背景）

        match_count = 0  # 当前帧匹配计数
        for tid in track_ids:
            # 提取该track_id对应的像素区域
            mask = (gt_image == tid)
            if not np.any(mask):
                continue  # 没有像素则跳过

            # 找出mask区域内SEG图像的主要cell_id
            overlapping_cells = seg_image[mask]  # 取出在track区域下的所有seg像素值
            # 若没有非零cell_id，则跳过该track
            overlapping_cells = overlapping_cells[overlapping_cells != 0]
            if overlapping_cells.size == 0:
                continue

            # 统计各cell_id出现次数，找出现频最高的cell_id
            cell_ids, counts = np.unique(overlapping_cells, return_counts=True)
            main_cell = cell_ids[np.argmax(counts)]  # 最大重叠的cell_id

            # 将映射加入字典
            gt_dict[(frame, int(main_cell))] = int(tid)
            match_count += 1
            total_matches += 1

        # 打印当前帧处理日志
        print(f"处理帧 {frame:03d}: 匹配了 {match_count} 个 track_id")

    print(f"所有帧处理完成，共匹配 {total_matches} 对映射。")
    return gt_dict

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="从GT和SEG文件夹生成细胞追踪标签映射字典，并保存为JSON")
    parser.add_argument("--gt_path", required=True, help="GT文件夹路径，包含man_trackXXX.tif文件")
    parser.add_argument("--seg_path", required=True, help="SEG文件夹路径，包含man_segXXX.tif文件")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="处理的帧数（默认100）")
    parser.add_argument("--output", default="gt_dict.json",
                        help="输出JSON文件路径（默认gt_dict.json）")
    args = parser.parse_args()

    gt_dir = args.gt_path
    seg_dir = args.seg_path
    num_frames = args.num_frames
    output_path = args.output

    # 路径合法性检查
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"提供的GT路径不存在或不是文件夹: {gt_dir}")
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"提供的SEG路径不存在或不是文件夹: {seg_dir}")

    # 构建GT_dict映射
    gt_dict = build_gt_dict(gt_dir, seg_dir, num_frames)

    # 保存为JSON文件
    try:
        # 将元组键转换为字符串键以便序列化
        dict_to_save = {f"{frame},{cell}": track for (frame, cell), track in gt_dict.items()}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict_to_save, f, ensure_ascii=False, indent=4)
        print(f"GT字典已保存至 {output_path}")
    except Exception as e:
        print(f"保存JSON文件失败: {e}")

if __name__ == "__main__":
    main()
