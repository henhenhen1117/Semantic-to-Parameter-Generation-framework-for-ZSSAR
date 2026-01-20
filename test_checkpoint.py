# import torch
# import numpy as np
# import json
# import os
# from pathlib import Path
# from typing import Dict, Any
#
#
# def save_adapter_npz(state_dict: Dict[str, torch.Tensor], out_path: Path, meta: Dict[str, Any]) -> None:
#     """
#     保存适配器权重为npz格式
#
#     Args:
#         state_dict: 适配器状态字典
#         out_path: 输出文件路径
#         meta: 元数据信息
#     """
#     try:
#         np_state = {}
#         total_params = 0
#
#         for k, v in state_dict['adapter'].items():
#             if isinstance(v, torch.Tensor):
#                 np_state[k] = v.cpu().numpy()
#                 total_params += v.numel()
#             else:
#                 print(f"警告: 跳过非张量参数 {k} (类型: {type(v)})")
#
#         # 保存权重文件
#         np.savez(out_path, **np_state)
#
#         # 保存元数据文件
#         meta_path = out_path.with_suffix('.npz.meta.json')
#         with open(meta_path, 'w', encoding='utf-8') as f:
#             json.dump(meta, f, indent=2, ensure_ascii=False)
#
#         print(f"✅ 成功保存适配器权重")
#         print(f"�� 权重文件: {out_path}")
#         print(f"�� 元数据文件: {meta_path}")
#         print(f"�� 参数统计: {len(np_state)} 个参数张量, 总共 {total_params} 个参数")
#
#     except Exception as e:
#         print(f"❌ 保存适配器权重失败: {e}")
#         raise
#
#
# def main():
#     """主函数"""
#     # 配置路径
#     ckpt_path = "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/weight/split_1_5_kl_des_support_factor0.9_lr0.05.pt"
#     out_dir = Path("/media/zzf/ljn/wsx/PGFA/PGFA-main/output/dataset/")
#
#     # 创建输出目录
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     try:
#         # 加载检查点
#         print("�� 加载检查点...")
#         ckpt = torch.load(ckpt_path, map_location='cpu')
#         print(f"✅ 检查点加载成功，包含键值: {list(ckpt.keys())}")
#
#         # 提取适配器权重
#         print("�� 提取适配器权重...")
#         state_dict = ckpt.get('state_dict', ckpt)
#         adapter_state = {k: v for k, v in state_dict.items() if 'adapter' in k}
#
#         if not adapter_state:
#             print("❌ 未找到适配器权重")
#             return
#
#         print(f"✅ 找到 {len(adapter_state)} 个适配器参数")
#         for k, v in adapter_state['adapter'].items():
#             print(f"   {k}: {v.shape}")
#
#         # 创建详细的元数据
#         seen_classes = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59]
#         meta = {
#             "sample_id": "adapter_0002",
#             "dataset": "NTU-RGB+D 60",
#             "split": "split_1",
#             "seen_classes": seen_classes,
#             "num_seen_classes": len(seen_classes),
#             "unseen_classes":[4, 19, 31, 47, 51],
#             "num_unseen_classes": len("unseen_classes"),
#             "adapter_type": "skeleton_adapter",
#             "parameters": {
#                 "shapes": {k: list(v.shape) for k, v in adapter_state['adapter'].items()},
#
#             },
#         }
#
#         # 设置输出路径
#         out_path = out_dir / "adapter_split_1_seen_55_classes.npz"
#
#         # 保存适配器权重
#         save_adapter_npz(adapter_state, out_path, meta)
#
#         print("�� 适配器权重导出完成!")
#
#     except FileNotFoundError:
#         print(f"❌ 检查点文件不存在: {ckpt_path}")
#     except Exception as e:
#         print(f"❌ 处理过程中发生错误: {e}")
#
#
# if __name__ == "__main__":
#     main()
# # build_flat_dataset.py
# # import numpy as np, torch, json
# # from pathlib import Path
# # import os
# #
# # def build_flat_dataset(samples_dir, out_path, manifest_out=None):
# #     samples_dir = Path(samples_dir)
# #     files = sorted(samples_dir.glob('*.npz'))
# #     dataset = {'weight_vectors': {}, 'metadata': {}, 'file_map': {}}
# #     for f in files:
# #         name = f.stem  # sample_id
# #         arr = np.load(str(f), allow_pickle=True)
# #         # 保证顺序一致：按 sorted(arr.files)
# #         parts = []
# #         for k in sorted(arr.files):
# #             parts.append(arr[k].ravel())
# #         flat = np.concatenate(parts).astype(np.float32)
# #         # load meta
# #         meta_file = f.with_suffix('.meta.json')
# #         meta = {}
# #         if meta_file.exists():
# #             meta = json.load(open(meta_file))
# #         dataset['weight_vectors'][name] = torch.from_numpy(flat)
# #         dataset['metadata'][name] = meta
# #         dataset['file_map'][name] = f.name
# #     # 保存为 torch 文件（更快加载）
# #     out_path = Path(out_path)
# #     torch.save(dataset, str(out_path))
# #     print("Saved flat dataset to", out_path)
# #     if manifest_out:
# #         import csv
# #         with open(manifest_out, 'w', newline='') as csvf:
# #             writer = csv.writer(csvf)
# #             writer.writerow(['sample_id','file','class_id','class_name','dim'])
# #             for sid in dataset['weight_vectors'].keys():
# #                 meta = dataset['metadata'][sid]
# #                 writer.writerow([sid, dataset['file_map'][sid], meta.get('class_id',''), meta.get('class_name',''), dataset['weight_vectors'][sid].numel()])
# #
# # # usage
# # # build_flat_dataset('/path/to/output/samples', '/path/to/output/diffusion_flat_dataset.pt', manifest_out='/path/to/output/manifest.csv')
import os

import torch
# root="/media/zzf/ljn/wsx/PGFA/PGFA-main/checkpoint"
# checkpoint_list=os.listdir(root)
# checkpoint_list = list([os.path.join(root, item) for item in checkpoint_list])
# structures = [{} for _ in range(len(checkpoint_list))]
# for i, checkpoint in enumerate(checkpoint_list):
#     chpt=torch.load(checkpoint)
#     print((chpt))
from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import snapshot_download
import os
# from huggingface_hub import snapshot_download
#
# local_dir = snapshot_download(
#     repo_id="sentence-transformers/stsb-bert-large",
#     local_dir="./stsb-bert-large",
#     local_dir_use_symlinks=False
# )
# print("Saved to:", local_dir)
import os
print("HTTP_PROXY =", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY =", os.environ.get("HTTPS_PROXY"))