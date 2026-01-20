# import torch
# import numpy as np
# import json
# from pathlib import Path
# from typing import Dict, List, Any
# import torch.nn as nn
# import re
# split={
#
# }
#
# class AdapterDatasetGenerator:
#     def __init__(self, weight_dir: str, output_dir: str):
#         self.weight_dir = Path(weight_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.splits = {
#         "split_1_0": [4, 19, 31, 47, 51, 0],
#         "split_1_1": [4, 19, 31, 47, 51, 1],
#         "split_1_2": [4, 19, 31, 47, 51, 2],
#         "split_1_3": [4, 19, 31, 47, 51, 3],
#         "split_1_5": [4, 19, 31, 47, 51, 5],
#         "split_1_6": [4, 19, 31, 47, 51, 6],
#         "split_1_7": [4, 19, 31, 47, 51, 7],
#         "split_1_8": [4, 19, 31, 47, 51, 8],
#         "split_1_9": [4, 19, 31, 47, 51, 9],
#         "split_1_10": [4, 19, 31, 47, 51, 10],
#         "split_1_11": [4, 19, 31, 47, 51, 11],
#         "split_1_12": [4, 19, 31, 47, 51, 12],
#         "split_1_13": [4, 19, 31, 47, 51, 13],
#         "split_1_14": [4, 19, 31, 47, 51, 14],
#         "split_1_15": [4, 19, 31, 47, 51, 15],
#         "split_1_16": [4, 19, 31, 47, 51, 16],
#         "split_1_17": [4, 19, 31, 47, 51, 17],
#         "split_1_18": [4, 19, 31, 47, 51, 18],
#         "split_1_20": [4, 19, 31, 47, 51, 20],
#         "split_1_21": [4, 19, 31, 47, 51, 21],
#         "split_1_22": [4, 19, 31, 47, 51, 22],
#         "split_1_23": [4, 19, 31, 47, 51, 23],
#         "split_1_24": [4, 19, 31, 47, 51, 24],
#         "split_1_25": [4, 19, 31, 47, 51, 25],
#         "split_1_26": [4, 19, 31, 47, 51, 26],
#         "split_1_27": [4, 19, 31, 47, 51, 27],
#         "split_1_28": [4, 19, 31, 47, 51, 28],
#         "split_1_29": [4, 19, 31, 47, 51, 29],
#         "split_1_30": [4, 19, 31, 47, 51, 30],
#         "split_1_32": [4, 19, 31, 47, 51, 32],
#         "split_1_33": [4, 19, 31, 47, 51, 33],
#         "split_1_34": [4, 19, 31, 47, 51, 34],
#         "split_1_35": [4, 19, 31, 47, 51, 35],
#         "split_1_36": [4, 19, 31, 47, 51, 36],
#         "split_1_37": [4, 19, 31, 47, 51, 37],
#         "split_1_38": [4, 19, 31, 47, 51, 38],
#         "split_1_39": [4, 19, 31, 47, 51, 39],
#         "split_1_40": [4, 19, 31, 47, 51, 40],
#         "split_1_41": [4, 19, 31, 47, 51, 41],
#         "split_1_42": [4, 19, 31, 47, 51, 42],
#         "split_1_43": [4, 19, 31, 47, 51, 43],
#         "split_1_44": [4, 19, 31, 47, 51, 44],
#         "split_1_45": [4, 19, 31, 47, 51, 45],
#         "split_1_46": [4, 19, 31, 47, 51, 46],
#         "split_1_48": [4, 19, 31, 47, 51, 48],
#         "split_1_49": [4, 19, 31, 47, 51, 49],
#         "split_1_50": [4, 19, 31, 47, 51, 50],
#         "split_1_52": [4, 19, 31, 47, 51, 52],
#         "split_1_53": [4, 19, 31, 47, 51, 53],
#         "split_1_54": [4, 19, 31, 47, 51, 54],
#         "split_1_55": [4, 19, 31, 47, 51, 55],
#         "split_1_56": [4, 19, 31, 47, 51, 56],
#         "split_1_57": [4, 19, 31, 47, 51, 57],
#         "split_1_58": [4, 19, 31, 47, 51, 58],
#         "split_1_59": [4, 19, 31, 47, 51, 59],
#         "split_1_unseen5": [4, 19, 31, 47, 51],
#         "split_1_unseen10": [4, 19, 31, 47, 51, 8, 21, 25, 46, 54],
#         "split_1_unseen15": [4, 19, 31, 47, 51, 2, 6, 7, 16, 24, 26, 29, 40, 57, 58],
#         "split_1_unseen20": [4, 19, 31, 47, 51, 8, 11, 12, 15, 17, 22, 25, 27, 30, 32, 36, 37, 45, 50, 57],
#         "split_1_unseen25": [4, 19, 31, 47, 51, 0, 2, 3, 5, 6, 7, 12, 14, 20, 24, 29, 30, 33, 35, 39, 45, 49, 50, 53, 56],
#         "split_1_unseen30": [4, 19, 31, 47, 51, 0, 1, 3, 7, 8, 10, 12, 14, 16, 20, 22, 23, 24, 25, 28, 39, 41, 42, 44, 45, 48, 49, 53, 54, 58],
#         "split_1_unseen35": [4, 19, 31, 47, 51, 1, 2, 5, 6, 7, 8, 10, 11, 12, 14, 15, 17, 18, 22, 24, 27, 28, 29, 30, 35, 38, 40, 41, 42, 44, 45, 48, 53, 54, 58],
#         "split_1_unseen40": [4, 19, 31, 47, 51, 0, 3, 5, 7, 8, 11, 12, 13, 14, 15, 16, 21, 22, 25, 26, 27, 29, 30, 32, 34, 35, 36, 38, 39, 40, 42, 43, 45, 46, 52, 53, 54, 55, 56, 57],
#         "split_1_unseen45": [4, 19, 31, 47, 51, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 33, 35, 36, 38, 40, 41, 42, 43, 45, 48, 50, 53, 55, 56, 58],
#         "split_1_unseen50": [4, 19, 31, 47, 51, 0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 49, 52, 53, 54, 55, 56, 58],
#         "split_1_unseen55": [4, 19, 31, 47, 51, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 55, 57]
#     }
#     def extract_split_name_v1(self,filename):
#
#         # 找到第一个数字后面的下划线位置
#         patterns = [
#             r'(split_\d+_unseen\d+)',  # split_1_unseen35
#             r'(split_\d+_\d+)',  # split_1_5
#             r'(split_\d+)',  # split_2
#         ]
#
#         for pattern in patterns:
#             match = re.match(pattern, filename)
#             if match:
#                 return match.group(1)
#
#         # 如果没有匹配到，尝试按第一个下划线后的特定后缀分割
#         suffixes = ['_kl_', '_support_', '_lr', '_factor', '_des_']
#         for suffix in suffixes:
#             if suffix in filename:
#                 return filename.split(suffix)[0]
#
#         return filename.rsplit('.', 1)[0]  # 去掉文件扩展名
#     def load_visible_data(self,unseen):
#         all_classes = set(range(60))
#         visible_classes = sorted(list(all_classes - set(unseen)))
#         return visible_classes
#
#     def load_all_adapter_weights(self) -> Dict[str, Dict]:
#         """加载所有类别的适配器权重"""
#         adapter_data = {}
#
#         for weight_file in self.weight_dir.glob("*.pt"):
#             try:
#                 # 加载权重
#                 # weights = np.load(weight_file)
#                 weights = torch.load(weight_file)
#                 weight_dict = {k: v for k, v in weights.items()}
#                 print(weight_file.stem)
#                 unseen=self.extract_split_name_v1(weight_file.stem)
#                 if unseen in self.splits:
#                     unseen_class=self.splits[unseen]
#                     seen_class = self.load_visible_data(unseen_class)
#                 # 加载元数据
#                 meta_file = weight_file.with_suffix('.npz.meta.json')
#                 if meta_file.exists():
#                     with open(meta_file, 'r') as f:
#                         metadata = json.load(f)
#                 else:
#                     metadata = {"class_name": weight_file.stem}
#
#                 class_id = metadata.get("class_id", len(adapter_data))
#                 adapter_data[class_id] = {
#                     "weights": weight_dict['text_adapter'],
#                     # "metadata": metadata,
#                     "seen_classes":seen_class,
#                     # "class_name": metadata.get("class_name", f"class_{class_id}")
#                     # "Accuracy":weight_dict['acc']
#                 }
#
#             except Exception as e:
#                 print(f"加载 {weight_file} 失败: {e}")
#                 continue
#
#         print(f"成功加载 {len(adapter_data)} 个类别的适配器权重")
#         return adapter_data
#
#     def extract_weight_vectors(self, adapter_data: Dict) -> Dict[str, torch.Tensor]:
#         """将权重参数展平为向量"""
#         weight_vectors = {}
#
#         for class_id, data in adapter_data.items():
#             weights_dict = data["weights"]
#             weight_list = []
#
#             # 按固定顺序展平所有权重和偏置
#             for key in sorted(weights_dict.keys()):
#                 param = weights_dict[key]
#                 weight_list.append(param.flatten())
#
#             # 拼接所有参数为一个向量
#             weight_vector = torch.cat(weight_list)#768*256+768
#             weight_vectors[class_id] = {
#                 "weight_vector": weight_vector,
#                 "metadata": data["metadata"],
#                 "original_shapes": {k: v.shape for k, v in weights_dict.items()}
#             }
#
#             print(f"类别 {class_id}: 参数向量维度 {weight_vector.shape}")
#
#         return weight_vectors
#
#     def get_text_embeddings(self, class_names: List[str]) -> Dict[int, torch.Tensor]:
#         """生成文本嵌入（条件信息）"""
#         # 使用预训练的语言模型生成文本嵌入
#         self.full_language = np.load( "./data/language/ntu60_des_embeddings.npy" )# des best)
#         self.full_language = torch.Tensor(self.full_language)
#         self.full_language = self.full_language.cuda()
#         # from transformers import AutoTokenizer, AutoModel
#         #
#         # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         # text_model = AutoModel.from_pretrained("bert-base-uncased")
#         #
#         text_embeddings = {}
#
#         for class_id, class_name in class_names.items():
#             # 简单的文本提示工程
#             # prompts = [
#             #     f"a person is {class_name}",
#             #     f"the action of {class_name}",
#             #     f"someone is {class_name}",
#             #     class_name
#             # ]
#             seen_language = self.full_language[class_name]  # 128, 768
#
#             # # 对多个提示取平均
#             # embeddings = []
#             # for prompt in prompts:
#             #     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#             #     with torch.no_grad():
#             #         outputs = text_model(**inputs)
#             #     embedding = outputs.last_hidden_state.mean(dim=1)  # 池化
#             #     embeddings.append(embedding)
#             #
#             # text_embedding = torch.mean(torch.stack(embeddings), dim=0)
#             text_embeddings[class_id] = seen_language
#
#         return text_embeddings
#
#     def create_diffusion_dataset(self):
#         """创建Diffusion训练数据集"""
#         # 1. 加载适配器权重
#         adapter_data = self.load_all_adapter_weights()
#
#         # 2. 提取权重向量
#         # weight_vectors = self.extract_weight_vectors(adapter_data)
#
#         # 3. 生成文本嵌入
#         class_names = {cid: data.get("seen_classes", f"class_{cid}")
#                        for cid, data in adapter_data.items()}
#         text_embeddings = self.get_text_embeddings(class_names)
#
#         # 4. 构建数据集
#         dataset = {
#             "weights": {},
#             "text_embeddings": {},
#             # "metadata": {},
#             "seen_classes": {},
#             # "Accuracy": {},
#
#         }
#
#         for class_id in adapter_data.keys():
#             dataset["weights"][class_id] = adapter_data[class_id]["weights"]
#             dataset["text_embeddings"][class_id] = text_embeddings[class_id]
#             # dataset["metadata"][class_id] = weight_vectors[class_id]["metadata"]
#             dataset["seen_classes"][class_id] = adapter_data[class_id]["seen_classes"]
#             # dataset["Accuracy"][class_id] = adapter_data[class_id]["Accuracy"]
#             # dataset["unseen_classes"][class_id] = weight_vectors[class_id]["metadata"]["unseen_classes"]
#             # dataset["metadata"][class_id]["original_shapes"] = weight_vectors[class_id]["original_shapes"]
#
#         # 5. 保存数据集
#         output_path = self.output_dir / "diffusion_vae_dataset.npy"
#         torch.save(dataset, output_path)
#
#         # 保存统计信息
#         # stats = {
#         #     "num_classes": len(dataset["weights"]),
#         #     "weight_vector_dim": next(iter(dataset["weight_vectors"].values())).shape[0],
#         #     "text_embedding_dim": next(iter(dataset["text_embeddings"].values())).shape[0],
#         #     "class_ids": list(dataset["weight_vectors"].keys())
#         # }
#         #
#         # with open(self.output_dir / "dataset_stats.json", 'w') as f:
#         #     json.dump(stats, f, indent=2)
#
#         print(f"Diffusion训练数据集已保存到: {output_path}")
#         # print(f"数据集统计: {stats}")
#
#         return dataset
#
#
# # 使用示例
# if __name__ == "__main__":
#     generator = AdapterDatasetGenerator(
#         weight_dir="/media/zzf/ljn/wsx/PGFA/PGFA-main/output/weight/checkpoint",
#         output_dir="/media/zzf/ljn/wsx/PGFA/PGFA-main/output/diffusion_dataset_150"
#     )
#
#     dataset = generator.create_diffusion_dataset()
import torch
import re
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List

class CheckpointEnhancer:
    def __init__(self, checkpoint_dir: str, output_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 定义所有分割方案
        self.splits = {
            "split_1": [4, 19, 31, 47, 51],
            "split_1_0": [4, 19, 31, 47, 51, 0],
            "split_1_1": [4, 19, 31, 47, 51, 1],
            "split_1_2": [4, 19, 31, 47, 51, 2],
            "split_1_3": [4, 19, 31, 47, 51, 3],
            "split_1_5": [4, 19, 31, 47, 51, 5],
            "split_1_6": [4, 19, 31, 47, 51, 6],
            "split_1_7": [4, 19, 31, 47, 51, 7],
            "split_1_8": [4, 19, 31, 47, 51, 8],
            "split_1_9": [4, 19, 31, 47, 51, 9],
            "split_1_10": [4, 19, 31, 47, 51, 10],
            "split_1_11": [4, 19, 31, 47, 51, 11],
            "split_1_12": [4, 19, 31, 47, 51, 12],
            "split_1_13": [4, 19, 31, 47, 51, 13],
            "split_1_14": [4, 19, 31, 47, 51, 14],
            "split_1_15": [4, 19, 31, 47, 51, 15],
            "split_1_16": [4, 19, 31, 47, 51, 16],
            "split_1_17": [4, 19, 31, 47, 51, 17],
            "split_1_18": [4, 19, 31, 47, 51, 18],
            "split_1_20": [4, 19, 31, 47, 51, 20],
            "split_1_21": [4, 19, 31, 47, 51, 21],
            "split_1_22": [4, 19, 31, 47, 51, 22],
            "split_1_23": [4, 19, 31, 47, 51, 23],
            "split_1_24": [4, 19, 31, 47, 51, 24],
            "split_1_25": [4, 19, 31, 47, 51, 25],
            "split_1_26": [4, 19, 31, 47, 51, 26],
            "split_1_27": [4, 19, 31, 47, 51, 27],
            "split_1_28": [4, 19, 31, 47, 51, 28],
            "split_1_29": [4, 19, 31, 47, 51, 29],
            "split_1_30": [4, 19, 31, 47, 51, 30],
            "split_1_32": [4, 19, 31, 47, 51, 32],
            "split_1_33": [4, 19, 31, 47, 51, 33],
            "split_1_34": [4, 19, 31, 47, 51, 34],
            "split_1_35": [4, 19, 31, 47, 51, 35],
            "split_1_36": [4, 19, 31, 47, 51, 36],
            "split_1_37": [4, 19, 31, 47, 51, 37],
            "split_1_38": [4, 19, 31, 47, 51, 38],
            "split_1_39": [4, 19, 31, 47, 51, 39],
            "split_1_40": [4, 19, 31, 47, 51, 40],
            "split_1_41": [4, 19, 31, 47, 51, 41],
            "split_1_42": [4, 19, 31, 47, 51, 42],
            "split_1_43": [4, 19, 31, 47, 51, 43],
            "split_1_44": [4, 19, 31, 47, 51, 44],
            "split_1_45": [4, 19, 31, 47, 51, 45],
            "split_1_46": [4, 19, 31, 47, 51, 46],
            "split_1_48": [4, 19, 31, 47, 51, 48],
            "split_1_49": [4, 19, 31, 47, 51, 49],
            "split_1_50": [4, 19, 31, 47, 51, 50],
            "split_1_52": [4, 19, 31, 47, 51, 52],
            "split_1_53": [4, 19, 31, 47, 51, 53],
            "split_1_54": [4, 19, 31, 47, 51, 54],
            "split_1_55": [4, 19, 31, 47, 51, 55],
            "split_1_56": [4, 19, 31, 47, 51, 56],
            "split_1_57": [4, 19, 31, 47, 51, 57],
            "split_1_58": [4, 19, 31, 47, 51, 58],
            "split_1_59": [4, 19, 31, 47, 51, 59],
            "split_12": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59],
            "split_12_0": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 0],
            "split_12_1": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 1],
            "split_12_2": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 2],
            "split_12_4": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 4],
            "split_12_5": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 5],
            "split_12_6": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 6],
            "split_12_8": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 8],
            "split_12_9": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 9],
            "split_12_10": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 10],
            "split_12_11": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 11],
            "split_12_12": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 12],
            "split_12_14": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 14],
            "split_12_15": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 15],
            "split_12_16": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 16],
            "split_12_17": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 17],
            "split_12_18": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 18],
            "split_12_20": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 20],
            "split_12_21": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 21],
            "split_12_23": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 23],
            "split_12_24": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 24],
            "split_12_25": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 25],
            "split_12_26": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 26],
            "split_12_27": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 27],
            "split_12_29": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 29],
            "split_12_30": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 30],
            "split_12_31": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 31],
            "split_12_32": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 32],
            "split_12_33": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 33],
            "split_12_35": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 35],
            "split_12_36": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 36],
            "split_12_37": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 37],
            "split_12_38": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 38],
            "split_12_39": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 39],
            "split_12_40": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 40],
            "split_12_42": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 42],
            "split_12_43": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 43],
            "split_12_44": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 44],
            "split_12_45": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 45],
            "split_12_46": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 46],
            "split_12_48": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 48],
            "split_12_49": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 49],
            "split_12_50": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 50],
            "split_12_51": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 51],
            "split_12_53": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 53],
            "split_12_54": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 54],
            "split_12_55": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 55],
            "split_12_57": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 57],
            "split_12_58": [3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59, 58]}

    def extract_split_name(self,filename):

        # 找到第一个数字后面的下划线位置
        patterns = [
            r'(split_\d+_unseen\d+)',  # split_1_unseen35
            r'(split_\d+_\d+)',  # split_1_5
            r'(split_\d+)',  # split_2
        ]

        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                return match.group(1)

        # 如果没有匹配到，尝试按第一个下划线后的特定后缀分割
        suffixes = ['_kl_', '_support_', '_lr', '_factor', '_des_']
        for suffix in suffixes:
            if suffix in filename:
                return filename.split(suffix)[0]

        return filename.rsplit('.', 1)[0]  # 去掉文件扩展名



    def get_visible_classes(self, unseen_classes: List[int]) -> List[int]:
        """根据不可见类别计算可见类别"""
        all_classes = set(range(60))
        visible_classes = sorted(list(all_classes - set(unseen_classes)))
        return visible_classes

    def enhance_checkpoint(self, checkpoint_path: Path) -> bool:
        """增强单个checkpoint文件"""
        try:
            # 加载原始checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"处理文件: {checkpoint_path.name}")

            # 提取分割方案名称
            split_name = self.extract_split_name(checkpoint_path.name)
            print(f"  提取的分割方案: {split_name}")

            # 检查是否在预定义的分割方案中
            if split_name not in self.splits:
                print(f"  警告: 分割方案 {split_name} 不在预定义列表中")
                return False

            # 获取不可见类别和可见类别
            unseen_classes = self.splits[split_name]
            visible_classes = self.get_visible_classes(unseen_classes)

            print(f"  不可见类别: {unseen_classes}")
            print(f"  可见类别数: {len(visible_classes)}")
            print(f"  不可见类别数: {len(unseen_classes)}")

            # 创建增强的checkpoint
            enhanced_checkpoint = OrderedDict()

            # 1. 添加原始模型权重
            if isinstance(checkpoint, OrderedDict):
                # 如果已经是OrderedDict，直接复制
                for key, value in checkpoint.items():
                    enhanced_checkpoint[key] = value
            else:
                # 如果是其他格式，尝试提取模型权重
                enhanced_checkpoint['text_adapter'] = checkpoint['text_adapter']

            # 2. 添加分割信息
            enhanced_checkpoint['seen_class'] = visible_classes
            #     # ('seen_class', split_name),
            #     # ('unseen_classes', unseen_classes),
            #     ('seen_classes', visible_classes),
            #     # ('num_unseen', len(unseen_classes)),
            #     # ('num_visible', len(visible_classes))
            # ])

            # # 3. 添加文件元数据
            # enhanced_checkpoint['metadata'] = OrderedDict([
            #     ('original_file', checkpoint_path.name),
            #     ('enhanced_time', '2024-01-01 10:00:00'),  # 可以改为实际时间
            #     ('total_classes', 60)
            # ])
            #
            # 保存增强的checkpoint
            output_path = self.output_dir / checkpoint_path.name
            torch.save(enhanced_checkpoint, output_path)
            print(f"  成功保存增强文件: {output_path}")

            return True

        except Exception as e:
            print(f"  处理文件 {checkpoint_path.name} 时出错: {e}")
            return False

    def batch_enhance_checkpoints(self):
        """批量增强所有checkpoint文件"""
        print("开始批量增强checkpoint文件...")
        print("=" * 60)

        success_count = 0
        total_count = 0

        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            total_count += 1
            if self.enhance_checkpoint(checkpoint_file):
                success_count += 1
            print("-" * 40)

        print("=" * 60)
        print(f"处理完成! 成功: {success_count}/{total_count} 个文件")

        return success_count, total_count

# 使用示例
if __name__ == "__main__":
    checkpoint_dir = "/media/zzf/ljn/wsx/output/weight_split_12/split_12"  # 原始checkpoint目录
    output_dir = "/media/zzf/ljn/wsx/output/weight_split_12/enhanced_checkpoints_all"  # 增强后文件输出目录
    enhancer = CheckpointEnhancer(checkpoint_dir, output_dir)
    success, total = enhancer.batch_enhance_checkpoints()