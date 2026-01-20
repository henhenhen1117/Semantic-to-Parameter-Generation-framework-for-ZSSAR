import glob
import json

import os
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import requests
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import openai
from config import *
from dataset import DataSet
from logger import Log
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

from module.gcn.st_gcn import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2,get_acc_dynamic
from sentence_transformers import SentenceTransformer
from module.prompt_learner import encode_multiview_keep_dim
from module.LinearMappin import LinearInMapper as Mapper


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
# openai.api_key = "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm"

# %%
class Processor:
    def load_txt(self,file_path):
        actions = []

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            # 解析格式: "动作名称" 描述
            if '"' in line:
                # 分割动作名称和描述
                parts = line.split('"')
                if len(parts) >= 3:
                    action_name = parts[1].strip()
                    # 提取描述部分（去掉开头的 "is" 或 "refers to" 等）
                    description = parts[2].strip()
                    # 去掉开头的动词短语
                    if description.startswith('is '):
                        description = description[3:]
                    elif description.startswith('refers to '):
                        description = description[10:]
                    elif description.startswith('involves '):
                        description = description[9:]

                    actions.append({
                        'name': action_name,
                        'description': description,
                        'full_text': line
                    })

        return actions
    def load_database(self):
        # def parse_lines_without_quotes(text):
        prompt_database = {}
        # 确保文件路径正确
        with open('optimized_prompts.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                # 如果是带引号的格式: "Drink water" refers to...
                match = re.match(r'^"([^"]+)"\s*(.*)', line)

                if match:
                    action_name = match.group(1)
                    full_text = line  # 或者 match.group(2) 如果只想要描述
                    prompt_database[action_name] = full_text
                else:
                    # 兼容无引号的临时处理（非常不推荐，容易切错）
                    # 尝试用常见连接词分割
                    parts = re.split(r'\s+(is|refers to|involves)\s+', line, 1)
                    if len(parts) >= 3:
                        action_name = parts[0]
                        prompt_database[action_name] = line

        return prompt_database
    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        self.test_aug_acc = -1
        self.best_aug_acc = -1
        self.best_aug_epoch = -1
        self.class_description=self.load_txt("/media/zzf/ljn/wsx/PGFA/PGFA-main/data/ntu60_des.txt")

        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()
        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)
        self.prompt_database=self.load_database()
        self.embeddings, self.texts = self.txt_to_sbert_embeddings(
            txt_file="optimized_prompts.txt",
            output_npy="prompt_embeddings.npy"
        )
        self.embeddings=torch.tensor(self.embeddings)
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict['encoder'])

    def load_text_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        # model.load_state_dict(pretrained_dict['text_adapter'])
        model.load_state_dict(pretrained_dict)
        return pretrained_dict

    def load_transfer_weight(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        # model.load_state_dict(pretrained_dict,strict=False)
        model.load_state_dict(pretrained_dict['mappin'])

    def load_ske_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict['ske_adapter'])

    def load_bias_weights(self, model=None, pretrained_dict=None):
        pretrained_dict = pretrained_dict
        model.load_state_dict(pretrained_dict,strict=False)

    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=15,
                             loss_mode='step', step=[50, 80]):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max - lr_min) * (
                        1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # if i == 0:
            #     param_group['lr'] = lr * 0.1
            # else:
            #     param_group['lr'] = lr

    def layernorm(self, feature):

        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature - mean) / torch.sqrt(var)

        return out

    def get_soft_targets(self, label, fixed_text_features, temperature=0.1):
        """
        根据当前 batch 的 label，从固定的文本特征中计算语义相似度作为 Soft Target
        fixed_text_features: [Total_Classes, Dim] 原始的、冻结的 SBERT 特征
        """
        # 取出当前 batch 对应的文本特征
        batch_text_feats = fixed_text_features[label]  # [B, Dim]

        # 计算 batch 内部的相似度矩阵 [B, B]
        # 归一化
        batch_text_feats = F.normalize(batch_text_feats, dim=-1)
        sim_matrix = batch_text_feats @ batch_text_feats.t()

        # 将相似度矩阵转化为概率分布 (Soft Labels)
        # 你的 KLLoss 期望输入是 log_probs，target 是 probs
        soft_targets = F.softmax(sim_matrix / temperature, dim=-1)

        return soft_targets
    @ex.capture
    def load_model(self, in_channels, hidden_channels, hidden_dim,
                   dropout, graph_args, edge_importance_weighting, visual_size, language_size, weight_path, loss_type,
                   fix_encoder, finetune):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                             hidden_dim=hidden_dim, dropout=dropout,
                             graph_args=graph_args,
                             edge_importance_weighting=edge_importance_weighting,
                             )
        self.encoder = self.encoder.cuda()
        # self.adapter = Linear().cuda()
        self.ske_adapter = Linear(256, 512).cuda()
        self.text_adapter = Linear(768, 512).cuda()
        # self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        # self.sbert = SentenceTransformer('sentence-transformers/stsb-bert-large')
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        self.bert = self.sbert[0].auto_model  # HuggingFace 的 BERT/Roberta 模型
        self.tokenizer = self.sbert[0].tokenizer  # 对应的 tokenizer
        self.hidden_size = self.bert.config.hidden_size  # ctx 维度，例如 1024
        self.mapper = Mapper(768, 256, rank=512, map_bias=True).cuda()


        # self.embedding=encode_multiview_keep_dim(self.tokenizer,self.bert)
        # self.embeddings=self.embeddings.cpu().numpy()
        # np.save('mean_description.npy', self.embeddings)
        # print("Saved mean_description.npy")
        if loss_type == "kl" or loss_type == "klv2" or loss_type == "kl+cosface" or loss_type == "kl+sphereface" or "kl+margin":
            self.loss = KLLoss().cuda()
        elif loss_type == "mse":
            self.loss = nn.MSELoss().cuda()
        elif loss_type == "kl+mse":
            self.loss_kl = KLLoss().cuda()
            self.loss_mse = nn.MSELoss().cuda()
        elif loss_type == "kl+kd":
            self.loss = KLLoss().cuda()
            self.kd_loss = KDLoss().cuda()
        else:
            raise Exception('loss_type Error!')
        self.logit_scale = self.ske_adapter.get_logit_scale()
        # self.logit_scale_v2 = self.adapter.get_logit_scale_v2()

        # if fix_encoder or finetune:
        self.load_weights(self.encoder, weight_path)
        self.weight=self.load_text_weights(self.text_adapter, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/save_param/generated_model.pth")
        self.load_transfer_weight(self.mapper, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/mapper/mapper_best_mappin2.pt")
        self.ske_bias = self.mapper(self.weight['adapter.weight'], self.weight['adapter.bias'].cuda())
        self.load_bias_weights(self.ske_adapter, self.ske_bias)
        # self.load_ske_weights(self.ske_adapter,weight_path)
        # self.load_text_weights(self.text_adapter,weight_path)

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.SGD([
            # {'params': self.encoder.parameters()},
            # {'params': self.prompt_learner.parameters()},
            {'params': self.ske_adapter.parameters()},
            {'params': self.text_adapter.parameters()},
        ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=False
        )
        # self.optimizer = torch.optim.AdamW([
            # {'params': self.encoder.parameters()},
            # {'params': self.prompt_learner.ctx},
            # {'params': self.ske_adapter.parameters()},
            # {'params': self.text_adapter.parameters()},
        # ],
        #     lr=0.0025,
        #     weight_decay=0.00001
        # )

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num, DA):  # print -> log.info
        self.log.info("main track")
        for epoch in range(epoch_num):
            # self.train_epoch(epoch)
            # # self.test_all_checkpoints(epoch)
            # with torch.no_grad():
            #     self.test_epoch(epoch=epoch)
            self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] train loss: {}".format(epoch, self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch, self.best_acc))
            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch, self.test_aug_acc))
                self.log.info("epoch [{}] gets the best DA acc: {}".format(self.best_aug_epoch, self.best_aug_acc))
            # if epoch > 5:
            #     self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            #     self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            # else:
            #     self.log.info("epoch [{}] : warm up epoch.".format(epoch))

    def txt_to_sbert_embeddings(self,txt_file, output_npy, model_name='all-mpnet-base-v2'):
        """
        将TXT文件中的文本通过Sentence-BERT编码为向量，并保存为npy文件

        参数:
            txt_file: 输入TXT文件路径
            output_npy: 输出npy文件路径
            model_name: Sentence-BERT模型名称
        """
        # 1. 读取TXT文件内容
        print(f"读取文件: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 过滤掉注释行和空行
        texts = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                texts.append(line)

        print(f" 找到 {len(texts)} 条有效文本")

        # 2. 加载Sentence-BERT模型
        print(f" 加载模型: {model_name}")
        model = SentenceTransformer(model_name)

        # 3. 编码文本为向量
        print(" 编码文本...")
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,  # 转换为PyTorch张量
            show_progress_bar=True,  # 显示进度条
            normalize_embeddings=True  # 归一化向量
        )

        # 4. 转换为numpy数组
        embeddings_np = embeddings.cpu().numpy()
        print(f" 向量维度: {embeddings_np.shape}")
        print(f" 第一个向量样本: {embeddings_np[0][:5]}...")  # 显示前5个值

        # 5. 保存为npy文件
        np.save(output_npy, embeddings_np)
        print(f"�� 向量已保存到: {output_npy}")

        # 6. 同时保存元数据（可选）
        meta_file = output_npy.replace('.npy', '_meta.txt')
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(f"# 文本向量化元数据\n")
            f.write(f"源文件: {txt_file}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"文本数量: {len(texts)}\n")
            f.write(f"向量维度: {embeddings_np.shape[1]}\n")
            f.write(f"生成时间: {np.datetime64('now')}\n\n")
            f.write("# 文本列表:\n")
            for i, text in enumerate(texts):
                f.write(f"{i}: {text}\n")

        print(f"数据已保存到: {meta_file}")

        return embeddings_np, texts
    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type, alpha, beta, m, fix_encoder):
        self.encoder.train()  # eval -> train
        if fix_encoder:
            self.encoder.eval()
        self.ske_adapter.train()
        self.text_adapter.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5,
                                  loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']
        for data, label in tqdm(loader):  # data shape:128,3,50,25,2;N,C,T,V,M
            data = data.type(torch.FloatTensor).cuda()
            # print(data.shape) #128,3,50,25,2
            # label = label.type(torch.LongTensor).cuda()
            label_g = gen_label(label)  # 128,128
            label = label.type(torch.LongTensor).cuda()

            # print(label.shape) # 128
            # print(label) # int
            # seen_language=self.class_description[label]
            # self.learnable_context=self.prompt_learner()
            # seen_language=self.learnable_context[label]
            seen_language = self.embeddings[label].cuda()  # 128, 768
            # print(seen_language.shape)

            feat = self.encoder(data)  # stgcn: skeleton feature extraction 128,256
            if fix_encoder:
                feat = feat.detach()
            skleton_feat = self.ske_adapter(feat)  # adapter project 256,768   shape:128,768
            seen_language=self.text_adapter(seen_language)
            # S0 = self.cosine_sim_matrix(self.full_language[label])  # 原始语言相似度
            # S1 = self.cosine_sim_matrix(self.learnable_context[label])  # context 后的相似度
            if loss_type == "kl":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            elif loss_type == "kl+margin":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                ones = torch.ones_like(ground_truth).cuda()
                ones -= m
                logits_per_skl = torch.where(ones < logits_per_skl, ones, logits_per_skl) * ground_truth + (
                            1 - ground_truth) * logits_per_skl
                logits_per_text = torch.where(ones < logits_per_text, ones, logits_per_text) * ground_truth + (
                            1 - ground_truth) * logits_per_text
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2

            elif loss_type == "kl+cosface":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                logits_per_skl -= ground_truth * m
                logits_per_text -= ground_truth * m
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            elif loss_type == "kl+sphereface":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                logits_per_skl = get_m_theta(logits_per_skl, m) * ground_truth + (1 - ground_truth) * logits_per_skl
                logits_per_text = get_m_theta(logits_per_text, m) * ground_truth + (1 - ground_truth) * logits_per_text
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            elif loss_type == "klv2":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                logit_skl_skl, logit_skl_skl_2 = create_logits(skleton_feat, skleton_feat, self.logit_scale, exp=True)
                loss_skl_skl = self.loss(logit_skl_skl, ground_truth)
                loss = alpha * (loss_skls + loss_texts) / 2 + beta * loss_skl_skl
                # loss = (loss_skls + loss_texts + loss_skl_skl)/3

            elif loss_type == "mse":
                skl_skl_sim, skl_text_sim, text_text_sim = create_sim_matrix(skleton_feat, seen_language)
                loss = self.loss(skl_text_sim, text_text_sim) * skl_text_sim.shape[0]
            elif loss_type == "kl+mse":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                loss_skls = self.loss_kl(logits_per_skl, ground_truth)
                loss_texts = self.loss_kl(logits_per_text, ground_truth)
                loss_kl = (loss_skls + loss_texts) / 2
                skl_skl_sim, skl_text_sim, text_text_sim = create_sim_matrix(skleton_feat, seen_language)
                loss_mse = self.loss_mse(skl_text_sim, text_text_sim)  # * skl_text_sim.shape[0]
                # loss_mse += self.loss_mse(skl_skl_sim, text_text_sim) #* skl_text_sim.shape[0]
                loss = alpha * loss_kl + beta * loss_mse
            elif loss_type == "kl+kd":
                margin = 0.5
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()  # one-hot
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2

                logits_per_skl_v2, logits_per_text_v2 = create_logits(skleton_feat, seen_language, logit_scale=1,
                                                                      exp=False)
                # logits_per_skl_v2, logits_per_text_v2 = logits_per_skl, logits_per_text
                ground_truth_v2 = gen_label_from_text_sim(seen_language) + (ground_truth - 1) * margin  # teacher logit
                loss_skls_v2 = self.kd_loss(logits_per_skl_v2, ground_truth_v2)
                loss_texts_v2 = self.kd_loss(logits_per_text_v2, ground_truth_v2)
                kd_loss = (loss_skls_v2 + loss_texts_v2) / 2
                loss = alpha * loss + beta * kd_loss
            elif loss_type == "optimized_kl":
                # A. 计算 Logits (Vision-Text & Text-Vision)
                # create_logits 内部做了 normalize 和 scale
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale,
                                                                exp=True)

                # B. 生成 Soft Targets (关键优化)
                # 使用 self.full_language (原始SBERT特征) 作为语义锚点
                # 这样即使 "Prompt Learner" 在变，Target 也是稳定的语义关系
                with torch.no_grad():
                    soft_targets = self.get_soft_targets(label, self.full_language, temperature=0.07)

                # C. 计算对比损失 (使用 Soft Labels)
                # 注意：KLLoss 通常实现为 nn.KLDivLoss(reduction='batchmean')
                # 输入需为 log_softmax，target 为 softmax (probability)

                # 对 logits 取 log_softmax
                log_probs_skl = F.log_softmax(logits_per_skl, dim=-1)
                log_probs_text = F.log_softmax(logits_per_text, dim=-1)

                loss_v2t = F.kl_div(log_probs_skl, soft_targets, reduction='batchmean')
                loss_t2v = F.kl_div(log_probs_text, soft_targets, reduction='batchmean')
                loss_contrastive = (loss_v2t + loss_t2v) / 2

                # D. Prompt 漂移正则化 (可选，防止 Prompt 学歪)
                # 限制 learnable context 后的特征不应偏离原始特征太远
                # original_text_feats = self.text_adapter(self.full_language[label])
                # loss_distill = F.mse_loss(seen_language, original_text_feats)

                loss = loss_contrastive
            else:
                raise Exception('loss_type Error!')

            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    def cosine_sim_matrix(self,x):
        x = F.normalize(x, dim=-1)
        return x @ x.t()  # [C, C]



    def find_top_confusions(self,all_preds, all_labels, class_names_list, top_k=3):
        """
        all_preds: 预测的类别ID列表
        all_labels: 真实的类别ID列表
        class_names_list: 类别ID到名称的映射列表，例如 ['drink', 'jump', ...]
        """

        # 1. 找出所有预测错误的索引
        incorrect_indices = np.where(all_preds != all_labels)[0]

        if len(incorrect_indices) == 0:
            return []

        # 2. 统计具体的错误对 (True Class, Predicted Class)
        error_pairs = []
        for idx in incorrect_indices:
            true_cls = all_labels[idx]
            pred_cls = all_preds[idx]
            error_pairs.append((true_cls, pred_cls))

        # 3. 统计每种错误发生的频率
        # 格式: {(True_ID, Pred_ID): Count}
        error_counts = {}
        for pair in error_pairs:
            error_counts[pair] = error_counts.get(pair, 0) + 1

        # 4. 排序，找到发生次数最多的错误
        # 按错误次数从高到低排序
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        # 5. 生成可读的报告
        report = []
        for (true_id, pred_id), count in sorted_errors[:top_k]:
            true_name = class_names_list[true_id]
            pred_name = class_names_list[pred_id]

            report.append({
                "target_action": true_name,  # 应该是什么
                "confused_with": pred_name,  # 被错认成什么
                "error_count": count  # 错了多少次
            })

        return report

    def generate_optimization_instruction(self,error_report):
        """
        根据错误报告生成给 LLM 的指令
        """
        instructions = []

        for item in error_report:
            target = item['target_action']
            confused = item['confused_with']

            # 构造给 LLM 的提示词
            prompt = f"""
            Current Status: The model frequently confuses the action '{target}' with '{confused}'.
            Task: Please rewrite the description for '{target}'.
            Requirement: Emphasize the kinematic differences between '{target}' and '{confused}'.
            For example, focus on the specific body parts involved, the amplitude of motion, or the temporal sequence.
            """
            instructions.append(prompt)

        return instructions

    # prompt_database =np.loadtxt('optimized_prompts.txt')
#     prompt_database = {
#     "drink water": "Drink water refers to the act of taking in water by mouth to hydrate or quench thirst as a human action.",
#     "eat meal/snack": "Eat meal/snack is consuming food for sustenance, nourishment, or pleasure.",
#     "brush teeth": "Brushing teeth is the act of cleaning teeth and gums using a toothbrush and toothpaste for oral hygiene.",
#     "brush hair": "Brushing hair involves using a brush or comb to groom and style hair.",
#     "drop": "Drop refers to accidentally or intentionally releasing an object from one's grasp, allowing it to fall to the ground or another surface.",
#     "pickup": "Pickup involves lifting or grasping an object with one's hands.",
#     "throw": "Throw is the act of projecting an object through the air by using the hand or arm.",
#     "sit down": "Sitting down involves moving from a standing position to a seated position.",
#     "stand up (from sitting position)": "Standing up (from sitting position) involves transitioning from a seated position to an upright position by pushing up with the legs.",
#     "clap": "Clapping involves striking one's palms together to make a sound, typically as a form of applause or approval.",
#     "read": "Reading involves interpreting and understanding written or printed words and symbols.",
#     "write": "Writing involves using a pen, pencil, or other writing tool to create letters, words, or symbols on a surface such as paper or a screen.",
#     "tear up paper": "Tear up paper involves using one's hands to rip apart paper into smaller pieces.",
#     "wear jacket": "Wear jacket is the act of putting on a garment designed to cover the upper body and arms.",
#     "take off jacket": "Take off jacket involves removing a jacket or outer garment from one's body.",
#     "wear a shoe": "Wear a shoe involves putting a shoe onto one's foot for protection or fashion purposes.",
#     "take off a shoe": "Take off a shoe involves removing a shoe from one's foot using one's hands.",
#     "wear on glasses": "Wear on glasses involves putting eyeglasses or spectacles on one's face to correct vision or protect the eyes.",
#     "take off glasses": "Take off glasses involves removing eyeglasses from one's face with one's hands.",
#     "put on a hat/cap": "Put on a hat/cap involves placing headwear over one's head, typically for protection from the sun.",
#     "take off a hat/cap": "Take off a hat/cap involves removing a piece of headwear from one's head, typically by lifting or pulling it off.",
#     "cheer up": "Cheer up involves trying to make oneself or someone else feel more positive or happy.",
#     "hand wave": "Hand waving is a gesture involving moving one's hand or hands to signal, greet, or draw attention.",
#     "kick something": "Kicking something is a human action that involves striking an object with the foot.",
#     "reach into pocket": "Reach into pocket is the act of extending one's hand into a pocket to retrieve an item or to adjust the pocket contents.",
#     "hop (one foot jumping)": "Hopping (one foot jumping) is a human action that involves jumping repeatedly on one foot.",
#     "jump up": "Jumping up involves leaping upward from a standing position as part of physical training.",
#     "make a phone call/answer phone": "Make a phone call/Answer phone involves using a phone to either initiate or receive a phone conversation.",
#     "play with phone/tablet": "Playing with phone/tablet involves manipulating electronic devices through various interactions such as tapping, swiping, or scrolling.",
#     "type on a keyboard": "Typing on a keyboard is the act of pressing keys on a keyboard to input information or commands into a computer or other electronic device.",
#     "point to something with finger": "Pointing to something with finger involves extending the arm and using the finger to indicate or draw attention to a specific object or location.",
#     "take a selfie": "Taking a selfie is using a camera to take a self-portrait photograph, typically for sharing on social media or personal keepsake.",
#     "check time (from watch)": "Checking time (from watch) involves looking at a wristwatch or other timekeeping device to determine the current time.",
#     "rub two hands together": "Rubbing two hands together involves moving one's hands back and forth against each other, often for warmth.",
#     "nod head/bow": "Nod headbow involves lowering one's head briefly in a sign of respect, greeting, or acknowledgment.",
#     "shake head": "Shake head is moving the head from side to side in a rapid or deliberate manner, often to indicate disagreement.",
#     "wipe face": "Wiping face involves using a cloth or one's hands to remove dirt, sweat, or moisture from one's face.",
#     "salute": "Salute is a gesture of respect or greeting, typically performed by raising one's hand to the forehead or brim of a hat.",
#     "put the palms together": "Putting the palms together is the act of pressing one's hands together, often as a sign of respect, greeting, or prayer.",
#     "cross hands in front (saying stop)": "Crossing hands in front (saying stop) is a gesture where one crosses their arms in front of their body and says 'stop'.",
#     "sneeze/cough": "Sneeze/cough is the involuntary or deliberate act of expelling air and sometimes mucus from the nose and mouth, typically due to illness or irritation.",
#     "stagger": "Staggering is the unsteady movement or swaying of the body, usually caused by intoxication, dizziness, or fatigue.",
#     "fall": "Falling is the sudden loss of balance resulting in a person dropping or collapsing to the ground or a lower surface.",
#     "touch head (headache)": "Touching head (headache) involves placing one's hands on the head to alleviate pain or discomfort in the head or neck area.",
#     "touch chest (stomachache/heart pain)": "Touching chest (stomachache/heart pain) is a human action that involves placing a hand on the chest to alleviate discomfort or pain in the chest area.",
#     "touch back (backache)": "Touching back (backache) involves placing one's hands on the back to alleviate or investigate pain, discomfort, or tension in the back.",
#     "touch neck (neckache)": "Touching neck (neckache) refers to the action of pressing or rubbing the neck with one's hands, often to relieve pain or discomfort in the neck area.",
#     "nausea or vomiting": "Nausea or vomiting is a condition characterized by the urge to vomit or the act of forcefully expelling stomach contents through the mouth.",
#     "use a fan (with hand or paper)": "Using a fan (with hand or paper) to feel warm involves creating airflow with a handheld or paper fan for the purpose of cooling down.",
#     "punch/slap other person": "Punching/Slapping other person is the act of striking another person with one's fist or open hand, respectively, typically for physical harm or aggression.",
#     "kick other person": "Kicking other person involves striking another person with one's foot, often for physical harm or aggression.",
#     "push other person": "Pushing other person involves applying force to another person with one's body, typically with the intent to move or manipulate them.",
#     "pat on back of other person": "Patting on back of other person is the act of lightly striking or tapping another person's back with one's hand.",
#     "point finger at other person": "Pointing finger at other person involves extending one's finger towards another person, typically to indicate something or express disapproval.",
#     "hug other person": "Hugging other person involves embracing another person with one's arms, often as a gesture of affection or comfort.",
#     "give something to other person": "Giving something to other person involves transferring an object or item to another person, typically as a gift or as part of an exchange.",
#     "touch other person's pocket": "Touching other person's pocket involves coming into contact with or manipulating the pocket of another person's clothing.",
#     "handshake": "Handshaking involves greeting another person by clasping and shaking their hand, typically as a gesture of respect or friendliness.",
#     "walk towards each other": "Walking towards each other is the act of moving one's body towards another person while standing or walking.",
#     "walk apart from each other": "Walking apart from each other involves moving away from another person while standing or walking."
# }

    # --- 核心函数 1: 调用 LLM ---
    def call_gpt4_optimizer(self,target_action, current_desc, confused_action, error_count):
        """
        调用 GPT-4 来优化 Prompt
        """

        # 构造系统提示 (System Prompt)，设定 LLM 的角色
        # system_prompt = (
        #     "You are an expert in human motion analysis and skeleton action recognition. "
        #     "Your goal is to optimize text descriptions of actions to help an AI model distinguish between similar behaviors."
        # )
        #
        # # 构造用户提示 (User Prompt)，包含具体的错误信息
        # user_prompt = f"""
        # Context:
        # We are training a Zero-Shot Skeleton Action Recognition model.
        #
        # Problem:
        # The model is confusing the action '{target_action}' with '{confused_action}'.
        # This error happened {error_count} times in the validation set.
        #
        # Current Description for '{target_action}':
        # "{current_desc}"
        #
        # Task:
        # Rewrite the description for '{target_action}' to make it distinct from '{confused_action}'.
        #
        # Requirements:
        # 1. Focus on kinematic details (joint angles, limb trajectories, temporal phases).
        # 2. Highlight what makes '{target_action}' physically different from '{confused_action}'.
        # 3. Keep the description concise (under 50 words).
        # 4. Return ONLY the new description text, no explanations.
        # """
        # system_prompt = (
        #     "You are an expert in human motion analysis and skeleton action recognition. "
        #     "Your goal is to optimize text descriptions of actions to help an AI model distinguish between similar behaviors."
        # )

        # 构造用户提示
        # user_prompt = f"""
        # Context:
        # We are training a Zero-Shot Skeleton Action Recognition model.
        #
        # Problem:
        # The model is confusing the action '{target_action}' with '{confused_action}'.
        # This error happened {error_count} times in the validation set.
        #
        # Current Description for '{target_action}':
        # "{current_desc}"
        #
        # Task:
        # Rewrite the description for '{target_action}' to make it distinct from '{confused_action}'.
        #
        # Requirements:
        #
        # # 1. Focus on kinematic details (joint angles, limb trajectories, temporal phases).
        # # 2. Highlight what makes '{target_action}' physically different from '{confused_action}'.
        # # 3. Keep the description concise (under 25 words).
        # # 4. Return ONLY the new description text, no explanations.
        #
        # """
        user_prompt=f"""
Prompt:

Please generate a standard description for the human action specified below.

Constraints:

Format: You must strictly follow the sentence structure: "[Action Name]" involves [physical movement], usually [purpose/intent].
Tone: Keep it simple, objective, and standard.
Example:
Input: "Put on a hat/cap"
Output: "Put on a hat/cap" involves putting a piece of headwear onto one's head, usually to shield from sunlight.

Task:
Input: "{target_action}"
Output:

"""
        url = "https://xh.v1api.cc/v1/chat/completions"
        api_key = "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm"  # 替换您的 Key
                    # 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 3. 设置请求体
        data = {
            "model" : "gpt-3.5-turbo",  # 或者 "gpt-3.5-turbo"
            "messages":[
                # {"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_prompt}],
            "temperature" : 0.7,  # 稍微有点创造性

        }

                    # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()  # 检查请求是否成功
        response=response.json()
        # try:
        #     response = openai.ChatCompletion.create(
        #         # model="gpt-4",  # 或者 "gpt-3.5-turbo"
        #         model="gpt-3.5-turbo",  # 或者 "gpt-3.5-turbo"
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": user_prompt}
        #         ],
        #         temperature=0.7,  # 稍微有点创造性
        #     )
        #
        #     # 提取生成的文本
        new_description = response['choices'][0]['message']['content'].strip()
        # 去掉可能的引号
        new_description = new_description.replace('"', '')
        return new_description

        # except Exception as e:
        #     print(f"Error calling OpenAI API: {e}")
        #     return None
#     def call_gpt4_optimizer(self, target_action, current_desc, confused_action, error_count):
#         """
#         调用 DeepSeek API 来优化 Prompt
#         """
#         # 构造系统提示 (System Prompt)，设定 LLM 的角色
#         system_prompt = (
#             "You are an expert in human motion analysis and skeleton action recognition. "
#             "Your goal is to optimize text descriptions of actions to help an AI model distinguish between similar behaviors."
#         )
#
#         # 构造用户提示 (User Prompt)，包含具体的错误信息
#         user_prompt = f"""
#         Context:
#         We are training a Zero-Shot Skeleton Action Recognition model.
#
#         Problem:
#         The model is confusing the action '{target_action}' with '{confused_action}'.
#         This error happened {error_count} times in the validation set.
#
#         Current Description for '{target_action}':
#         "{current_desc}"
#
#         Task:
#         Rewrite the description for '{target_action}' to make it distinct from '{confused_action}'.
#
#         Please rewrite the provided input text following these STRICT rules:
#
# ### RULES:
# 1.  **Format:** Every line MUST start with the **exact action name** followed by the description.
#     *   Example: "Drink water is the act of..."
#     *   Do NOT start with "The action involves..." or "It refers to...".
#
# 2.  **NO Negative/Contrastive Descriptions:**
#     *   **CRITICAL:** Text encoders often ignore negation words.
#     *   **FORBIDDEN:** "distinct from...", "unlike...", "not...", "different than...".
#     *   If you mention "distinct from throwing," the model will just see "throwing" and get confused. Only describe what the action IS.
#
# 3.  **Visual Kinematics over Abstract Intent:**
#     *   Focus on observable body movements (arms, legs, posture, object interaction).
#     *   Avoid purely abstract definitions like "for the purpose of hydration" or "to show respect". Instead, describe the physical gesture (e.g., "raising a cup to the mouth", "bringing hand to forehead").
#
# 4.  **Concise & Generalizable:**
#     *   Keep descriptions under **25 words**. Long sentences dilute the attention mechanism.
#     *   Avoid over-specifying minute details (like "gaze direction" or "exact finger speed") unless they are the *defining* feature of the action.
#
# ### EXAMPLES:
#
# **Bad (Too complex, uses negative, missing start label):**
# "The action involves a rapid downward arm extension, distinct from the raised bent-arm pose of throwing."
# *(Why bad: Missing label, uses 'distinct from', too specific.)*
#
# **Good (Visual, Positive, Concise):**
# "Drop involves opening the hand to release an object and letting it fall downwards."
#
# **Bad (Abstract intent):**
# "Drink water refers to the act of taking in water to quench thirst."
# *(Why bad: 'Quench thirst' is not visual.)*
#
# **Good (Visual):**
# "Drink water involves raising a cup or bottle to the mouth and tilting the head back to swallow."
#
# ---
#
#
#         """
#
#
#         try:
#             # 导入必要的库
#             import requests
#             import json
#             import os
#
#             # 获取 DeepSeek API Key (建议通过环境变量设置)
#             api_key = "sk-7fc35b3296b4427fbe2fdbef9ac4d246"
#             if not api_key:
#                 print("错误: 未找到环境变量 DEEPSEEK_API_KEY")
#                 return None
#
#             # DeepSeek API 端点
#             url = "https://api.deepseek.com/v1/chat/completions"  # 修改为DeepSeek端点[citation:1][citation:8]
#
#             # 请求头
#             headers = {
#                 "Authorization": f"Bearer {api_key}",  # DeepSeek认证方式[citation:1][citation:3]
#                 "Content-Type": "application/json"
#             }
#
#             # 请求体
#             data = {
#                 "model": "deepseek-chat",  # 修改为DeepSeek模型。也可用 "deepseek-reasoner"[citation:8][citation:9]
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 "temperature": 0.7,
#                 "max_tokens": 50  # 可根据需要调整
#             }
#
#             # 发送POST请求
#             response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
#             response.raise_for_status()  # 检查请求是否成功
#
#             # 解析响应 (根据DeepSeek的实际返回结构调整)[citation:8]
#             result = response.json()
#             # 注意: 根据DeepSeek API返回的实际结构访问文本内容
#             # 通常路径是 result['choices'][0]['message']['content']
#             new_description = result['choices'][0]['message']['content'].strip()
#
#             # 去掉可能的引号
#             new_description = new_description.replace('"', '')
#             return new_description
#
#         except requests.exceptions.RequestException as e:
#             print(f"调用DeepSeek API时网络错误: {e}")
#             return None
#         except json.JSONDecodeError as e:
#             print(f"解析DeepSeek API响应JSON错误: {e}")
#             return None
#         except KeyError as e:
#             print(f"在DeepSeek API响应中未找到预期字段: {e}")
#             print(f"API返回的完整响应: {result}")  # 打印响应以便调试
#             return None
#         except Exception as e:
#             print(f"调用DeepSeek API时发生未知错误: {e}")
#             return None
    # --- 核心函数 2: 处理错误报告并更新数据库 ---
    def optimize_prompts_from_errors(self,error_report, database):
        """
        遍历错误报告，逐个优化 Prompt
        """
        print(f"--- Starting Prompt Optimization for {len(error_report)} top errors ---")

        for item in error_report:
            target = item['target_action']
            confused = item['confused_with']
            count = item['error_count']

            # 1. 获取当前的旧描述
            current_desc = database.get(target, "Unknown action")

            print(f"\n[Optimizing]: '{target}' (Confused with '{confused}')")
            print(f"  Old Desc: {current_desc}")

            # 2. 调用 LLM 生成新描述
            new_desc = self.call_gpt4_optimizer(target, current_desc, confused, count)

            if new_desc:
                # print(f"  New Desc: {new_desc}")

                # # 3. 更新数据库 (这里是内存更新，实际使用建议保存到 json 文件)
                # database[target] = new_desc
                #
                # # 标记已更新（可选：防止一轮更新太多次同一个词）
                new_desc = new_desc.strip()

                # 2. 检测 LLM 是否重复了动作名（例如返回了 "Drop involves..."）
                # 如果开头包含动作名（忽略大小写），先把它切掉，避免重复
                if new_desc.lower().startswith(target.lower()):
                    # 切掉动作名长度的字符
                    new_desc = new_desc[len(target):].strip()

                # 3. 处理一种特殊情况：如果切完后开头还有引号或冒号（比如 ": involves..."），也去掉
                new_desc = new_desc.lstrip('":, ')

                # 4. 【强制格式化】：无论 LLM 返回什么，强行加上带引号的动作名
                final_desc = f'"{target}" {new_desc}'

                # 5. 更新数据库
                database[target] = final_desc
                print(f"  New Desc: {final_desc}")
            else:
                print("  Skipped due to API error.")

        print("\n--- Optimization Cycle Completed ---")
        return database

    def save_prompt_database(self,database, filename="updated_prompts.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            for action, description in database.items():
                # f.write(f'"{action}": "{description}"\n')
                f.write(f' {description}\n')
        print(f"✅ Prompt数据库已保存到 {filename}")

    @ex.capture
    def test_epoch(self, epoch,unseen_label,  DA, support_factor):
        self.encoder.eval()
        # self.adapter.eval()
        self.text_adapter.eval()
        self.ske_adapter.eval()
        # class_names = ["drink water", "eat meal/snack", "brush teeth", "brush hair", "drop", "pickup", "throw", "sit down", "stand up (from sitting position)", "clap", "read", "write", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand wave", "kick something", "reach into pocket", "hop (one foot jumping)", "jump up", "make a phone call/answer phone", "play with phone/tablet", "type on a keyboard", "point to something with finger", "take a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (saying stop)", "sneeze/cough", "stagger", "fall", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)", "nausea or vomiting", "use a fan (with hand or paper)", "punch/slap other person", "kick other person", "push other person", "pat on back of other person", "point finger at other person", "hug other person", "give something to other person", "touch other person's pocket", "handshake", "walk towards each other", "walk apart from each other"]
        class_names = ["Drink water", "Eat meal/snack", "Brushing teeth", "Brushing hair", "Drop", "“Pickup”", "Throw", "Sitting down", "Standing up (from sitting position)", "Clapping", "Reading", "Writing", "Tear up paper", "Wear jacket", "Take off jacket", "Wear a shoe", "Take off a shoe", "Wear on glasses", "Take off glasses", "Put on a hat/cap", "Take off a hat/cap", "Cheer up", "Hand waving", "Kicking something", "Reach into pocket", "Hopping (one foot jumping)", "Jumping up", "Make a phone call/Answer phone", "Playing with phone/tablet", "Typing on a keyboard", "Pointing to something with finger", "Taking a selfie", "Checking time (from watch)", "Rubbing two hands together", "Nod headbow", "Shake head", "Wiping face", "Salute", "Putting the palms together", "Crossing hands in front (saying stop)", "Sneeze/cough", "Staggering", "Falling", "Touching head (headache)", "Touching chest (stomachache/heart pain)", "Touching back (backache)", "Touching neck (neckache)", "Nausea or vomiting", "Using a fan (with hand or paper) to feel warm", "Punching/Slapping other person", "Kicking other person", "Pushing other person", "Patting on back of other person", "Pointing finger at the other person", "Hugging other person", "Giving something to other person", "Touching other person's pocket", "Handshaking", "Walking towards each other", "Walking apart from each other"]
        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        all_preds=[]
        all_labels=[]
        acc_list = []
        ent_list = []
        feat_list = []
        old_pred_list = []
        all_ske_features = []  # 存放骨骼特征 (Samples)
        all_labels = []  # 存放真实标签
        text_prototypes = None  # 存放文本特征 (Class Centers)
        text_labels = None  # 存放文本对应的类别ID

        for data, label in tqdm(loader):

            # y_t = label.numpy().tolist()
            # y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            # unseen_language = self.embeddings[unseen_label]
            # unseen_language = torch.from_numpy(unseen_language).float()
            # unseen_language=self.full_language[unseen_label]
            # unseen_language=self.full_language[unseen_label]
            # learnable_context = self.prompt_learner()
            unseen_language = self.embeddings[unseen_label].cuda()
            # print(unseen_label)
            # inference
            feature = self.encoder(data)
            feature = self.ske_adapter(feature)
            unseen_language=self.text_adapter(unseen_language)
            # print(unseen_language.shape)
            if DA:
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred, old_pred, ent, feat = get_acc_v2(feature, unseen_language, unseen_label, label)
                ent_list.append(ent)
                feat_list.append(feat)
                old_pred_list.append(old_pred)
            else:
                acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                # acc_batch, pred = get_acc_dynamic(feature, unseen_language, unseen_label, label)
                all_preds.append(pred.detach().cpu().numpy())
                # all_labels.append(label.detach().cpu().numpy())
                all_ske_features.append(feature.detach().cpu().numpy())
                all_labels.append(label.detach().cpu().numpy())

                # 收集文本特征 (只需要存一次即可，代表类别的中心)
                if text_prototypes is None:
                    text_prototypes = unseen_language.detach().cpu().numpy()

                    # 处理文本对应的标签索引
                    if isinstance(unseen_label, torch.Tensor):
                        text_labels = unseen_label.detach().cpu().numpy()
                    else:
                        text_labels = np.array(unseen_label)
            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p

            acc_list.append(acc_batch)
        all_preds = np.concatenate(all_preds)
        all_label = np.concatenate(all_labels)

        top_errors = self.find_top_confusions(all_preds, all_label, class_names)
        print(top_errors)
        updated_db = self.optimize_prompts_from_errors(top_errors, self.prompt_database)
        # #
        # self.save_prompt_database(updated_db, "optimized_prompts.txt")
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            self.best_epoch = epoch
            # self.save_model()
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # np.save("y_true_3.npy",y_true)
            # np.save("y_pred_3.npy",y_pred)
            # print("save ok!")
        self.test_acc = acc



        # # # --- 3. 数据处理与 t-SNE ---
        # X_ske = np.concatenate(all_ske_features, axis=0)  # Shape: [Total_Samples, 512]
        # y_true = np.concatenate(all_labels, axis=0)  # Shape: [Total_Samples]
        # X_text = text_prototypes  # Shape: [Num_Classes, 512]
        # print(f"骨骼特征形状: {X_ske.shape}")
        # print(f"文本特征形状: {X_text.shape}")
        #
        # # =======================================================
        # # [新增] 计算文本特征到对应骨架簇的距离 (Alignment Metric)
        # # =======================================================
        # print("正在计算文本与骨架簇的对齐距离 (Cosine Distance)...")
        # class_dists = []
        #
        # # 遍历每一个文本类别
        # for i, class_id in enumerate(text_labels):
        #     # 1. 提取当前类的文本特征 (1, 512)
        #     curr_text_feat = X_text[i].reshape(1, -1)
        #
        #     # 2. 提取当前类的所有骨骼样本 (N, 512)
        #     ske_indices = np.where(y_true == class_id)[0]
        #     if len(ske_indices) == 0:
        #         continue  # 如果测试集中没有这个类别的样本，跳过
        #     curr_ske_feats = X_ske[ske_indices]
        #
        #     # 3. 计算余弦距离 (0=重合, 1=正交, 2=相反)
        #     # 我们计算文本点 到 该类所有骨骼点的平均距离
        #     dists = cosine_distances(curr_text_feat, curr_ske_feats)
        #     mean_dist = np.mean(dists)
        #     class_dists.append(mean_dist)
        #
        #     # 如果你想看某些具体类别的表现，可以取消注释下面这行：
        #     print(f"Class {class_id} alignment dist: {mean_dist:.4f}")
        #
        # # 计算全局平均距离 (越小越好)
        # global_align_score = np.mean(class_dists) if class_dists else 0.0
        # print(f"Global Alignment Score (Cosine Dist): {global_align_score:.4f} (越小越好)")
        # # =======================================================
        # # [新增] 计算文本特征到对应骨架簇的距离 (Alignment Metric)
        # # =======================================================
        #
        # print(f"骨骼特征形状 (Samples): {X_ske.shape}")  # 比如 (2000, 512)
        # print(f"文本特征形状 (Prototypes): {X_text.shape}")  # 比如 (20, 512)
        #
        # # --- 3. t-SNE 降维 ---
        # # 关键：将骨骼特征和文本特征拼在一起做 t-SNE，确保空间一致性
        # X_combined = np.concatenate([X_ske, X_text], axis=0)
        #
        # print("正在运行 t-SNE (将 512维 降至 2维)...")
        # tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30)
        # X_embedded = tsne.fit_transform(X_combined)
        #
        # # 拆分结果
        # n_samples = len(X_ske)
        # X_ske_2d = X_embedded[:n_samples]  # 骨骼点的 2D 坐标
        # X_text_2d = X_embedded[n_samples:]  # 文本点的 2D 坐标
        #
        # # --- 4. 可视化绘图 ---
        # plt.figure(figsize=(14, 12))
        #
        # # 获取所有出现的类别
        # unique_classes = np.unique(y_true)
        # # 生成颜色盘
        # colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))

        # A. 绘制骨骼样本 (圆点)
        # for i, cls in enumerate(unique_classes):
        #     # 找到属于该类别的样本索引
        #     idxs = np.where(y_true == cls)
        #     plt.scatter(X_ske_2d[idxs, 0], X_ske_2d[idxs, 1],
        #                 color=colors[i],
        #                 alpha=0.4, s=20, label=f'Class {cls}')

        # B. 绘制文本原型 (五角星)
        # for i, cls in enumerate(text_labels):
        #     # 尝试匹配颜色
        #     c = 'black'  # 默认黑色
        #     if cls in unique_classes:
        #         idx = np.where(unique_classes == cls)[0][0]
        #         c = colors[idx]
        #
        #     plt.scatter(X_text_2d[i, 0], X_text_2d[i, 1],
        #                 color=c, marker='*', s=400, edgecolors='black', linewidth=2,
        #                 label=f'Text {cls}')
        #
        # # 设置图例和标题
        # # 这里的逻辑是只显示文本原型的图例，或者精简图例，防止图例太长
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        #
        # plt.title('t-SNE: Skeleton Features ')
        # plt.tight_layout()
        # plt.savefig('tsne_512_dim_result.png', dpi=300)
        # print("绘图完成，已保存为 tsne_512_ori_result.png")
        # plt.show()

        #
        #
        #
        #

        if DA:
            ent_all = torch.cat(ent_list)
            feat_all = torch.cat(feat_list)
            old_pred_all = torch.cat(old_pred_list)
            z_list = []
            for i in range(len(unseen_label)):
                mask = old_pred_all == i
                class_support_set = feat_all[mask]
                class_ent = ent_all[mask]
                class_len = class_ent.shape[0]
                if int(class_len * support_factor) < 1:
                    z = self.full_language[unseen_label[i:i + 1]]
                else:
                    _, indices = torch.topk(-class_ent, int(class_len * support_factor))
                    z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                z_list.append(z)

            z_tensor = torch.cat(z_list)
            aug_acc_list = []
            for data, label in tqdm(loader):
                # y_t = label.numpy().tolist()
                # y_true += y_t

                data = data.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                unseen_language = z_tensor
                # inference
                feature = self.encoder(data)
                feature = self.adapter(feature)
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)

                # y_p = pred.cpu().numpy().tolist()
                # y_pred += y_p
                aug_acc_list.append(acc_batch)
            aug_acc = torch.tensor(aug_acc_list).mean()
            if aug_acc > self.best_aug_acc:
                self.best_aug_acc = aug_acc
                self.best_aug_epoch = epoch
            self.test_aug_acc = aug_acc

    def initialize(self):
        self.load_data()
        self.load_model()
        # self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
                    # 'encoder': self.encoder.state_dict(),
                    # 'ske_adapter': self.ske_adapter.state_dict(),
                    # 'text_adapter': self.text_adapter.state_dict()
            'promt_learner':self.prompt_learner.state_dict()
        }, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/prompt_learner/prmpt_learner_{}".format(self.best_acc))

        with torch.no_grad():
            learnable_context = self.prompt_learner()
        learnable_context_np = learnable_context.cpu().numpy()
        np.save("/media/zzf/ljn/wsx/PGFA/PGFA-main/data/learnable_context.npy", learnable_context_np)
        self.log.info(f"Learnable context saved to {save_path}, shape: {learnable_context_np.shape}")

    def start(self):
        self.initialize()
        self.optimize()
        # self.save_model()


class SotaProcessor:

    @ex.capture
    def load_data(self, sota_train_list, sota_train_label,
                  sota_test_list, sota_test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        self.test_aug_acc = -1
        self.best_aug_acc = -1
        self.best_aug_epoch = -1

        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()
        self.dataset['train'] = DataSet(sota_train_list, sota_train_label)
        self.dataset['test'] = DataSet(sota_test_list, sota_test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=15,
                             loss_mode='step', step=[50, 80]):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max - lr_min) * (
                        1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @ex.capture
    def load_model(self, in_channels, hidden_channels, hidden_dim,
                   dropout, graph_args, edge_importance_weighting, model_choice_for_sota):
        if model_choice_for_sota == 'st-gcn':
            self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                 hidden_dim=hidden_dim, dropout=dropout,
                                 graph_args=graph_args,
                                 edge_importance_weighting=edge_importance_weighting,
                                 )
        else:
            self.encoder = ShiftGCN()
        self.encoder = self.encoder.cuda()
        self.adapter=Linear().cuda()
        self.loss = KLLoss().cuda()
        self.logit_scale = self.adapter.get_logit_scale()
        self.logit_scale_v2 = self.adapter.get_logit_scale_v2()

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.SGD([
            {'params': self.encoder.parameters()},
            {'params': self.adapter.parameters()}],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=False
        )

    @ex.capture
    def optimize(self, epoch_num, DA):  # print -> log.info
        self.log.info("sota track")
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] train loss: {}".format(epoch, self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch, self.best_acc))
            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch, self.test_aug_acc))
                self.log.info("epoch [{}] gets the best DA acc: {}".format(self.best_aug_epoch, self.best_aug_acc))

    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type):
        self.encoder.train()  # eval -> train
        self.adapter.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5,
                                  loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            # data = data[:, :, 1:49, :, :]
            data = F.interpolate(data, scale_factor=(1.28, 1, 1), mode='trilinear', align_corners=False)
            # data = F.interpolate(data, scale_factor=(0.96, 1, 1), mode='trilinear', align_corners=False)

            # print(data.shape) #128,3,50,25,2
            # label = label.type(torch.LongTensor).cuda()
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()
            # print(label.shape) # 128
            # print(label) # int
            seen_language = self.full_language[label]  # 128, 768
            # print(seen_language.shape)

            feat = self.encoder(data)
            skleton_feat = self.adapter(feat)
            if loss_type == "kl":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            else:
                raise Exception('loss_type Error!')

            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, sota_unseen, epoch, DA, support_factor):
        self.encoder.eval()
        self.adapter.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        acc_list = []
        ent_list = []
        feat_list = []
        old_pred_list = []
        for data, label in tqdm(loader):
            # y_t = label.numpy().tolist()
            # y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            # data = data[:, :, 1:49, :, :]
            data = F.interpolate(data, scale_factor=(1.28, 1, 1), mode='trilinear', align_corners=False)
            # data = F.interpolate(data, scale_factor=(0.96, 1, 1), mode='trilinear', align_corners=False)
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[sota_unseen]
            # inference
            feature = self.encoder(data)
            feature = self.adapter(feature)
            if DA:
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred, old_pred, ent, feat = get_acc_v2(feature, unseen_language, sota_unseen, label)
                ent_list.append(ent)
                feat_list.append(feat)
                old_pred_list.append(old_pred)
            else:
                acc_batch, pred = get_acc(feature, unseen_language, sota_unseen, label)

            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p

            acc_list.append(acc_batch)

        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            self.save_model()
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # np.save("y_true_3.npy",y_true)
            # np.save("y_pred_3.npy",y_pred)
            # print("save ok!")
        self.test_acc = acc

        if DA:
            ent_all = torch.cat(ent_list)
            feat_all = torch.cat(feat_list)
            old_pred_all = torch.cat(old_pred_list)
            z_list = []
            for i in range(len(sota_unseen)):
                mask = old_pred_all == i
                class_support_set = feat_all[mask]
                class_ent = ent_all[mask]
                class_len = class_ent.shape[0]
                if int(class_len * support_factor) < 1:
                    z = self.full_language[sota_unseen[i:i + 1]]
                else:
                    _, indices = torch.topk(-class_ent, int(class_len * support_factor))
                    z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                z_list.append(z)

            z_tensor = torch.cat(z_list)
            aug_acc_list = []
            for data, label in tqdm(loader):
                # y_t = label.numpy().tolist()
                # y_true += y_t

                data = data.type(torch.FloatTensor).cuda()
                # data = data[:, :, 1:49, :, :]
                data = F.interpolate(data, scale_factor=(1.28, 1, 1), mode='trilinear', align_corners=False)
                # data = F.interpolate(data, scale_factor=(0.96, 1, 1), mode='trilinear', align_corners=False)
                label = label.type(torch.LongTensor).cuda()
                unseen_language = z_tensor
                # inference
                feature = self.encoder(data)
                feature = self.adapter(feature)
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred = get_acc(feature, unseen_language, sota_unseen, label)

                # y_p = pred.cpu().numpy().tolist()
                # y_pred += y_p
                aug_acc_list.append(acc_batch)
            aug_acc = torch.tensor(aug_acc_list).mean()
            if aug_acc > self.best_aug_acc:
                self.best_aug_acc = aug_acc
                self.best_aug_epoch = epoch
            self.test_aug_acc = aug_acc

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'encoder': self.encoder.state_dict(), 'adapter': self.adapter.state_dict()}, save_path)

    def start(self):
        self.initialize()
        self.optimize()


# %%
@ex.automain
def main(track):
    if "sota" in track:
        p = SotaProcessor()
    elif "main" in track:
        p = Processor()
    p.start()
