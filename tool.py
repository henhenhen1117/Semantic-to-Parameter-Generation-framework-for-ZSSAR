import numpy
import torch
import math
import torch.nn.functional as F
mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def gen_label_from_text_sim(x):
    x = x / x.norm(dim=-1, keepdim=True)
    return x @ x.t()

def get_m_theta(cos_theta, m=4):
    cos_m_theta = mlambda[m](cos_theta)
    temp = cos_theta.clone().detach()
    theta = torch.acos(temp.clamp(-1.+1e-6, 1.-1e-6))
    k = (theta*m / math.pi).floor()
    sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
    phi_theta = sign * cos_m_theta - 2. * k
    return phi_theta
    # d_theta = phi_theta - cos_theta
    # return d_theta + x

#对比学习中的相似度计算
def create_logits(x1, x2, logit_scale, exp=True):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    if exp:
        scale = logit_scale.exp()
    else:
        scale = logit_scale

    # cosine similarity as logits
    logits_per_x1 = scale * x1 @ x2.t()
    logits_per_x2 = logits_per_x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2
#logits_per_x1: 形状 [batch_size, batch_size]，表示每个 x1 样本与所有 x2 样本的相似度
#logits_per_x2: 形状 [batch_size, batch_size]，表示每个 x2 样本与所有 x1 样本的相似度

def create_sim_matrix(x1, x2, alpha=1):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    x1x1 = alpha * x1 @ x1.t()
    x1x2 = alpha * x1 @ x2.t()
    x2x2 = alpha * x2 @ x2.t()
    return x1x1,x1x2,x2x2


def get_acc_dynamic(x1, multi_view_text_feats, unseen_label, label):
    """
    Args:
        x1: [Batch, Dim]
        multi_view_text_feats: [N_Unseen, N_Views, Dim] (已经只包含 Unseen 类了)
        unseen_label: list (用于最后将预测结果映射回全局ID)
        label: [Batch] (Ground Truth 全局ID)
    """

    # 1. 准备数据
    # ★关键修改：直接使用传入的特征，不再索引
    target_text_feats = multi_view_text_feats

    # 容错：如果输入是 2 维 [N, D]，强行升维到 [N, 1, D] 以兼容后续逻辑
    if target_text_feats.dim() == 2:
        target_text_feats = target_text_feats.unsqueeze(1)

    # 2. 归一化
    x1 = F.normalize(x1, dim=-1)  # [B, D]
    target_text_feats = F.normalize(target_text_feats, dim=-1)  # [N_U, V, D]

    B, D = x1.shape
    N_U, V, _ = target_text_feats.shape

    # 3. 计算动态权重 (Attention)
    # [B, 1, 1, D] * [1, N_U, V, D] -> Sum -> [B, N_U, V]
    similarity_scores = (x1.view(B, 1, 1, D) * target_text_feats.unsqueeze(0)).sum(dim=-1)

    # Temperature 0.1
    attn_weights = F.softmax(similarity_scores / 0.1, dim=-1)

    # 4. 融合文本特征
    # [1, N_U, V, D] * [B, N_U, V, 1] -> Sum -> [B, N_U, D]
    weighted_text = (target_text_feats.unsqueeze(0) * attn_weights.unsqueeze(-1)).sum(dim=2)

    # 5. 计算 Logits
    # [B, 1, D] * [B, N_U, D] -> Sum -> [B, N_U]
    logits = (x1.unsqueeze(1) * weighted_text).sum(dim=-1)

    # 6. 计算 Acc
    # pred_idx 是 0 ~ (N_U-1) 的局部索引
    pred_idx = torch.argmax(logits, dim=1)

    # ★关键：将局部索引映射回全局 ID 进行比对
    unseen_label_tensor = torch.tensor(unseen_label, device=x1.device)
    pred_global = torch.index_select(unseen_label_tensor, 0, pred_idx)

    acc = pred_global.eq(label.view_as(pred_global)).float().mean()

    return acc, pred_global


def get_acc(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred

def get_acc_v2(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    unseen_len = len(unseen_label)
    
    old_pred = pred
    ent = softmax_entropy(logits)
    
    # unseen_len = len(unseen_label)
    # for i in range(unseen_len):
    #     class_support_set = x1[pred == i]
    #     class_logit = logits[pred == i]
    #     class_ent = softmax_entropy(class_logit)
    #     _, indices = torch.topk(class_ent, 5)
    #     z = torch.mean(class_support_set[indices], dim=-1)
    #     z_list.append(z)
    
        
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred, old_pred, ent, x1

def get_acc_v3(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    ent = softmax_entropy(logits)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred, ent

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * math.log2(math.e) * x.log_softmax(1)).sum(1)
