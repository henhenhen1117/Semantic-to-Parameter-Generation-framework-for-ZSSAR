from config import *
from Dataset import AdapterDataset as DataSet
from dataset import DataSet as Zero_DataSet
from logger import Log
from module.pdiff import OneDimVAE,PDiff
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm
import collections
from module.gcn.st_gcn import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2
from module.LinearMappin import LinearInMapper as Mapper

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


# %%
class Processor:

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
        self.best_loss=1000
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()
        self.dataset['train'] = Zero_DataSet(train_list, train_label)
        self.dataset['test'] = Zero_DataSet(test_list, test_label)
        self.unseen_label=test_label
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
        self.divide_slice_length=768
        self.train_set = DataSet("/media/zzf/ljn/wsx/PGFA/PGFA-main/output/enhanced_checkpoints", dim_per_token=768,
                          )
        print("Dataset length:", self.train_set.real_length)#300
        print("input shape:", self.train_set[0][0].flatten().shape)
        self.sequence_length= self.train_set.sequence_length * self.divide_slice_length
        print(f"sequence length: {self.sequence_length}")
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=1,
            num_workers=4,
            persistent_workers=True,
            drop_last=True,
            shuffle=True,
        )

    def load_vae_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict['vae'])

    def load_ske_weights(self, model=None,pretrained_dict=None):
        pretrained_dict = pretrained_dict
        model.load_state_dict(pretrained_dict)

    def load_bias_weights(self, model=None, pretrained_dict=None):
        pretrained_dict = pretrained_dict
        model.load_state_dict(pretrained_dict,strict=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict['encoder'])

    def load_transfer_weight(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        # model.load_state_dict(pretrained_dict,strict=False)
        model.load_state_dict(pretrained_dict['mappin'])

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
        self.mapper=Mapper(768,256,rank=512, map_bias=True).cuda()
        # self.mapper = Mapper().cuda()
        self.ske_adapter = Linear(256, 512).cuda()
        self.text_adapter = Linear(768, 512).cuda()
        self.vae=OneDimVAE([64, 128, 256, 256, 32],128,self.sequence_length,7).cuda(device=1)
        self.diffusion=PDiff(self.sequence_length).cuda(device=1)
        # if loss_type == "kl" or loss_type == "klv2" or loss_type == "kl+cosface" or loss_type == "kl+sphereface" or "kl+margin":
        #     self.loss = KLLoss().cuda()
        # elif loss_type == "mse":
        #     self.loss = nn.MSELoss().cuda()
        # elif loss_type == "kl+mse":
        #     self.loss_kl = KLLoss().cuda()
        #     self.loss_mse = nn.MSELoss().cuda()
        # elif loss_type == "kl+kd":
        #     self.loss = KLLoss().cuda()
        #     self.kd_loss = KDLoss().cuda()
        # else:
        #     raise Exception('loss_type Error!')
        # self.logit_scale = self.ske_adapter.get_logit_scale()
        # self.logit_scale_v2 = self.adapter.get_logit_scale_v2()

        # if fix_encoder or finetune:
        self.load_weights(self.encoder, weight_path)
        # self.load_vae_weights(self.vae, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/VAE_model/vae/best_vae_0.418508917093277.pt")
        # self.load_vae_weights(self.vae, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/VAE_model/best_vae_bestloss:0.06.pt")
        # self.load_vae_weights(self.vae, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/VAE_model/vae/best_vae_0.16012847423553467.pt")
        self.load_transfer_weight(self.mapper,"/media/zzf/ljn/wsx/PGFA/PGFA-main/output/mapper/mapper_best_mappin2.pt")
        # self.load_ske_weights(self.ske_adapter,weight_path)
        self.load_text_weights(self.text_adapter,weight_path)



    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        # self.optimizer = torch.optim.SGD([
        #     # {'params': self.encoder.parameters()},
        #     {'params': self.ske_adapter.parameters()},
        #     {'params': self.text_adapter.parameters()},
        # ],
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     momentum=0.9,
        #     nesterov=False
        # )

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.vae_optimizer, 100)
        self.vae_optimizer = torch.optim.AdamW(
            params=self.vae.parameters(),
            # lr=0.00001,
            lr=0.00002,
            weight_decay=0,
        )
        self.diffusion_optimizer=torch.optim.AdamW(
            params=self.diffusion.parameters(),
            lr=0.0001,
            weight_decay=0,
        )
        self.vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.vae_optimizer,
            T_max=500,
        )
        self.diffusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.diffusion_optimizer,
            T_max=1000,
        )
    @ex.capture
    def optimize(self, epoch_num, DA):  # print -> log.info
        self.log.info("main track")

        for epoch in range(epoch_num):

            self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch, self.best_acc))

    @ex.capture

    def load_text_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict['text_adapter'])
        ske_state=self.mapper(pretrained_dict['text_adapter']['adapter.weight'],pretrained_dict['text_adapter']['adapter.bias'])
        self.load_ske_weights(self.ske_adapter,ske_state)
        # return pretrained_dict
        # model.load_state_dict(pretrained_dict)
    @ex.capture

    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
        self.encoder.eval()
        # self.adapter.eval()
        self.text_adapter.eval()
        self.ske_adapter.eval()
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
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[unseen_label].cuda()
            # inference
            feature = self.encoder(data)
            feature = self.ske_adapter(feature)
            unseen_language=self.text_adapter(unseen_language)
            if DA:
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred, old_pred, ent, feat = get_acc_v2(feature, unseen_language, unseen_label, label)
                ent_list.append(ent)
                feat_list.append(feat)
                old_pred_list.append(old_pred)
            else:
                acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)

            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p

            acc_list.append(acc_batch)

        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            # self.save_model()
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
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_vae_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torch.save({
        #             'encoder': self.encoder.state_dict(),
        #             'ske_adapter': self.ske_adapter.state_dict(),
        #             'text_adapter': self.text_adapter.state_dict()
        # }, save_path)
        torch.save({
                    'vae': self.vae.state_dict(),
                    # 'ske_adapter': self.ske_adapter.state_dict(),
                    # 'text_adapter': self.text_adapter.state_dict()
        }, save_path)

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
                    'encoder': self.encoder.state_dict(),
                    'ske_adapter': self.ske_adapter.state_dict(),
                    'text_adapter': self.text_adapter.state_dict()
        },  save_path)
        # torch.save({
        #             'vae': self.vae.state_dict(),
        #             # 'ske_adapter': self.ske_adapter.state_dict(),
        #             # 'text_adapter': self.text_adapter.state_dict()
        # }, './output/VAE_model/vae/best_vae_{}.pt'.format(self.best_loss))

    def save_diffusion_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torch.save({
        #             'encoder': self.encoder.state_dict(),
        #             'ske_adapter': self.ske_adapter.state_dict(),
        #             'text_adapter': self.text_adapter.state_dict()
        # }, save_path)
        torch.save({
            'diffusion': self.diffusion.state_dict(),
            # 'ske_adapter': self.ske_adapter.state_dict(),
            # 'text_adapter': self.text_adapter.state_dict()
        }, save_path)

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
