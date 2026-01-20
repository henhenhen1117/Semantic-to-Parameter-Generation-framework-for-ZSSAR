from config import *
from dataset import DataSet
from logger import Log
from Dataset.__init__trainsfer import BaseTransferDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm
from module.LinearMappin import LinearInMapper
from module.gcn.st_gcn import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2

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
        self.best_loss = 10000
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()
        # self.dataset['train']=BaseTransferDataset("/media/zzf/ljn/wsx/PGFA/PGFA-main/output/weight/checkpoint",normalize=True)
        # self.dataset['train']=BaseTransferDataset("./output/learnable_weight",normalize=True)
        self.dataset['train']=BaseTransferDataset("/media/zzf/ljn/wsx/output/weight_split_12/split_12",normalize=True)
        self.in_ske,self.in_text=self.dataset['train'].load_shape()
        # self.dataset['train'] = DataSet(train_list, train_label)
        # self.dataset['test'] = DataSet(test_list, test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=1,
            num_workers=0,
            shuffle=True,
            drop_last=True)

        # self.data_loader['test'] = torch.utils.data.DataLoader(
        #     dataset=self.dataset['test'],
        #     batch_size=64,
        #     num_workers=16,
        #     shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict['mappin'])

    def load_bias_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.bias_mapper.load_state_dict(pretrained_dict['bias'])

    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15, loss_mode='step', step=[50, 80]):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
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
        out = (feature-mean) / torch.sqrt(var)

        return out

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, visual_size, language_size, weight_path, loss_type, fix_encoder, finetune):
        # self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
        #                     hidden_dim=hidden_dim,dropout=dropout,
        #                     graph_args=graph_args,
        #                     edge_importance_weighting=edge_importance_weighting,
        #                     )
        # self.encoder = self.encoder.cuda()
        # self.adapter = Linear().cuda()
        self.mapper=LinearInMapper( in_text=self.in_text,in_ske=self.in_ske, rank=512, map_bias=True).cuda()
        # self.mapper=LinearInMapper().cuda()
        # self.mapper=Linear(768,256).cuda()
        # self.mapper=ResidualMLP().cuda()
        # self.mapper = nn.Sequential(
        #     nn.Linear(768, 1024),  # 升维而不是降维
        #     # nn.LayerNorm(512),  # 只在中间层使用LayerNorm
        #     nn.ReLU(),  # 使用ReLU而不是GELU
        #     # 移除Dropout          # ✅ 关键：移除Dropout
        #     nn.Linear(1024, 256),  # 直接映射到目标维度
        # ).cuda()
        # self.mapper=LinearInMapper().cuda()

        # self.bias_mappin=Linear(512,512).cuda()
        # self.bias_mapper = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     # nn.BatchNorm1d(1024),
        #     nn.LayerNorm(1024),
        #     nn.ReLU(),
        #     # nn.Dropout(0.05),
        #     nn.Linear(1024, 512),
        #     # nn.BatchNorm1d(512),
        #     # nn.LayerNorm(512),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     # nn.Linear(512, 512)
        # ).cuda()
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
        # self.logit_scale = self.adapter.get_logit_scale()
        # self.logit_scale_v2 = self.adapter.get_logit_scale_v2()
        #
        if fix_encoder or finetune:
            self.load_weights(self.encoder, weight_path)
        # self.load_weights(self.mapper, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/mapper/mapper_epoch199_loss:0.09.pt")
        # self.load_bias_weights(self.mapper, "/media/zzf/ljn/wsx/PGFA/PGFA-main/output/mapper/mapper_best_add_layernorm_loss:0.0004.pt")


    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        # self.optimizer = torch.optim.SGD([
        #     {'params': self.encoder.parameters()},
        #     {'params': self.adapter.parameters()}],
        #      lr=lr,
        #      weight_decay=weight_decay,
        #      momentum=0.9,
        #      nesterov=False
        #      )
          self.optimizer = torch.optim.SGD([
            # {'params': self.encoder.parameters()},
            # {'params': self.adapter.parameters()}
            {'params': self.mapper.parameters()}
            # {'params': self.bias_mapper.parameters()}
          ],
             # lr=lr,
             # lr = 0.05,
             lr = 0.05,
              # weight_decay = 1e-5,
             # weight_decay=0.0005,#ori
             weight_decay=0.00005,
             momentum=0.9,
             nesterov=False
             )

          self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epoch_num, eta_min=1e-6
            )

    # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num, DA): # print -> log.info
        self.log.info("main track") 
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            # with torch.no_grad():
            #     self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] train loss: {}".format(epoch,self.dim_loss))
            self.log.info("epoch [{}] get best loss: {}".format(self.best_epoch,self.best_loss))
            # self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            # self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch,self.test_aug_acc))
                self.log.info("epoch [{}] gets the best DA acc: {}".format(self.best_aug_epoch,self.best_aug_acc))
            # if epoch > 5:
            #     self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            #     self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            # else:
            #     self.log.info("epoch [{}] : warm up epoch.".format(epoch))

    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type, alpha, beta, m, fix_encoder):
        # self.encoder.train() # eval -> train
        # if fix_encoder:
        #     self.encoder.eval()
        self.mapper.train()

        # self.vae.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5, loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']
        mse_loss=nn.MSELoss()
        for W_ske, b_ske, W_text, b_text in tqdm(loader):#data shape:128,3,50,25,2;N,C,T,V,M
            W_ske = W_ske.cuda()
            W_text = W_text.cuda()
            if b_ske is not None:
                b_ske = b_ske.cuda()
            if b_text is not None:
                b_text = b_text.cuda()


            # W_pred, b_pred = self.mapper(W_text, b_text)
            W_pred,b_pred=self.mapper(W_text,b_text)
            # b_pred=self.bias_mapper(b_text)
            loss_b=mse_loss(b_pred,b_ske)

            loss_w=mse_loss(W_pred,W_ske)
            loss=loss_w+0.1*loss_b
            # loss_w = mse_loss(W_pred, W_ske)
            # if b_text is not None and b_pred is not None:
            #     loss_b = mse_loss(b_pred, b_ske)
            # # loss = loss_w + 0.1 * loss_b
            # loss=loss_w+0.5*loss_b
            # loss.backward()
            # self.optimizer.step()
                
            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), max_norm=1.0)
        # self.scheduler.step()
        # current_lr = self.optimizer.param_groups[0]['lr']
        # self.log.info(f"epoch [{epoch}] current lr: {current_lr:.6f}")
        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()
        if self.best_loss > self.dim_loss:
            self.best_loss = self.dim_loss

            self.best_epoch=epoch
            self.save_model()

    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
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
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[unseen_label]
            # inference
            feature = self.encoder(data)
            feature = self.adapter(feature)
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
            for i in range(len(unseen_label)):
                mask = old_pred_all == i
                class_support_set = feat_all[mask]
                class_ent = ent_all[mask]
                class_len = class_ent.shape[0]
                if int(class_len*support_factor) < 1:
                    z = self.full_language[unseen_label[i:i+1]]
                else:
                    _, indices = torch.topk(-class_ent, int(class_len*support_factor))
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
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
            # 'encoder':self.encoder.state_dict(),
            #  'adapter':self.adapter.state_dict()
            'mappin': self.mapper.state_dict(),
            # 'bias': self.bias_mapper.state_dict()
            # }, './output/mapper/mapper_epoch{}_loss:{}.pt'.format(self.best_epoch,self.best_loss))
            }, 'weight_log/mapper_best.pt')

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
    
    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15, loss_mode='step', step=[50, 80]):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, model_choice_for_sota):
        if model_choice_for_sota == 'st-gcn':
            self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                                hidden_dim=hidden_dim,dropout=dropout, 
                                graph_args=graph_args,
                                edge_importance_weighting=edge_importance_weighting,
                                )
        else:
            self.encoder = ShiftGCN()
        self.encoder = self.encoder.cuda()
        self.adapter = Linear().cuda()
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
    def optimize(self, epoch_num, DA): # print -> log.info
        self.log.info("sota track")
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] train loss: {}".format(epoch,self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch,self.test_aug_acc))
                self.log.info("epoch [{}] gets the best DA acc: {}".format(self.best_aug_epoch,self.best_aug_acc))

    
    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type):
        self.encoder.train() # eval -> train
        self.adapter.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5, loss_mode=loss_mode, step=step)
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
            seen_language = self.full_language[label] # 128, 768
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
                if int(class_len*support_factor) < 1:
                    z = self.full_language[sota_unseen[i:i+1]]
                else:
                    _, indices = torch.topk(-class_ent, int(class_len*support_factor))
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
        torch.save({'encoder':self.encoder.state_dict(), 'adapter':self.adapter.state_dict()}, save_path)

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
