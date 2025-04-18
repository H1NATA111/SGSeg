
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .selfattention import SelfAttentionBlock


'''FeaturesMemory'''
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, 
                 num_feats_per_cls=1, use_hard_aggregate=False):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.use_hard_aggregate = use_hard_aggregate
        # init memory
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)# 171，1 ，768
        self.text = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)# 171，1 ，768
        # define self_attention module
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self_attention = SelfAttentionBlock(
                    key_in_channels=feats_channels,
                    query_in_channels=feats_channels,
                    transform_channels=transform_channels,
                    out_channels=feats_channels,
                    share_key_query=False,
                    query_downsample=None,
                    key_downsample=None,
                    key_query_num_convs=2,
                    value_out_num_convs=1,
                    key_query_norm=True,
                    value_out_norm=True,
                    matmul_norm=True,
                    with_out_project=True,
                    
                )
                self.self_attentions.append(self_attention)
            self.fuse_memory_conv = nn.Sequential(
                nn.Conv2d(feats_channels * self.num_feats_per_cls, feats_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.SyncBatchNorm(num_features=feats_channels),
                nn.ReLU(),
            )
        else:
            self.self_attention = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
                
            )
        # whether need to fuse the contextual information within the input image
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(num_features=feats_channels),
            nn.ReLU(),
        )
             
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        x1 = x1.float()
        x2 = x2.float()
        dot = torch.sum(x1 * x2, dim)
        norm1 = torch.norm(x1, 2, dim)
        norm2 = torch.norm(x2, 2, dim)
        cos_sim = dot / (norm1 * norm2).clamp(min=eps)
        return cos_sim

    '''forward'''
    def forward(self, feats, preds, text):#2, 768,24,24;          2,171,24,24          2,171,1,768

        '''text = text_feats[0, :, :, :]
        text = text.squeeze(1)# 171,768           200,768
        memory_feature = self.memory.data[:, 0, :]#171,768

        if self.__dict__['training'] == True:
            return memory_feature
            

        
        if self.__dict__['training'] == False:
            text_coco = self.text.data[:, 0, :]#171,768
            _,num_clses,_,_ = text_feats.shape
            
            cos_sim = F.cosine_similarity(text.unsqueeze(1), text_coco.unsqueeze(0), dim=2)
            # 找出相似度最高的索引
            max_sim_indices = torch.argmax(cos_sim, dim=1)
            # 根据索引提取特征向量
            selected_features = torch.index_select(memory_feature, 0, max_sim_indices)
            # 构建新的tensor D
            memory_new_feature = selected_features
            return memory_new_feature#200,768'''


        #原版记忆模块
        batch_size, num_channels, h, w = feats.size()
        _,num_classes,_,_ = preds.shape
        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()#torch.Size([3, 24, 24, 171])
        weight_cls = weight_cls.reshape(-1, num_classes)#torch.Size([1728, 171])
        weight_cls = F.softmax(weight_cls, dim=-1)#torch.Size([1728, 171])
        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory_list = []
        if num_classes == 171:
            memory = self.memory.data[:, 0, :]#torch.Size([171, 768])
        else:
            text_coco = self.text.data[:, 0, :]#171,768
            text = text.data[0, :, :, :]#200,1,768
            text = text.squeeze(1)#200,768
            coco_memory_feature = self.memory.data[:, 0, :]#171,768
            # 计算余弦相似度
            D = torch.zeros_like(text)
            for i in range(len(text)):
                a = text[i].unsqueeze(0)#[1,768]
                similarities = torch.cosine_similarity(a, text_coco, dim=1)
                max_index = torch.argmax(similarities)
                D[i] = coco_memory_feature[max_index]
            memory = D




            
        selected_memory = torch.matmul(weight_cls, memory)#torch.Size([1728, 768])
        selected_memory_list.append(selected_memory.unsqueeze(1))
        # calculate selected_memory according to the num_feats_per_cls
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                # --(B*H*W, C) --> (B, H, W, C)
                selected_memory = selected_memory.view(batch_size, h, w, num_channels)
                # --(B, H, W, C) --> (B, C, H, W)
                selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
                # --append
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
            # --concat
            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            selected_memory = self.fuse_memory_conv(selected_memory)
        else:
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0].squeeze(1)#torch.Size([1728, 768])
            # --(B*H*W, C) --> (B, H, W, C)
            selected_memory = selected_memory.view(batch_size, h, w, num_channels)#torch.Size([3, 24, 24, 768])
            # --(B, H, W, C) --> (B, C, H, W)
            selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()#torch.Size([3, 768, 24, 24])
            # --feed into the self attention module
            selected_memory = self.self_attention(feats, selected_memory)#torch.Size([2, 24, 24, 768])
        # return
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        return  memory_output
    '''update'''
    def update(self, features, segmentation,text, ignore_index=255, strategy='cosine_similarity'):#torch.Size([2, 512, 512, 512]),torch.Size([2, 512, 512])
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()#3,768,384,384
        momentum = 0.9
        # use features to update memory
        segmentation = segmentation.long()#torch.Size([3, 384, 384])
        features = features.permute(0, 2, 3, 1).contiguous()#torch.Size([3, 384, 384, 768])
        features = features.view(batch_size * h * w, num_channels)#torch.Size([442368, 768])
        clsids = segmentation.unique()#torch.Size([27])
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)#torch.Size([442368])
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]#torch.Size([42233, 768])
            # --init memory by using extracted features
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update: continue
            # --update according to the selected strategy
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))#42233
                    weight = (1 - similarity) / (1 - similarity).sum()#42233
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)#torch.Size([768])
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)#torch.Size([1, 768])
                self.memory[clsid].data.copy_(feats_cls)
            else:
                assert strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1), 
                    F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = (argmax == idx)
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)
        text1 = text[0, :, :, :]#171,1,768
        self.text.data.copy_(text1)
