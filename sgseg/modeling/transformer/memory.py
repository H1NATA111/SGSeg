
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


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
        self.linear = nn.Sequential(
            nn.Linear(768, 1),
            nn.ReLU(),
        )

        # whether need to fuse the contextual information within the input image
  
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        x1 = x1.float()
        x2 = x2.float()
        dot = torch.sum(x1 * x2, dim)
        norm1 = torch.norm(x1, 2, dim)
        norm2 = torch.norm(x2, 2, dim)
        cos_sim = dot / (norm1 * norm2).clamp(min=eps)
        return cos_sim

    '''forward'''
    def forward(self, feats):#2, 768,24,24;          2,17,24,24          2,17,1,768




        batch_size, num_channels, h, w = feats.size()
        Rn = self.memory.squeeze(1)#17,768
        e1 = self.linear(Rn)#17,1
        e = F.softmax(e1, dim=0)#17,1
        RnT = Rn.transpose(0, 1)
        R = torch.matmul(RnT, e)                   


        return  R
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
