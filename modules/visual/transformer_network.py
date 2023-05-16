import torch
import torch.nn as nn
import clip
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from modules.visual.lca import LCA, LCA_branch_1, LCA_branch_2

class CLIPVideo(nn.Module):
    def __init__(self, vit_backbone="ViT-B/16"):
        super(CLIPVideo, self).__init__()
        device = 'cuda'
        self.model, _ = clip.load(vit_backbone, device=device)

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = self.model.encode_image(x) # Input requirement h=w=224
        # (bs*nc*l, 512)
        video_features = video_features.reshape(bs, nc*l, -1) # (bs, nc*l, 512)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    
class CLIPClassifier(nn.Module):
    def __init__(self, out_features, vit_backbone="ViT-B/16"):
        super(CLIPClassifier, self).__init__()
        self.regressor = nn.Linear(512, out_features)
        self.vit_backbone = vit_backbone

    def forward(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float()
        video_features = video_features.reshape(bs, nc*l, -1) # (bs, nc*l, 512)
        video_features = torch.mean(video_features, 1) # (bs, 512)

        video_features = self.regressor(video_features)
        return video_features
    
    def output(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float()
        video_features = video_features.reshape(bs, nc*l, -1) # (bs, nc*l, 512)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features


class TransformerBlock(nn.Module):
    def __init__(self, LCA_drops, d_model, n_head, drop_attn=0.0, droppath=0.0, lca_branch=3):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=drop_attn)
        if lca_branch >= 3:
            self.lca = LCA(LCA_drops, embed_dim=d_model)
        elif lca_branch == 2:
            self.lca = LCA_branch_2(LCA_drops, embed_dim=d_model)
        else:
            self.lca = LCA_branch_1(LCA_drops, embed_dim=d_model)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        #can keep droppath rate = 0.2 for ViTB/16. Source: 

    def attention(self, x):
        return self.mha(x, x, x, need_weights=False)[0]
        
    def forward(self, x):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.lca(self.ln_2(x)))
        return x
    

class CLIPTransformer(nn.Module):
    def __init__(self, T, LCA_drops, num_blocks=2, embed_dim=512, drop_attn=0.0, droppath=0.0, vit_backbone="ViT-B/16", lca_branch=3):
        super(CLIPTransformer, self).__init__()
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.vit_backbone = vit_backbone
        self.embed_dim = embed_dim

        n_head = embed_dim // 64
        self.transformer_block1 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath, lca_branch=lca_branch)
        # self.transformer_block2 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)
        # self.transformer_block3 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)

    def forward(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)

        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float() # (bs*nc*l, 512)
        video_features = video_features.reshape(bs*nc, l, -1) # (bs*nc, l, 512)

        # Positional embedding
        video_features = video_features + self.positional_embedding
        video_features = self.transformer_block1(video_features)
        # video_features = self.transformer_block2(video_features)
        # video_features = self.transformer_block3(video_features)

        # (bs*nc, t, 512)
        video_features = video_features.reshape(bs, nc*l, self.embed_dim)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    
class CLIPTransformerClassifier(nn.Module):
    def __init__(self, out_features, T, LCA_drops, num_blocks=2, embed_dim=512, drop_attn=0.0, droppath=0.0, vit_backbone="ViT-B/16", lca_branch=3):
        super(CLIPTransformerClassifier, self).__init__()
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.vit_backbone = vit_backbone
        self.embed_dim = embed_dim

        n_head = embed_dim // 64
        self.transformer_block1 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath, lca_branch=lca_branch)
        # self.transformer_block2 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)
        # self.transformer_block3 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)

        self.regressor = nn.Linear(512, out_features)

    def output(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float() # (bs*nc*l, 512)
        video_features = video_features.reshape(bs*nc, l, -1) # (bs*nc, l, 512)

        # Positional embedding
        video_features = video_features + self.positional_embedding
        video_features = self.transformer_block1(video_features)
        # video_features = self.transformer_block2(video_features)
        # video_features = self.transformer_block3(video_features)

        # (bs*nc, t, 512)
        video_features = video_features.reshape(bs, nc*l, self.embed_dim)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    
    def forward(self, x):
        video_features = self.output(x)
        video_features = self.regressor(video_features)
        return video_features

