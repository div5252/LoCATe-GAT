from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from modules.visual.cnn_network import C3D, ResNet18
from modules.visual.transformer_network import CLIPVideo, CLIPClassifier, CLIPTransformer, CLIPTransformerClassifier

def get_network(opt):
    """
    Selection function for available networks.
    """
    if opt.semantic == 'word2vec' or opt.semantic == 'fasttext':
        output_features = 300
    elif opt.semantic == 'sent2vec':
        output_features = 600
    elif 'clip' in opt.semantic:
        if opt.vit_backbone in ['ViT-B/16', 'ViT-B/32', 'RN101']:
            output_features = 512
        elif opt.vit_backbone == 'ViT-L/14':
            output_features = 768
        elif opt.vit_backbone in ['RN50']:
            output_features = 1024

    if opt.network == 'c3d':
        return C3D(out_features=output_features, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    elif opt.network == 'r2plus1d':
        return ResNet18(r2plus1d_18, out_features=output_features, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained, weights=R2Plus1D_18_Weights.DEFAULT)
    elif opt.network == 'clip':
        return CLIPVideo(vit_backbone=opt.vit_backbone)
    elif opt.network == 'clip_classifier':
        return CLIPClassifier(out_features=output_features, vit_backbone=opt.vit_backbone)
    elif opt.network == 'clip_transformer':
        return CLIPTransformer(T=opt.clip_len, LCA_drops=opt.LCA_drops, embed_dim=output_features, drop_attn=opt.drop_attn_prob, droppath=opt.droppath, vit_backbone=opt.vit_backbone, lca_branch=opt.lca_branch)
    elif opt.network == 'clip_transformer_classifier':
        return CLIPTransformerClassifier(out_features=output_features, T=opt.clip_len, LCA_drops=opt.LCA_drops, drop_attn=opt.drop_attn_prob, droppath=opt.droppath, vit_backbone=opt.vit_backbone, lca_branch=opt.lca_branch)
    else:
        raise Exception('Network {} not available!'.format(opt.network))


