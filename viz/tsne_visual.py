from dataset import dataset
from utils import visual_network
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

# Parser options - 
parser = argparse.ArgumentParser()
parser.add_argument('--viz', type=str, required=True, help='Type of visualization: [clip, transformer]')

parser.add_argument('--action', default='viz', type=str, help='Action: [train, test, gzsl_test, fsl_train, fsl_test, sup_train, sup_test, viz]')
parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, olympics, test]')
parser.add_argument('--dataset_path', required=True, type=str, help='Path of the datasets')
parser.add_argument('--split_index', required=True, type=int, help='Index for splitting of classes')
parser.add_argument('--novelty', default='LCA', type=str, help='Novel component: [LCA, other]')
parser.add_argument('--early_stop_thresh', default=150, type=int, help='Number of training epochs before early stopping')
parser.add_argument('--t_blocks', default=1, type=int, help='Number of transformer blocks')


parser.add_argument('--network', default='r2plus1d', type=str, help='Network backend choice: [c3d, r2plus1d, clip, clip_classifier]')
parser.add_argument('--vit_backbone', default='ViT-B/16', type=str, help='Backbonesof clip: [ViT-B/16, ViT-B/32, ViT-L/14, RN50, RN101]')
parser.add_argument('--lca_branch', default=3, type=int, help='Number of LCA dilation branches.')
parser.add_argument('--semantic', default='word2vec', type=str, help='Semantic choice: [word2vec, fasttext, sent2vec, clip, clip_manual]')

parser.add_argument('--clip_len', default=16, type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips_train', default=1, type=int, help='Number of clips per video (training)')
parser.add_argument('--n_clips_test', default=25, type=int, help='Number of clips per video (testing)')
parser.add_argument('--image_size', default=224, type=int, help='Image size in input.')

parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs', default=150, type=int, help='Number of training epochs.')
parser.add_argument('--batch_size', default=22, type=int, help='Mini-Batchsize size per GPU.')
parser.add_argument('--drop_attn_prob', default=0.0, type=float, help='Dropout probability for MHSA module.')
parser.add_argument('--droppath', default=0.0, type=float, help='Drop path probability.')



parser.add_argument('--fixconvs', action='store_false', default=True, help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=True, help='Pretrain network.')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers for training.')

parser.add_argument('--trained_weights', default=False, type=str, help='Use trained weights of Brattoli on Kinetics')
parser.add_argument('--no_val', action='store_true', default=False, help='Perform no validation')
parser.add_argument('--val_freq', default=2, type=int, help='Frequency for running validation.')

parser.add_argument('--seed', default=806, help='Seed for initialization')
parser.add_argument('--count_params', action='store_true', default=False, help='Only for counting trainable parameters')

parser.add_argument('--save_path', required=True, type=str, help='Where to save log and checkpoint.')
parser.add_argument('--weights', default=None, type=str, help='Weights to load from a previously run.')
opt = parser.parse_args()

def test(test_dataloader, model, opt):
    # Perform testing.
    model.eval()
    with torch.no_grad():
        n_samples = len(test_dataloader.dataset)
        
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

        predicted_embd = np.zeros([n_samples, output_features], 'float32')
        true_label = np.zeros(n_samples, 'int')
        
        data_iterator = tqdm(test_dataloader)
        it = 0
        for data in data_iterator:
            video, label, embd, idx, seen = data
            if len(video) == 0:
                continue

            # Run network on batch
            pred_embd = model(video.to(opt.device))

            pred_embd_np = pred_embd.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            predicted_embd[it:it + len(label)] = pred_embd_np
            true_label[it:it + len(label)] = label.squeeze()
            it += len(label)

    predicted_embd = predicted_embd[:it]
    true_label = true_label[:it]

    #returned val loss
    return predicted_embd, true_label


def plot_tsne(embd, label, opt):
    tsne = TSNE()
    embd = tsne.fit_transform(embd)

    df = pd.DataFrame()
    df["label"] = all_unseen_classes[label]
    df["x"] = embd[:,0]
    df["y"] = embd[:,1]

    sns.set(rc={'figure.figsize':(25,25)})
    sns_plot = sns.scatterplot(data=df, x="x", y="y", hue="label", s=100,
                legend="brief", palette=sns.color_palette("hls", len(all_unseen_classes)))
    
    mp = {}
    for line in range(0, df.shape[0]):
        if df.label[line] not in mp:
            sns_plot.text(df.x[line]+0.01, df.y[line], 
            df.label[line], horizontalalignment='left', size='xx-small', color='black')
            mp[df.label[line]] = 1
            
    
    sns_plot.grid(False)
    sns.move_legend(sns_plot, "upper left", bbox_to_anchor=(1, 1))
    scatter_fig = sns_plot.get_figure()
    scatter_fig.savefig(opt.viz + '_viz_one.png')
    scatter_fig.savefig(opt.viz + '_viz_one.pdf')



if __name__ == '__main__':
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    
    if torch.cuda.is_available():
        opt.device = 'cuda'
    elif torch.has_mps:
        opt.device = 'mps'
    else:
        opt.device = 'cpu'
    
    dataloaders, all_seen_classes, all_unseen_classes = dataset.load_datasets(opt)

    opt.video_model_path = opt.save_path + '/checkpoint.pth.tar'
    opt.model_path = opt.save_path + '/checkpoint_kg.pth.tar'

    video_model = visual_network.get_network(opt)

    if opt.viz != 'clip':
        if os.path.isfile(opt.video_model_path):
            j = len('module.')
            weights = torch.load(opt.video_model_path)['state_dict']
            model_dict = video_model.state_dict()
            weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
            model_dict.update(weights)
            video_model.load_state_dict(model_dict)

    if opt.viz == 'clip':
        embd, label = test(dataloaders['testing'][0], video_model, opt)
        plot_tsne(embd, label, opt)

    elif opt.viz == 'transformer':
        embd, label = test(dataloaders['testing'][0], video_model, opt)
        plot_tsne(embd, label, opt)

    