from dataset import dataset
from utils import semantic_embed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, test]')
parser.add_argument('--dataset_path', required=True, type=str, help='Path of the datasets')
parser.add_argument('--split_index', required=True, type=int, help='Index for splitting of classes')
parser.add_argument('--semantic', default='word2vec', type=str, help='Semantic choice: [word2vec, fasttext, sent2vec, clip, clip_manual]')
parser.add_argument('--vit_backbone', default='ViT-B/16', type=str, help='Backbonesof clip: [ViT-B/16, ViT-B/32, ViT-L/14]')
parser.add_argument('--save_name', required=True, type=str, help='Save name of tsne embeddings')
opt = parser.parse_args()

if __name__ == '__main__':
    _, _, classes, _ = dataset.get_test_data(opt.dataset, opt.dataset_path)
    train_classes, test_classes = dataset.get_split(opt.dataset, opt.split_index, classes, opt.dataset_path)

    embd = semantic_embed.semantic_embeddings(opt.semantic, opt.dataset, classes, opt.vit_backbone)
    tsne = TSNE()
    embd = tsne.fit_transform(embd)
    
    label2index = {label: index for index,
                            label in enumerate(sorted(set(classes)))}
    label_array = np.array(
            [label2index[label] for label in classes], dtype=int)

    df = pd.DataFrame()
    df["label"] = classes
    df["x"] = embd[:,0]
    df["y"] = embd[:,1]

    sns.set(rc={'figure.figsize':(12,12)})
    sns_plot = sns.scatterplot(data=df, x="x", y="y", hue="label",
                legend=False, palette=sns.color_palette("hls", 101))

    for line in range(0, df.shape[0]):
        sns_plot.text(df.x[line]+0.01, df.y[line], 
        df.label[line], horizontalalignment='left', size='xx-small', color='black')
    
    sns_plot.grid(False)
    scatter_fig = sns_plot.get_figure()
    scatter_fig.savefig(opt.save_name + '.png')
    scatter_fig.savefig(opt.save_name + '.pdf')