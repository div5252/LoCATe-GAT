import os
from collections import defaultdict
import enum
import torch
import scipy.sparse as sp
import pandas as pd
from scipy.stats import entropy
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
from igraph.drawing.colors import RainbowPalette
from modules.semantic import graph_models
from dataset import graph_dataset, dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='attention', type=str, help='Visualisation type: [attention, embedding, entropy]')
parser.add_argument('--action', default='test', type=str, help='Action: [train, test, gzsl_test, fsl_train, fsl_test, sup_train, sup_test]')
parser.add_argument('--split_index', required=True, type=int, help='Index for splitting of classes')
parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, olympics, test]')
parser.add_argument('--dataset_path', required=True, type=str, help='Path of the datasets')
parser.add_argument('--model_path', required=True, type=str, help='KG model path')

parser.add_argument('--network', default='r2plus1d', type=str, help='Network backend choice: [c3d, r2plus1d, clip, clip_classifier, clip_transformer]')
parser.add_argument('--semantic', default='word2vec', type=str, help='Semantic choice: [word2vec, fasttext, sent2vec, clip, clip_manual]')
parser.add_argument('--vit_backbone', default='ViT-B/16', type=str, help='Backbonesof clip: [ViT-B/16, ViT-B/32, ViT-L/14]')
parser.add_argument('--lca_branch', default=3, type=int, help='Number of LCA dilation branches.')

parser.add_argument('--clip_len', default=16, type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips_train', default=1, type=int, help='Number of clips per video (training)')
parser.add_argument('--n_clips_test', default=25, type=int, help='Number of clips per video (testing)')
parser.add_argument('--image_size', default=112, type=int, help='Image size in input.')

parser.add_argument('--batch_size', default=22, type=int, help='Mini-Batchsize size per GPU.')

parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--save_path', required=True, help='Path to save the visualization results')
opt = parser.parse_args()


def draw_entropy_histogram(entropy_array, title, color='blue', uniform_distribution=False, num_bins=30):
    max_value = np.max(entropy_array)
    bar_width = (max_value / num_bins) * (1.0 if uniform_distribution else 0.75)
    histogram_values, histogram_bins = np.histogram(entropy_array, bins=num_bins, range=(0.0, max_value))

    plt.bar(histogram_bins[:num_bins], histogram_values[:num_bins], width=bar_width, color=color)
    plt.xlabel(f'entropy bins')
    plt.ylabel(f'# of node neighborhoods')
    plt.title(title)

def convert_adj_to_edge_index(adjacency_matrix):
    assert isinstance(adjacency_matrix, np.ndarray), f'Expected NumPy array got {type(adjacency_matrix)}.'
    height, width = adjacency_matrix.shape
    assert height == width, f'Expected square shape got = {adjacency_matrix.shape}.'

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] > 0:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)

def color_edge(weight):
    num_colors = 10
    pal = RainbowPalette(num_colors)
    weight *= (num_colors - 1)
    return pal[int(weight)]


def visualize_gat_properties(opt):
    adj, features, train_dataloader, test_dataloader, val_dataloader, num_train, num_test = graph_dataset.get_kg_datasets(opt)
    fnames, labels, classes, _ = dataset.get_test_data(opt.dataset, opt.dataset_path)
    subset, _ = dataset.get_split(opt.dataset, opt.split_index, classes, opt.dataset_path)
    fnames1, labels1, classes1, fnames2, labels2, classes2 = dataset.subset_classes(subset,
        fnames, labels, classes)
    
    classes = np.append(classes1, classes2)

    # print(adj.shape)
    # print(features.shape)

    adj = adj.to(opt.device)
    features = features.to(opt.device)
    adj = adj.float()
    features = features.float()

    nclass = 512
    model = graph_models.GAT(nfeat=features.shape[1], nclass=nclass)
    if os.path.isfile(opt.model_path):
        model.load_state_dict(torch.load(opt.model_path))
        print("LOADED KG MODEL:  " + opt.model_path + "\n")
    model.to(opt.device)

    model.eval()
    with torch.no_grad():
        all_nodes_unnormalized_scores = model(features, adj)  # shape = (N, num of classes)
        # print(all_nodes_unnormalized_scores.shape)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    # print(all_nodes_unnormalized_scores.sum())

    adj_np = adj.cpu().detach().numpy()
    edge_index = convert_adj_to_edge_index(adj_np)
    # print(edge_index)

    if opt.type == 'attention':
        # The number of nodes for which we want to visualize their attention over neighboring nodes
        # (2x this actually as we add nodes with highest degree + random nodes)
        num_nodes_of_interest = 4  # 4 is an arbitrary number you can play with these numbers
        head_to_visualize = 0  # plot attention from this multi-head attention's head
        gat_layer_id = 0  # plot attention from this GAT layer

        # Build up the complete graph
        # node_features shape = (N, FIN), where N is the number of nodes and FIN number of input features
        total_num_of_nodes = len(features)
        complete_graph = ig.Graph()
        complete_graph.add_vertices(total_num_of_nodes)  # igraph creates nodes with ids [0, total_num_of_nodes - 1]
        edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))  # igraph requires this format
        complete_graph.add_edges(edge_index_tuples)

        nodes_of_interest_ids = [i for i in range(len(classes))]

        target_node_ids = edge_index[1]
        source_nodes = edge_index[0]

        for target_node_id in nodes_of_interest_ids:
            # Step 1: Find the neighboring nodes to the target node
            src_nodes_indices = np.equal(target_node_ids, target_node_id)
            # print(src_nodes_indices.shape)
            source_node_ids = source_nodes[src_nodes_indices]
            
            # remove self loop
            source_node_ids = [id for id in source_node_ids if id < len(classes)]

            # print(target_node_id, source_node_ids)
            size_of_neighborhood = len(source_node_ids)

            # Step 2: Fetch their labels
            labels = classes[source_node_ids]
            # print("this", classes[target_node_id])
            # print("Neighbors", labels)

            all_attention_weights = model.attentions1[head_to_visualize].attention_weights
            attention_weights = all_attention_weights[source_node_ids].cpu().numpy()

            # Build up the neighborhood graph whose attention we want to visualize
            # igraph constraint - it works with contiguous range of ids so we map e.g. node 497 to 0, 12 to 1, etc.
            id_to_igraph_id = dict(zip(source_node_ids, range(len(source_node_ids))))
            ig_graph = ig.Graph()
            ig_graph.add_vertices(size_of_neighborhood)
            ig_graph.add_edges([(id_to_igraph_id[neighbor], id_to_igraph_id[target_node_id]) for neighbor in source_node_ids])

            edge_weights = attention_weights[:, target_node_id]
            edge_weights = [edge_weight if id != target_node_id else 0 for (edge_weight, id) in zip(edge_weights, source_node_ids)]
            # print(edge_weights)

            # Normalized to 1
            if np.max(edge_weights) != 0:
                edge_weights /= np.max(edge_weights)
            edge_color = [color_edge(edge_weight) for edge_weight in edge_weights]

            edge_weights = [5 if id != target_node_id else 0 for (edge_weight, id) in zip(edge_weights, source_node_ids)]

            vertex_color = ["gray" if id != target_node_id else "orange" for id in source_node_ids]

            fig, ax = plt.subplots()
            # Prepare the visualization settings dictionary and plot
            visual_style = {
                "edge_width": edge_weights,
                "layout": ig_graph.layout_reingold_tilford_circular(),  # layout for tree-like graphs
                "target": ax,
                "vertex_label": labels,
                "vertex_color": vertex_color,
                "edge_color": edge_color,
            }
            
            ig.plot(ig_graph, **visual_style)

            if os.path.exists(opt.save_path) == False:
                os.mkdir(opt.save_path)
            if not os.path.exists(os.path.join(opt.save_path, opt.type)):    
                os.mkdir(os.path.join(opt.save_path, opt.type))

            dataset_path = os.path.join(opt.save_path, opt.type, opt.dataset+str(opt.split_index))
            if os.path.exists(dataset_path) == False:
                os.mkdir(dataset_path)

            fig.savefig(os.path.join(dataset_path, str(target_node_id) + '.png'), bbox_inches='tight')
            plt.close(fig)
            # ig_graph.save(opt.save_path)

    elif opt.type == 'embedding':  # visualize embeddings (using t-SNE)
        num_classes = len(classes)

        t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(all_nodes_unnormalized_scores)
        # print(t_sne_embeddings.shape)

        df = pd.DataFrame()
        df["label"] = classes
        df["x"] = t_sne_embeddings[:num_classes,0]
        df["y"] = t_sne_embeddings[:num_classes,1]

        sns.set(rc={'figure.figsize':(12,12)})
        sns_plot = sns.scatterplot(data=df, x="x", y="y", hue="label",
                    legend=False, palette=sns.color_palette("hls", 101))

        for line in range(0, df.shape[0]):
            sns_plot.text(df.x[line]+0.01, df.y[line], 
            df.label[line], horizontalalignment='left', size='xx-small', color='black')
        
        sns_plot.grid(False)
        scatter_fig = sns_plot.get_figure()
        scatter_fig.savefig(opt.save_path)

    elif opt.type == 'entropy':
        num_heads_per_layer = [4, 4, 6] # heads defined in layers.py
        num_layers = len(num_heads_per_layer)
        num_of_nodes = len(features)

        source_node_ids = edge_index[0]
        target_node_ids = edge_index[1]
        # print(target_node_ids.shape)

        attention_layers = [model.attentions1, model.attentions2, model.out_att]

        # For every GAT layer and for every GAT attention head plot the entropy histogram
        for layer_id in range(num_layers):
            for head_id in range(num_heads_per_layer[layer_id]):
                all_attention_weights = attention_layers[layer_id][head_id].attention_weights.cpu().numpy()

                uniform_dist_entropy_list = []  # save the ideal uniform histogram as the reference
                neighborhood_entropy_list = []

                # This can also be done much more efficiently via scatter_add_ (no for loops)
                # pseudo: out.scatter_add_(node_dim, -all_attention_weights * log(all_attention_weights), target_index)
                for target_node_id in range(num_of_nodes):  # find the neighborhood for every node in the graph
                    # These attention weights sum up to 1 by GAT design so we can treat it as a probability distribution
                    neighborhood = [e[1] for e in edge_index if e[0] == target_node_id]
                    
                    neigborhood_attention = all_attention_weights[neighborhood].flatten()
                    # Reference uniform distribution of the same length
                    ideal_uniform_attention = np.ones(len(neigborhood_attention))/len(neigborhood_attention)

                    # Calculate the entropy, check out this video if you're not familiar with the concept:
                    # https://www.youtube.com/watch?v=ErfnhcEV1O8 (Aurélien Géron)
                    neighborhood_entropy_list.append(entropy(neigborhood_attention, base=2))
                    uniform_dist_entropy_list.append(entropy(ideal_uniform_attention, base=2))

                title = f'{opt.dataset} entropy histogram layer={layer_id}, attention head={head_id}'
                draw_entropy_histogram(uniform_dist_entropy_list, title, color='orange', uniform_distribution=True)
                draw_entropy_histogram(neighborhood_entropy_list, title, color='dodgerblue')

                fig = plt.gcf()  # get current figure
                if os.path.exists(opt.save_path) == False:
                    os.mkdir(opt.save_path)
                fig.savefig(os.path.join(opt.save_path, f'layer_{layer_id}_head_{head_id}.jpg'))


if __name__ == '__main__':
    if torch.cuda.is_available():
        opt.device = 'cuda'
    elif torch.has_mps:
        opt.device = 'mps'
    else:
        opt.device = 'cpu'

    visualize_gat_properties(opt)
