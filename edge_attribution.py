import math
import os.path

import captum.metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import copy
import networkx as nx
from torch import nn
from torch_geometric.utils import is_undirected
import datetime

from sklearn.metrics import accuracy_score, top_k_accuracy_score, r2_score, auc


from util import *
from attribution_analysis import visualize_stem_overlap
from graph_inference import GSATViGReasoning
from models.gnn_models import init_model
from models.stem_approaches import get_receptive_field

from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from fvcore.nn import FlopCountAnalysis


class PVIGNodeEncoder(nn.Module):
    def __init__(self, pvig_model, num_backbone_layers_to_apply=None):
        super().__init__()
        self.pvig_model = pvig_model
        self.pvig_model.zero_grad()
        self.pvig_model.eval()
        # self.num_backbone_layers_to_apply = len(self.pvig_model.visual_encoder.backbone)
        # if num_backbone_layers_to_apply is not None:
        #     self.num_backbone_layers_to_apply = num_backbone_layers_to_apply

    def forward(self, input):
        # print(self.pvig_model.visual_encoder.print())
        # x = self.pvig_model.visual_encoder.stem(input) + self.pvig_model.visual_encoder.pos_embed
        # B, C, H, W = x.shape
        # for i in range(self.num_backbone_layers_to_apply):
        #     l = self.pvig_model.visual_encoder.backbone[i]
        #     if i == self.num_backbone_layers_to_apply - 1:
        #         print("Last layer in the backbone: {}".format(l))
        #     x = l(x)
        
        x = self.pvig_model.visual_encoder(input)["image_feature_map"]
        return x
    def print(self):
        return "PVIG_node_encoder"

class GNNExplainerModel(nn.Module):
    def __init__(self, gnn_bottlneck):
        super().__init__()
        self.gnn_bottleneck = gnn_bottlneck
        self.gnn_bottleneck.zero_grad()

    def forward(self, node_embeddings, edge_index, batch,):
        return self.gnn_bottleneck(node_embeddings, edge_index, batch)["prediction"]


def resolve_node_positions(image, node_encoder):
    image.requires_grad = True
    with torch.no_grad():
        node_embeddings = node_encoder(image)
    print(node_embeddings.shape)
    print("Resolving node positions")
    total_num_nodes = node_embeddings.shape[2] * node_embeddings.shape[3]
    print("Total num nodes: {}".format(total_num_nodes))
    node_positions = []
    nodes_receptive_fields_in_px = []
    for idx_row in range(node_embeddings.shape[2]):
        for idx_col in range(node_embeddings.shape[3]):
            node_receptive_field = get_receptive_field(image, node_encoder, idx_row, idx_col)
            nodes_receptive_fields_in_px.append(len(node_receptive_field))
            print("Node receptive field for node {}: {}".format(idx_row * node_embeddings.shape[3] + idx_col, len(node_receptive_field)))
            upper_left = min(node_receptive_field)
            bottom_right = max(node_receptive_field)
            node_positions.append([upper_left, bottom_right])

    return np.array(node_positions), nodes_receptive_fields_in_px


def visualize_backbone_graph(backbone_layer_idx):
    visualization_dir = os.path.join(model_root_dir, "graph_visualizer")
    os.makedirs(visualization_dir, exist_ok=True)
    for idx, data in enumerate(test_data_loader):
        if idx == 856:
            vig_model.zero_grad()
            image = data["image"].cuda()
            image.requires_grad = True

            graph_processing_layer = PVIGNodeEncoder(vig_model, backbone_layer_idx).eval()
            edge_index = {}
            def collect_edge_info():
                def hook(model, input, output):
                    if gnn_approach == "WIGNN":
                        edge_indices = output[1].squeeze().cpu().detach().numpy()
                        blocks, num_windows, window_size, k = edge_indices.shape
                        print(blocks, num_windows, window_size, k)
                        edge_indices = edge_indices.reshape((blocks, num_windows, -1))
                        print("After reshape: {}".format(edge_indices.shape))
                        for window_idx in range(num_windows):
                            for i in range(edge_indices.shape[2]):
                                node1_idx = edge_indices[0][window_idx][i]
                                edge_indices[0, window_idx, i] = window_idx * window_size + node1_idx
                                node2_idx = edge_indices[1][window_idx][i]
                                edge_indices[1][window_idx][i] = window_idx * window_size + node2_idx
                        edge_index[0] = edge_indices.reshape((blocks, -1))
                    else:
                        edge_index[0] = output.squeeze().reshape((2, -1)).cpu().detach().numpy()
                return hook

            if gnn_approach == "WIGNN":
                edges_hook = graph_processing_layer.pvig_model.backbone[backbone_layer_idx-1][0].graph_conv.register_forward_hook(collect_edge_info())
            else:
                edges_hook = graph_processing_layer.pvig_model.backbone[backbone_layer_idx-1][0].graph_conv.dilated_knn_graph._dilated.register_forward_hook(collect_edge_info())

            feature_encoding = graph_processing_layer(copy.deepcopy(image))

            edge_index = edge_index[0]

            graph_processing_layer.zero_grad()
            edges_hook.remove()

            ###node receptive field before the graph
            node_visualizer = PVIGNodeEncoder(vig_model, backbone_layer_idx-1).eval()
            node_positions_before_graph, _ = resolve_node_positions(copy.deepcopy(image), node_visualizer)
            node_visualizer.zero_grad()

            edge_index_t_list = edge_index.transpose().tolist()
            for edge in edge_index_t_list:
                    print("Found edge: {}".format(edge))
            num_cols = int(math.sqrt(node_positions_before_graph.shape[0]))

            visualize_graph_on_image("{}_{}_{}".format(idx, 57, 241),
                                     visualization_dir,
                                     copy.deepcopy(image),
                                     node_positions_before_graph,
                                     edge_index,
                                     draw_all_nodes=False,
                                     edges_to_filter=[57, 241])
            visualize_stem_overlap(graph_processing_layer, image, 57, 241, num_cols, visualization_dir)

            visualize_graph_on_image("{}_{}_{}".format(idx, 65, 831),
                                     visualization_dir,
                                     copy.deepcopy(image),
                                     node_positions_before_graph,
                                     edge_index,
                                     draw_all_nodes=False,
                                     edges_to_filter=[65, 831])
            visualize_stem_overlap(graph_processing_layer, image, 65, 831, num_cols, visualization_dir)


            # for edge_idx in edge_index_t_list:
            #     src = edge_idx[0]
            #     dst = edge_idx[1]
            #     if abs(dst - src) >= 10:
            #         visualize_graph_on_image("{}_{}_{}".format(idx, src, dst),
            #                                  visualization_dir,
            #                                  copy.deepcopy(image),
            #                                  node_positions_before_graph,
            #                                  edge_index,
            #                                  draw_all_nodes=False,
            #                                  edges_to_filter=[src, dst])
            #
            #         visualize_stem_overlap(graph_processing_layer, image, src, dst, num_cols, visualization_dir)




def visualize_graph_on_image(example_id, example_output_dir, image, node_positions, edge_index, edge_att_norm=None, draw_all_nodes=False, edges_to_filter=None, num_random_connections=0, percentile=95.0, proportions=False):
    node_colours = ["darkblue", "magenta"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5, 1.5))

    image_viz = image[0].cpu().detach().numpy()
    image_viz = np.transpose(image_viz, (1, 2, 0))
    ax.imshow(image_viz)
    g = nx.DiGraph()

    edge_colours = []
    edge_index_t_list = edge_index.transpose().tolist()
    nodes_to_include = set()
    alphas=None
    random_connections_added = 0
    if edge_att_norm is not None:
        edge_weights = edge_att_norm.flatten()

        percentile_value = np.percentile(edge_weights, percentile)
        percentile_10 = np.percentile(edge_weights, 20)
        percentile_50 = np.percentile(edge_weights, 50)
        percentile_95 = np.percentile(edge_weights, 98)

        num_percentile_10 = 0
        num_percentile_50 = 0
        num_percentile_95 = 0

        print("95th percentile value {}: ".format(percentile_value), flush=True)
        alphas = []
        for i in range(len(edge_index_t_list)):
            edge_weight = edge_weights[i]
            include_edge = False
            if proportions:
                if edge_weight >= percentile_95:
                    edge_weight_to_use = 1.0
                    num_percentile_95 = num_percentile_95 + 1
                    include_edge = True
                elif edge_weight <= percentile_10 and num_percentile_10 < 60:
                    edge_weight_to_use = 1.0
                    num_percentile_10 = num_percentile_10 + 1
                    include_edge = True
                if include_edge:
                    edge_idx = edge_index_t_list[i]
                    src = edge_idx[1]
                    dst = edge_idx[0]
                    g.add_edge(src, dst, alpha=edge_weight_to_use)
                    alphas.append(edge_weight_to_use)
                    nodes_to_include.add(src)
                    nodes_to_include.add(dst)
                    edge_colours.append("r")
            elif edge_weight >= percentile_value:
                edge_idx = edge_index_t_list[i]
                src = edge_idx[0]
                dst = edge_idx[1]
                g.add_edge(src, dst, alpha=edge_weight)
                alphas.append(edge_weight)
                nodes_to_include.add(src)
                nodes_to_include.add(dst)
                edge_colours.append("r")
        print("Example id {} - average alpha {} ".format(example_id, np.array(alphas).mean()))
    else:
        for i in range(len(edge_index_t_list)):
            edge_idx = edge_index_t_list[i]
            src = edge_idx[0]
            dst = edge_idx[1]
            keep_edge = False
            if edges_to_filter is not None and (dst in edges_to_filter):
                keep_edge = True
                edge_colours.append(node_colours[edges_to_filter.index(dst)])
            else:
                if num_random_connections > 0 and random_connections_added < num_random_connections:
                    if abs(dst - src) >= 5:
                        keep_edge = True
                        print(dst, src)
                        random_connections_added = random_connections_added + 1

            if keep_edge:
                g.add_edge(dst,src,alpha=1.0)
                nodes_to_include.add(dst)
                nodes_to_include.add(src)
                edge_colours.append("r")

    nodes_to_include = sorted(list(nodes_to_include))
    pos = {}
    for i in range(node_positions.shape[0]):
        if i in nodes_to_include or draw_all_nodes:
            node_pos = node_positions[i]
            upper_left_pos = node_pos[0]
            bottom_right_pos = node_pos[1]
            y_dist = bottom_right_pos[0] - upper_left_pos[0]
            x_dist = bottom_right_pos[1] - upper_left_pos[1]

            mid_pos_y = upper_left_pos[0] + (y_dist) // 2
            mid_pos_x = upper_left_pos[1] + (x_dist) // 2

            pos[i] = [mid_pos_x, mid_pos_y]
            # print("Mid pos: {}".format(pos[i]))
            g.add_node(i)
            node_patch = patches.Rectangle((upper_left_pos[1], upper_left_pos[0]), x_dist, y_dist, linewidth=0.8, edgecolor='w', facecolor='none')
            ax.add_patch(node_patch)

    node_colours = ["yellow" for i in range(len(nodes_to_include))]
    node_sizes = [5 for i in range(len(nodes_to_include))]
    if edges_to_filter is not None:
        for node in edges_to_filter:
            print(node)
            node_idx = nodes_to_include.index(node)
            node_colours[node_idx] = "blue"
            node_sizes[node_idx] = 15
    nx.draw_networkx_nodes(g, pos, nodelist=nodes_to_include,ax=ax, node_size=node_sizes, node_color=node_colours, margins=0.0)
    if draw_all_nodes:
        alphas = [0 for i in range(len(alphas))]
    nx.draw_networkx_edges(g, pos, edge_color=edge_colours, width=0.8, node_size=node_sizes, connectionstyle="arc3,rad=0.1", alpha=alphas)

    fig.tight_layout()
    fig_name = "graph_{}.png".format(example_id)
    if draw_all_nodes:
        fig_name = "graph_only_nodes_{}.png".format(example_id)
    plt.savefig(os.path.join(example_output_dir, fig_name), dpi=200, bbox_inches='tight', transparent=True)
    plt.close()

def redundancy_loss(h_emb, eps=1e-8):
    a_n = h_emb.norm(dim=2).unsqueeze(2)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
    loss_cos = sim_matrix.mean()

    return loss_cos

def visualize_gsat_edge_importance():

    print("Visualizing edge attention")
    output_attr_dir = os.path.join(model_root_dir, "gsat_analysis")
    if not os.path.exists(output_attr_dir):
        os.makedirs(output_attr_dir)

    node_encoder = PVIGNodeEncoder(vig_model)
    print(vig_model.visual_encoder.print())
    node_positions, nodes_receptive_fields_in_px = resolve_node_positions(torch.randn((1, 3, 256, 256)).cuda(), node_encoder)
    node_receptive_fields_in_px_df = pd.DataFrame({"node_id": [i for i in range(len(nodes_receptive_fields_in_px))],
                                                   "range": nodes_receptive_fields_in_px})
    node_receptive_fields_in_px_df.to_csv(os.path.join(output_attr_dir, "node_repe.csv"))

    edge_index = {}
    node_embeddings = {}
    def collect_edge_info():
        def hook(model, input, output):
            edge_index[0] = input[0].edge_index.cpu().detach().numpy()
            node_embeddings[0] = output[1].cpu().detach().numpy()
        return hook

    edges_hook = vig_model.gnn_bottleneck.gnn_model.register_forward_hook(collect_edge_info())

    total_example_idx = 0
    edge_ranges = []
    mean_edge_weight = []
    std_edge_weight = []
    pre_graph_emb_redundancy = []
    graph_processing_emb_redundancy = []
    for data in test_data_loader:
        vig_model.zero_grad()

        image = data["image"].cuda()
        image.requires_grad = True
        label = data["label"].cuda().cpu().detach().numpy()[0].item()

        example_output_dir = output_attr_dir
        if num_classes > 1:
            example_output_dir = os.path.join(output_attr_dir, str(label))

        if not os.path.exists(example_output_dir):
            os.makedirs(example_output_dir)

        if total_example_idx == 856:
            visualize_stem_overlap(node_encoder, image, 20 , 52, 8, example_output_dir)

        result = vig_model(image)
        edge_att = result["edge_att"]
        prediction = result["prediction"]
        node_embeddings_pre_graph = result["node_embeddings"]
        node_embeddings_graph_processing = result["node_embeddings_graph_processing"]
        if node_embeddings_graph_processing is not None:
            within_emb_redundancy = redundancy_loss(node_embeddings_graph_processing)
            graph_processing_emb_redundancy.append(within_emb_redundancy.item())
            print("Within emb redundancy: {}".format(within_emb_redundancy.item()))
        
        if node_embeddings_pre_graph is not None:
            pregraph_redundancy = redundancy_loss(node_embeddings_pre_graph)
            pre_graph_emb_redundancy.append(pregraph_redundancy.item())
            print("Within emb redundancy: {}".format(pregraph_redundancy.item()))

        edge_att_np = edge_att.cpu().detach().squeeze().numpy()
        # print("Shape of edge att: {}".format(edge_att_np.shape), flush=True)

        min_edge_value = np.min(edge_att_np)
        max_edge_value = np.max(edge_att_np)
        mean_edge_weight.append(np.mean(edge_att_np))
        std_edge_weight.append(np.std(edge_att_np))
        edge_ranges.append(max_edge_value - min_edge_value)

        fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0))
        ax.hist(edge_att_np)
        #ax.set_xlabel('Edge Weight', fontsize=10)
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.xticks(fontsize=14)
        #ax.set_ylabel("Number of Edges ")
        fig.tight_layout()
        plt.savefig(os.path.join(example_output_dir, "edgeAttention_{}.png".format(total_example_idx)), dpi=200, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.0))
        ax.set_xlim(-0.15, 1.15) 
        sns.kdeplot(x=edge_att_np, ax=ax)
        #ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)

        ax.set_xlabel('')
        ax.set_ylabel('')
        fig.tight_layout()

        plt.savefig(os.path.join(example_output_dir, "kde_edgeAttention_{}.png".format(total_example_idx)), dpi=200, bbox_inches='tight')
        plt.close()

        edge_att_norm = edge_att_np ** 10
        edge_att_norm = (edge_att_norm - edge_att_norm.min()) / (edge_att_norm.max() - edge_att_norm.min() + 1e-6)
        # histogram of the edge attributions
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(edge_att_norm)
        ax.set_xlabel('Normalized Edge Attention', fontsize=10)
        ax.set_ylabel("Frequency")
        ax.set_title("Target = {}".format(label))
        plt.savefig(os.path.join(example_output_dir, "normalized_edgeatt_{}.png".format(total_example_idx)))
        plt.close()
        
        
        visualize_graph_on_image("{}".format(total_example_idx), example_output_dir, image, node_positions, edge_index[0], edge_att_np)
        #visualize_graph_on_image(total_example_idx, example_output_dir, image, node_positions, edge_index[0], edge_att_np, draw_all_nodes=True, percentile=0.0)
        #visualize_graph_on_image("{}_proportions".format(total_example_idx), example_output_dir, image, node_positions, edge_index[0], edge_att_np, proportions=True)
        

        if total_example_idx == 856:
            visualize_graph_on_image("{}_rand_connections".format(total_example_idx), example_output_dir, image, node_positions, edge_index[0], edge_att_norm=None, num_random_connections=15)

        total_example_idx = total_example_idx + 1

    gsat_metrics = pd.DataFrame({"example_id": [i for i in range(0, total_example_idx)],
                                 "weights_mean": mean_edge_weight,
                                 "weights_std": std_edge_weight,
                                 "range": edge_ranges,
                                 "pre_graph_emb_redundancy": np.array(pre_graph_emb_redundancy),
                                 "emb_layer_redundancy": np.array(graph_processing_emb_redundancy)})
    gsat_metrics.to_csv(os.path.join(output_attr_dir, "gsat_metrics.csv"))


def visualize_pgeexplainer_importance():

    output_attr_dir = os.path.join(model_root_dir, "pgeexplainer_analysis")
    if not os.path.exists(output_attr_dir):
        os.makedirs(output_attr_dir)

    explanation_model = copy.deepcopy(vig_model).float().eval()
    vig_model_gnn_part = GNNExplainerModel(explanation_model.gnn_bottleneck).float().cuda().eval()

    mode = "regression" if task == "regression" else "multiclass_classification"
    return_type = "raw"

    pge_explainer = PGExplainer(epochs=50, lr=0.01, coeff_size=0.0001, coeff_ent=0.0001).float().cuda()
    
    pge_explainer = Explainer(
        model=vig_model_gnn_part,
        algorithm=pge_explainer,
        explanation_type="phenomenon",
        edge_mask_type='object',
        model_config=dict(
            mode=mode,
            task_level='graph',
            return_type=return_type))

    node_encoder = PVIGNodeEncoder(vig_model)
    node_positions, nodes_receptive_fields_in_px = resolve_node_positions(torch.randn((1, 3, 256, 256)).cuda(), node_encoder)
    node_receptive_fields_in_px_df = pd.DataFrame({"node_id": [i for i in range(len(nodes_receptive_fields_in_px))],
                                                   "range": nodes_receptive_fields_in_px})
    node_receptive_fields_in_px_df.to_csv(os.path.join(output_attr_dir, "node_repe.csv"))

    for epoch in range(50):
        epoch_losses = []
        for data in test_data_loader:
            explanation_model.zero_grad()

            image = data["image"].cuda()
            image.requires_grad = True
            label = data["label"].cuda()

            graph, patch_predictions = explanation_model.image_to_graph(image)
            x = graph.x.detach().clone().float().requires_grad_(True)
            loss = pge_explainer.algorithm.train(epoch, vig_model_gnn_part, x, graph.edge_index, batch=graph.batch, target=label)
            epoch_losses.append(loss)
        avg_loss = np.mean(epoch_losses)
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:02d}, Loss: {avg_loss:.4f}')

    total_example_idx = 0
    edge_ranges = []
    mean_edge_weight = []
    std_edge_weight = []
    with torch.no_grad():
        for data in test_data_loader:
            images = data["image"].cuda()
            labels = data["label"].cuda()

            for img_idx in range(images.shape[0]):
                image = images[img_idx:img_idx+1]  # Keep batch dimension
                label = labels[img_idx].item()

                graph, patch_predictions = explanation_model.image_to_graph(image)
                explanation = pge_explainer(graph.x.double(), graph.edge_index, batch=graph.batch, target=label)
                
                edge_att = explanation.edge_mask
                print(edge_att.shape)
                edge_index = graph.edge_index.cpu().detach().numpy()

                example_output_dir = output_attr_dir
                if num_classes > 1:
                    example_output_dir = os.path.join(output_attr_dir, str(label))
                
                if not os.path.exists(example_output_dir):
                    os.makedirs(example_output_dir)
        
                edge_att_np = edge_att.cpu().detach().squeeze().numpy()
                print("Saving edge att at {}".format(os.path.join(example_output_dir, "gnnexplainer_edge_att_{}.npy".format(total_example_idx))))
                np.save(os.path.join(example_output_dir, "gnnexplainer_edge_att_{}.npy".format(total_example_idx)), edge_att_np) # save

                min_edge_value = np.min(edge_att_np)
                max_edge_value = np.max(edge_att_np)
                print("Max edge value: {}, min edge value: {}".format(max_edge_value, min_edge_value))
                mean_edge_weight.append(np.mean(edge_att_np))
                std_edge_weight.append(np.std(edge_att_np))
                edge_ranges.append(max_edge_value - min_edge_value)

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.hist(edge_att_np)
                ax.set_xlabel('Edge Weight', fontsize=10)
                #ax.set_ylabel("Number of Edges ")
                plt.savefig(os.path.join(example_output_dir, "edgeAttention_{}.png".format(total_example_idx)))
                plt.close()

                edge_att_norm = edge_att_np ** 10
                edge_att_norm = (edge_att_norm - edge_att_norm.min()) / (edge_att_norm.max() - edge_att_norm.min() + 1e-6)
                # histogram of the edge attributions
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.hist(edge_att_norm)
                ax.set_xlabel('Normalized Edge Attention', fontsize=10)
                ax.set_ylabel("Frequency")
                ax.set_title("Target = {}".format(label))
                plt.savefig(os.path.join(example_output_dir, "normalized_edgeatt_{}.png".format(total_example_idx)))
                plt.close()

                visualize_graph_on_image(total_example_idx, example_output_dir, image, node_positions, edge_index, edge_att_np)
                
                total_example_idx = total_example_idx + 1

    gsat_metrics = pd.DataFrame({"example_id": [i for i in range(0, total_example_idx)],
                                 "weights_mean": mean_edge_weight,
                                 "weights_std": std_edge_weight,
                                 "range": edge_ranges})
    gsat_metrics.to_csv(os.path.join(output_attr_dir, "metrics.csv"))


def visualize_gnnexplainer_importance():

    output_attr_dir = os.path.join(model_root_dir, "gnnexplainer_analysis")
    if not os.path.exists(output_attr_dir):
        os.makedirs(output_attr_dir)

    explanation_model = copy.deepcopy(vig_model).eval()
    vig_model_gnn_part = GNNExplainerModel(explanation_model.gnn_bottleneck).eval()

    mode = "regression" if task == "regression" else "multiclass_classification"
    return_type = "raw" if task == "regression" else "log_probs"
    
    gnn_explainer = Explainer(
        model=vig_model_gnn_part,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode=mode,
            task_level='graph',
            return_type=return_type))

    node_encoder = PVIGNodeEncoder(vig_model)
    node_positions, nodes_receptive_fields_in_px = resolve_node_positions(torch.randn((1, 3, 256, 256)).cuda(), node_encoder)
    node_receptive_fields_in_px_df = pd.DataFrame({"node_id": [i for i in range(len(nodes_receptive_fields_in_px))],
                                                   "range": nodes_receptive_fields_in_px})
    node_receptive_fields_in_px_df.to_csv(os.path.join(output_attr_dir, "node_repe.csv"))

    total_example_idx = 0
    edge_ranges = []
    mean_edge_weight = []
    std_edge_weight = []
    for data in test_data_loader:
        explanation_model.zero_grad()

        image = data["image"].cuda()
        image.requires_grad = True
        label = data["label"].cuda().cpu().detach().numpy()[0].item()

        graph, patch_predictions = explanation_model.image_to_graph(image)
        x = graph.x.detach().clone().requires_grad_(True)

        example_output_dir = output_attr_dir
        if num_classes > 1:
            example_output_dir = os.path.join(output_attr_dir, str(label))
            explanation = gnn_explainer(x, graph.edge_index, batch=graph.batch)
        else:
            explanation = gnn_explainer(x, graph.edge_index, batch=graph.batch)
        
        #visualize_stem_overlap(node_encoder, data["image"].cuda(), 4 , 20, 8, example_output_dir)


        print(f'Generated explanations in {explanation.available_explanations}')

        if not os.path.exists(example_output_dir):
            os.makedirs(example_output_dir)

        edge_index = graph.edge_index.cpu().detach().numpy()
        edge_att = explanation.edge_mask
        print(edge_att)
        print(edge_index)
        edge_att_np = edge_att.cpu().detach().squeeze().numpy()
        print("Saving edge att at {}".format(os.path.join(example_output_dir, "gnnexplainer_edge_att_{}.npy".format(total_example_idx))))
        np.save(os.path.join(example_output_dir, "gnnexplainer_edge_att_{}.npy".format(total_example_idx)), edge_att_np) # save
        # print("Shape of edge att: {}".format(edge_att_np.shape), flush=True)

        min_edge_value = np.min(edge_att_np)
        max_edge_value = np.max(edge_att_np)
        mean_edge_weight.append(np.mean(edge_att_np))
        std_edge_weight.append(np.std(edge_att_np))
        edge_ranges.append(max_edge_value - min_edge_value)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(edge_att_np)
        ax.set_xlabel('Edge Weight', fontsize=10)
        #ax.set_ylabel("Number of Edges ")
        plt.savefig(os.path.join(example_output_dir, "edgeAttention_{}.png".format(total_example_idx)))
        plt.close()

        edge_att_norm = edge_att_np ** 10
        edge_att_norm = (edge_att_norm - edge_att_norm.min()) / (edge_att_norm.max() - edge_att_norm.min() + 1e-6)
        # histogram of the edge attributions
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(edge_att_norm)
        ax.set_xlabel('Normalized Edge Attention', fontsize=10)
        ax.set_ylabel("Frequency")
        ax.set_title("Target = {}".format(label))
        plt.savefig(os.path.join(example_output_dir, "normalized_edgeatt_{}.png".format(total_example_idx)))
        plt.close()

        visualize_graph_on_image(total_example_idx, example_output_dir, image, node_positions, edge_index, edge_att_np)
        
        total_example_idx = total_example_idx + 1

    gsat_metrics = pd.DataFrame({"example_id": [i for i in range(0, total_example_idx)],
                                 "weights_mean": mean_edge_weight,
                                 "weights_std": std_edge_weight,
                                 "range": edge_ranges})
    gsat_metrics.to_csv(os.path.join(output_attr_dir, "metrics.csv"))


def model_quantiative_eval(explanation_algorithm, perc_to_add=.05, metric="insertion", edge_order="descending"):
    output_attr_dir = os.path.join(model_root_dir, "quantative_analysis", explanation_algorithm)
    if not os.path.exists(output_attr_dir):
        os.makedirs(output_attr_dir)
    
    vig_model.gnn_bottleneck.gnn_model.add_edges = True
    vig_model.gnn_bottleneck.gnn_model.perc_to_add = perc_to_add
    vig_model.gnn_bottleneck.gnn_model.descending = edge_order == "descending"
    vig_model.gnn_bottleneck.gnn_model.insertion = metric == "insertion"

    print("{}: {}".format(metric, vig_model.gnn_bottleneck.gnn_model.add_edges))
    labels = []
    predictions = []
    losses = []
    for i, data in enumerate(test_data_loader):
        vig_model.zero_grad()
        image = data["image"].cuda()
        label = data["label"].cuda()
        if explanation_algorithm == "gnn_explainer":
            label_str=""
            if num_classes > 1:
                label_str = str(label.cuda().cpu().detach().numpy()[0].item())
            edge_att_explanation = np.load(os.path.join(model_root_dir, "gnnexplainer_analysis", label_str, "gnnexplainer_edge_att_{}.npy".format(i)))
            edge_att_explanation = torch.from_numpy(edge_att_explanation).cuda()
            vig_model.gnn_bottleneck.gnn_model.explanation_edge_att = edge_att_explanation
        
        result = vig_model(image)
        edge_att = result["edge_att"]
        logits = result["prediction"]
        logits = logits.double()
        if task == "classification":
            loss_instance = nn.functional.cross_entropy(logits, label, label_smoothing=0.1).cpu().detach().numpy().tolist()
            prediction = logits.argmax(dim=-1).cpu().detach().numpy().tolist()
        else:
            prediction = logits.flatten().cpu().detach().numpy().tolist()
            loss_instance = nn.functional.mse_loss(logits, label).cpu().detach().numpy().tolist()

        #loss_instance.backward()
        #print(edge_att_orig.grad)
        label_orig = label.cpu().detach().numpy().tolist()
        labels.extend(label_orig)
        predictions.extend(prediction)
        losses.append(loss_instance)

    average_loss = np.mean(np.array(losses))
    print("Loss: {}".format(average_loss))

    if task == "classification":
        goodness_of_fit = accuracy_score(np.array(labels), np.array(predictions))
        print("Accuracy : {}".format(goodness_of_fit))
    else:
        goodness_of_fit = r2_score(np.array(labels), np.array(predictions))
        print("R2 score : {}".format(goodness_of_fit))

    results_df = pd.DataFrame({"label": labels, "prediction": predictions, "loss": losses})
    results_filename = "predictions_{}_{}_perc_edges_{}.csv".format(metric, perc_to_add, edge_order)
    results_df.to_csv(os.path.join(output_attr_dir, results_filename))

    return goodness_of_fit, average_loss

def quantiative_eval(explanation_algorithm, metric, edge_order):
    percentages = []
    accuracies = []
    losses = []
    for perc in [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]:
        accuracy_perc_edges, loss_perc_edges = model_quantiative_eval(explanation_algorithm, perc_to_add=perc, metric=metric, edge_order=edge_order)
        percentages.append(perc)
        accuracies.append(accuracy_perc_edges)
        losses.append(loss_perc_edges)

    # Min-max normalization of accuracies
    accuracies = np.array(accuracies)
    accuracies = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min())

    results_df = pd.DataFrame({"{}".format(metric): percentages, "performance": accuracies, "loss": losses})
    results_df.to_csv(os.path.join(model_root_dir, "quantative_analysis", explanation_algorithm, "results_{}_{}.csv".format(metric,edge_order)))

    auc_score = auc(percentages, accuracies)
    print("AUC score: {}".format(auc_score))
    results_df = pd.DataFrame([{"id":0,"AUC": auc_score}])
    results_df.to_csv(os.path.join(model_root_dir, "quantative_analysis", explanation_algorithm, "auc_{}_{}.csv".format(metric,edge_order)))







if __name__ == '__main__':


    vig_stem_timestamp = "2024-10-18_14.30.03"
    vig_stem_checkpoint = "vig_resisc45-epoch=194-val_accuracy=0.95.ckpt"

    bagnet_timestamp = "2024-11-02_18.35.08"
    bagnet_checkpoint = "vig_resisc45-epoch=496-val_accuracy=0.92.ckpt"

    pyramid_vig_timestamp = "2024-10-29_10.22.57"
    pyramid_vig_checkpoint = "vig_resisc45-epoch=198-val_accuracy=0.94.ckpt"

    #liveability model
    wignn_timestamp = "2025-11-02_18.11.15"
    wignn_checkpoint = "vig_resisc45-epoch=449-val_accuracy=0.93.ckpt"

    # resisc45 model
    # wignn_timestamp = "2025-01-28_19.27.45"
    # wignn_checkpoint = "vig_resisc45-epoch=499-val_accuracy=0.92.ckpt"

    ##### iWiViG models final ####

    resisc45_iwivig_timestamp = "2025-11-17_18.27.34"
    resisc45_iwivig_checkpoint = "vig_resisc45-epoch=359-val_accuracy=0.94.ckpt"

    sun397_iwivig_timestamp = "2025-11-02_18.23.21"
    sun397_iwivig_checkpoint = "vig_sun397-epoch=299-val_accuracy=0.42.ckpt"

    liveability_iwivig_timestamp = "2025-11-03_19.55.57"
    liveability_iwivig_checkpoint = "vig_Liveability-epoch=239-val_R2_entire_set=0.64.ckpt"


    ##### iWiViG models without attention ####

    # resisc45_iwivig_timestamp = "2026-03-02_15.43.56"
    # resisc45_iwivig_checkpoint = "vig_resisc45-epoch=239-val_accuracy=0.93.ckpt"

    # sun397_iwivig_timestamp = "2026-03-02_15.42.29"
    # sun397_iwivig_checkpoint = "vig_sun397-epoch=209-val_accuracy=0.39.ckpt"

    # liveability_iwivig_timestamp = "2026-03-02_15.43.55"
    # liveability_iwivig_checkpoint = "vig_Liveability-epoch=239-val_R2_entire_set=0.68.ckpt"

    iwivig_timestamp = resisc45_iwivig_timestamp
    iwivig_checkpoint = resisc45_iwivig_checkpoint

    r = 0.5
    graph_bottleneck_layers = 3

    stem_approach = "pyramid_vig_stem"
    gnn_approach = "iWiViG"

    hidden_channels = 48 if gnn_approach != "vig" else 192
    encoder_backbone_wig_blocks=2
    encoder_window_size = 4
    encoder_downsample_wo_overlap=True
    use_patch_predictions=False
    if gnn_approach == "pvig":
        backbone_layer_idx = 4
        timestamp = pyramid_vig_timestamp
        checkpoint_path = pyramid_vig_checkpoint
    elif gnn_approach == "vig":
        backbone_layer_idx = 1
        timestamp = vig_stem_timestamp
        checkpoint_path = vig_stem_checkpoint
    else:
        backbone_layer_idx = 4
        timestamp = iwivig_timestamp
        checkpoint_path = iwivig_checkpoint
        encoder_model = "WIGNN"

    #dataset = "sun397"
    dataset = "resisc45"
    task = "classification"
    #task = "regression"

    num_classes, image_size,train_data_loader, val_data_loader, test_data_loader = get_resisc45_data_loaders(dataset, batch_size=1)
    vig_model, model_label = init_model(stem_approach,
                                        hidden_channels,
                                        gnn_approach,
                                        image_size,
                                        num_classes,
                                        encoder_model,
                                        encoder_downsample_wo_overlap,
                                        encoder_backbone_wig_blocks,
                                        encoder_window_size,
                                        graph_bottleneck_layers=graph_bottleneck_layers,
                                        gsat_r=r,
                                        lambda_gsat_loss=1.0,
                                        use_patch_predictions=use_patch_predictions,
                                        learn_edge_att=True)
    
    
    print("Model label: {}".format(model_label))

    lambda_graph_redundancy_loss = .0
    if lambda_graph_redundancy_loss > 0:
        model_label = os.path.join(model_label, "w_graph_redundancy_loss")

    sparsity_budget = 0
    lambda_gsat_weights_variance_loss = 0
    if lambda_gsat_weights_variance_loss > 0:
        model_label = os.path.join(model_label, "w_gsat_weights_variance_loss_{}".format(lambda_gsat_weights_variance_loss))
        if sparsity_budget is not None:
            model_label = os.path.join(model_label, "sparsity_{}".format(sparsity_budget))

    if use_patch_predictions:
        model_label = os.path.join(model_label, "with_patch_predictions")

    model_root_dir = os.path.join("/home/results/graph_image_understanding/{}/{}/{}".format(dataset, model_label, timestamp))
    vig_model = load_trained_model(vig_model, model_root_dir, checkpoint_path)

    
    visualize_gsat_edge_importance()
    quantiative_eval("gsat", "deletion", "descending")
    quantiative_eval("gsat", "insertion", "descending")

    
    visualize_gnnexplainer_importance()
    quantiative_eval("gnn_explainer", "descending")
    quantiative_eval("gnn_explainer","ascending")
