import os.path

import captum.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import copy

from PIL import Image, ImageDraw

from util import *
from graph_inference import GSATViGReasoning
from models.gnn_models import init_model
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution, Occlusion
from captum.attr import visualization as viz
import torch.nn.functional as F

from quantus import Selectivity, Sparseness
from quantus.functions.normalise_func import normalise_by_max
from captum.metrics import infidelity, sensitivity_max
from models.stem_approaches import get_receptive_field, resolve_stem_module

def visualize_stem_overlap(stem_model, image, src, dst, num_cols, output_dir):
    def get_row_col_idx(node, num_cols):
        return (node // num_cols, node % num_cols)

    node1 = get_row_col_idx(src, num_cols)
    node2 = get_row_col_idx(dst, num_cols)

    light_coral_rgb = np.array([0, 0, 139, 128]).astype(np.uint8)
    aquamarine_rgb = np.array([255, 0, 255, 128]).astype(np.uint8)
    overlap_colour = np.array([211,211,255, 128]).astype(np.uint8)

    node1_receptive_field = get_receptive_field(image, stem_model, node1[0], node1[1])
    node2_receptive_field = get_receptive_field(image, stem_model, node2[0], node2[1])

    intersection_pixels = list(set(node1_receptive_field) & set(node2_receptive_field))
    print("Number of intersecting pixels: {}".format(len(intersection_pixels)))

    image_viz = image[0].cpu().detach().numpy() * 255
    image_viz = image_viz.transpose((1, 2, 0))
    image_viz = image_viz.astype(np.uint8)



    alpha_channel = np.ones((image_viz.shape[0], image_viz.shape[1]), dtype=np.uint8) * 255
    image_viz = np.dstack((image_viz, alpha_channel))


    for pixel in node1_receptive_field:
        if pixel in intersection_pixels:
            alpha = overlap_colour[3] / 255.0
            image_viz[pixel[0], pixel[1], :3] = (
                    alpha * overlap_colour[:3] + (1 - alpha) * image_viz[pixel[0], pixel[1], :3]
            )
            image_viz[pixel[0], pixel[1], 3] = 255  # Full opacity for the blended pixel
        else:
            alpha = light_coral_rgb[3] / 255.0
            image_viz[pixel[0], pixel[1], :3] = (
                    alpha * light_coral_rgb[:3] + (1 - alpha) * image_viz[pixel[0], pixel[1], :3]
            )
            image_viz[pixel[0], pixel[1], 3] = 255

    for pixel in node2_receptive_field:
        if pixel not in intersection_pixels:
            alpha = aquamarine_rgb[3] / 255.0
            image_viz[pixel[0], pixel[1], :3] = (
                    alpha * aquamarine_rgb[:3] + (1 - alpha) * image_viz[pixel[0], pixel[1], :3]
            )
            image_viz[pixel[0], pixel[1], 3] = 255

    image_pil = Image.fromarray(image_viz)
    draw = ImageDraw.Draw(image_pil)

    node1_rf_array = np.array(node1_receptive_field)
    node1_center = (int(node1_rf_array[:, 1].mean()), int(node1_rf_array[:, 0].mean()))
    
    node2_rf_array = np.array(node2_receptive_field)
    node2_center = (int(node2_rf_array[:, 1].mean()), int(node2_rf_array[:, 0].mean()))
    
    # Draw yellow circles at the centers
    circle_radius = 6
    yellow = (255, 255, 0)  # Yellow in RGB
    
    # Circle for node1
    draw.ellipse(
        [node1_center[0] - circle_radius, node1_center[1] - circle_radius,
         node1_center[0] + circle_radius, node1_center[1] + circle_radius],
        fill=yellow,
        outline=yellow
    )
    
    # Circle for node2
    draw.ellipse(
        [node2_center[0] - circle_radius, node2_center[1] - circle_radius,
         node2_center[0] + circle_radius, node2_center[1] + circle_radius],
        fill=yellow,
        outline=yellow
    )
    
    # Convert back to numpy array
    image_viz = np.array(image_pil)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    imgplot = ax.imshow(image_viz)
    xax = ax.axes.get_xaxis()
    xax = xax.set_visible(False)

    yax = ax.axes.get_yaxis()
    yax = yax.set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "receptive_field_{}_{}.png".format(src, dst)), dpi=200, bbox_inches='tight', transparent=True)
    plt.close()

def calc_sparsity(abs_feature_importances):
    a = abs_feature_importances.flatten()
    print(a.shape)
    a += 0.0000001
    a = np.sort(a)
    score = (np.sum((2 * np.arange(1, a.shape[0] + 1) - a.shape[0] - 1) * a)) / (
            a.shape[0] * np.sum(a))
    return score

def pq_sparsity(x, dim, p=0.5, q=1.0):
    mask = torch.ones_like(x)
    d = mask.to(x.device).float().sum(dim=dim)
    # si = (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p) / \
    #      (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
    # si = d ** ((1 / q) - (1 / p)) * torch.linalg.norm(x, p, dim=dim) / torch.linalg.norm(x, q, dim=dim)
    # si = (torch.linalg.norm(x, q, dim=dim) -
    #       d ** ((1 / q) - (1 / p)) * torch.linalg.norm(x, p, dim=dim)) / torch.linalg.norm(x, q, dim=dim)
    x = x * mask.to(x.device).float()
    si = 1 - (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p) / \
         (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
    si[d == 0] = 0
    si[si == -float('inf')] = 0
    si[torch.logical_and(si > - 1e-5, si < 0)] = 0
    return si

def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().cuda()
    return noise, inputs - noise

def forward_fn(x):
    return gnn_model(x)[1]

def calculate_attributions(method, patch_size=20):

    output_attr_dir = os.path.join(model_root_dir, "{}_attribution".format(method))

    if method == "gradcam":
        attribution_method = LayerGradCam(forward_fn, gnn_model.pooling_layer)
    elif method == "occlusion":
        attribution_method = Occlusion(forward_fn)
        output_attr_dir = output_attr_dir + "_patch_size={}".format(patch_size)
    else:
        attribution_method = IntegratedGradients(forward_fn, multiply_by_inputs=False)

    pq_sparsities = []
    gini_sparsities = []
    saliencies = []
    infidelities = []

    total_example_idx = 0
    for data in test_data_loader:
        gnn_model.zero_grad()

        image = data["image"].cuda()
        image.requires_grad = True
        label = data["label"].cuda().cpu().detach().numpy()[0].item()

        target = label
        if num_classes == 1:
            target = None
            output_attr_dir_label = os.path.join(output_attr_dir, "liveability_attributions")
        else:
            output_attr_dir_label = os.path.join(output_attr_dir, "{}".format(label))

        os.makedirs(output_attr_dir_label, exist_ok=True)

        if method == "gradcam":
            feature_importance = attribution_method.attribute(image, target=target, attribute_to_layer_input=True)
            feature_importance = LayerAttribution.interpolate(feature_importance, image.shape[2:])
        elif method == "occlusion":
            feature_importance = attribution_method.attribute(image, target=target,
                                                              sliding_window_shapes=(3, patch_size, patch_size),
                                                              strides=patch_size)
        else:
            feature_importance = attribution_method.attribute(image, target=target)

        example_infidelity = infidelity(forward_fn, perturb_fn, image, feature_importance, target=target).detach().cpu().numpy().item()
        infidelities.append(example_infidelity)

        feature_importance_np = feature_importance[0].detach().cpu().numpy()
        print(feature_importance_np.shape)

        #sparsity
        sparsity_input = torch.abs(feature_importance[0]).sum(dim=0).flatten()
        example_pq_sparsity = pq_sparsity(sparsity_input, dim=0).detach().cpu().numpy().item()
        pq_sparsities.append(example_pq_sparsity)

        example_gini_sparsity = calc_sparsity(sparsity_input.detach().cpu().numpy())
        gini_sparsities.append(example_gini_sparsity)
        print("Example id: {}, pq sparsity: {}, gini: {}".format(total_example_idx, example_pq_sparsity, example_gini_sparsity))

        pixel_importance = np.abs(feature_importance_np).sum(axis=0).flatten()
        print("{}, : {}".format(total_example_idx, np.sum(pixel_importance)))

        test_image = image[0].cpu().detach().numpy()
        test_image_viz = np.transpose(test_image, (1, 2, 0))
        attr_for_test_image_viz = np.transpose(feature_importance_np, (1, 2, 0))

        # fig, ax = viz.visualize_image_attr(None, test_image_viz,
        #                       method="original_image", title="Original Image, class {}".format(label))
        # plt.savefig(os.path.join(output_attr_dir_label, "input_{}.png".format(total_example_idx)))
        # plt.close()

        fig_attr_viz, ax_attr_viz = viz.visualize_image_attr(attr_for_test_image_viz, test_image_viz, method="blended_heat_map",
                                                             sign="absolute_value", show_colorbar=True, fig_size=(6, 6), title="")
        plt.savefig(os.path.join(output_attr_dir_label, "attribution_{}.png".format(total_example_idx)),
                    bbox_inches='tight',pad_inches = 0, dpi=300)
        plt.close()

        # histogram of the saliencies
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(pixel_importance)
        ax.set_xlabel('Pixel saliency', fontsize=10)
        ax.set_ylabel("Frequency")
        ax.set_title("{} saliency".format(method))
        fig.tight_layout()
        plt.savefig(os.path.join(output_attr_dir_label, "saliency_{}.png".format(total_example_idx)), dpi=300)
        plt.close()

        saliencies.append(np.mean(pixel_importance))
        total_example_idx = total_example_idx + 1


    metrics_per_example = pd.DataFrame({"example_id":[i for i in range(total_example_idx)],
                                        "pq_sparsity": pq_sparsities,
                                        "gini_sparsity": gini_sparsities,
                                        "saliency": saliencies,
                                        "infidelity": infidelities})

    metrics_per_example.to_csv(os.path.join(output_attr_dir, "explanation_metrics.csv"))

if __name__ == '__main__':

    vig_stem_timestamp = "2024-10-18_14.53.05"
    vig_stem_checkpoint = "vig_Liveability-epoch=198-val_R2_entire_set=0.61.ckpt"

    bagnet_timestamp = "2024-11-02_18.35.08"
    bagnet_checkpoint = "vig_resisc45-epoch=496-val_accuracy=0.92.ckpt"

    pyramid_vig_timestamp = "2024-10-31_17.48.59"
    pyramid_vig_checkpoint = "vig_Liveability-epoch=175-val_R2_entire_set=0.66.ckpt"

    wignn_timestamp = "2024-11-14_08.10.59"
    wignn_checkpoint = "vig_Liveability-epoch=499-val_R2_entire_set=0.65.ckpt"

    resnet_timestamp = "2025-01-29_10.22.28"
    resnet_checkpoint = "vig_Liveability-epoch=179-val_R2_entire_set=0.64.ckpt"

    r = 0.7
    num_layers = 4

    grid_approach = "vig_stem"
    gnn_approach = "resnet"
    window_size = 4
    hidden_channels = 192 if gnn_approach != "pvig" else 48

    gsat_as_subgraph = False
    downsample_wo_overlap = False
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
        timestamp = resnet_timestamp
        checkpoint_path = resnet_checkpoint
        gsat_as_subgraph = True
        downsample_wo_overlap = True

    dataset = "Liveability"
    task = "regression"
    #task = "classification"

    num_classes, image_size, _, _, test_data_loader = get_liveability_data_loaders(dataset, batch_size=1)

    gnn_model, model_label = init_model(grid_approach, hidden_channels, gnn_approach, image_size, num_classes, backbone_wig_blocks=2,
                                        gsat_as_subgraph=gsat_as_subgraph,
                                        graph_bottleneck_layers=num_layers, gsat_r=r, window_size=window_size,
                                        downsample_wo_overlap=downsample_wo_overlap)

    print(gnn_model)
    model_root_dir = os.path.join(
        "/home/results/graph_image_understanding/{}/{}/{}".format(dataset, model_label, timestamp))
    gnn_model = load_trained_model(task, gnn_model, model_root_dir, checkpoint_path)
    calculate_attributions("ig")
    calculate_attributions("occlusion", patch_size=30)
    #calculate_attributions("gradcam")
