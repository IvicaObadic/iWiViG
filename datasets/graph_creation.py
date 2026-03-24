import argparse
import os
import cv2 as cv
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.neighbors import kneighbors_graph

from skimage.segmentation import mark_boundaries

import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torchvision.transforms as T
from torch_geometric.transforms import ToSLIC
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_networkx
from util import show_anns

import networkx as nx

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#ROOT_DIR = "C:/Users/datasets/FloodNet/"
class_names = ["Non-Flooded", "Flooded"]
knn_graph_transform = KNNGraph(k=5)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_dir', help='the root folder of the dataset', default="/home/datasets/FloodNet")
    parser.add_argument(
        '--encoding_method', type=str, default="SIFT", help='the encoding method from which to extract the graph from the image')
    parser.add_argument(
        '--num_SLIC_segments', type=int, default=500,
        help='number of segments to use as input to SLIC')

    args, _ = parser.parse_known_args()
    return args

def visualize_sift_descriptors(gray_image, image, img_id, kp, save_dir):
    # plot SIFT descriptors
    img = cv.drawKeypoints(gray_image, kp, image)
    img_with_keypoints_root_dir = os.path.join(save_dir, "keypoints_viz")
    if not os.path.exists(img_with_keypoints_root_dir):
        os.makedirs(img_with_keypoints_root_dir)
    cv.imwrite(os.path.join(img_with_keypoints_root_dir, "{}.png".format(img_id)), img)


def visualize_SLIC_superpixels(image, img_id, slic_graph, slic_images_and_graphs_dir):
    image_segments = torch.squeeze(slic_graph.seg).detach().numpy()
    image_with_slic_boundaries = mark_boundaries(image, image_segments)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.imshow(image_with_slic_boundaries)
    axs.axis("off")
    plt.savefig(os.path.join(slic_images_and_graphs_dir, img_id))
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    graph_viz = to_networkx(slic_graph)
    nx.draw(graph_viz, ax=axs)
    plt.savefig(os.path.join(slic_images_and_graphs_dir, "graph_{}".format(img_id)))
    plt.close()

def create_sift_graph(sift_descriptor, save_directory, image, image_id, label, visualize_descriptor=True):

    positions_similarity_save_dir = os.path.join(save_directory, "position_knn")
    if not os.path.exists(positions_similarity_save_dir):
        os.makedirs(positions_similarity_save_dir)

    embeddings_similarity_save_dir = os.path.join(save_directory, "embeddings_knn")
    if not os.path.exists(embeddings_similarity_save_dir):
        os.makedirs(embeddings_similarity_save_dir)


    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp, des = sift_descriptor.detectAndCompute(gray_image, None)
    if visualize_descriptor:
        visualize_sift_descriptors(gray_image, image, image_id, kp, save_directory)

    # create KNN graph based on keypoint locations
    kp_positions = cv.KeyPoint_convert(kp)
    graph_data_positions_similarity = Data(torch.from_numpy(des),
                                           pos=torch.from_numpy(kp_positions),
                                           y=label,
                                           id=image_id)
    graph_data_positions_similarity = knn_graph_transform(graph_data_positions_similarity)
    torch.save(graph_data_positions_similarity,
               os.path.join(positions_similarity_save_dir, f'graph_{image_id}.pt'))

    A = kneighbors_graph(des, 5, mode='connectivity')
    edge_index = from_scipy_sparse_matrix(A)[0]
    graph_data_embeddings_similarity = Data(
        torch.from_numpy(des),
        pos=torch.from_numpy(kp_positions),
        edge_index=edge_index,
        y=label,
        id=image_id)
    torch.save(graph_data_embeddings_similarity,
               os.path.join(embeddings_similarity_save_dir, f'graph_{image_id}.pt'))

def create_and_save_SIFT_graphs(image_folder,  flood_label, save_dir, num_keypoints=10000):
    print("Graph creation based on SIFT keypoints")
    save_dir = save_dir + "_{}_segments".format(num_keypoints)

    sift_descriptor = cv.SIFT_create(num_keypoints)
    images = os.listdir(image_folder)
    for i, img_id in enumerate(images):
        if i%100 == 0:
            print("Creating {}-th graph for image {}".format(i, img_id))
        image_path = os.path.join(image_folder, img_id)
        image = cv.imread(image_path)
        create_sift_graph(sift_descriptor, save_dir, image, img_id, flood_label)


def create_SLIC_graphs(image_folder,  flood_label, save_dir, n_segments=1000):
    print("Creating SLIC graphs")
    slic_transform = ToSLIC(n_segments=n_segments, add_seg=True, add_img=False)
    graph_transform = T.Compose([slic_transform, knn_graph_transform])

    save_dir = save_dir + "_{}_segments".format(n_segments)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    positions_similarity_save_dir = os.path.join(save_dir, "position_knn")
    if not os.path.exists(positions_similarity_save_dir):
        os.makedirs(positions_similarity_save_dir)

    embeddings_similarity_save_dir = os.path.join(save_dir, "embeddings_knn")
    if not os.path.exists(embeddings_similarity_save_dir):
        os.makedirs(embeddings_similarity_save_dir)

    slic_images_and_graphs_dir = os.path.join(save_dir, "slic_viz")
    if not os.path.exists(slic_images_and_graphs_dir):
        os.makedirs(slic_images_and_graphs_dir)

    for i, img_id in enumerate(os.listdir(image_folder)):
        if i % 100 == 0:
            print(i, img_id)

        image_path = os.path.join(image_folder, img_id)
        image = cv.imread(image_path)

        # permuting axis for compatibility with pyg SLIC implementation
        img_slic_input = torch.from_numpy(image).permute(2, 0, 1)
        slic_graph_position_similarity = graph_transform(img_slic_input)
        slic_graph_position_similarity.y = flood_label
        visualize_SLIC_superpixels(image, img_id, slic_graph_position_similarity, slic_images_and_graphs_dir)
        #delete graph segments due to large memory requirements
        del slic_graph_position_similarity.seg
        torch.save(slic_graph_position_similarity,
                   os.path.join(positions_similarity_save_dir, f'graph_{img_id.split(".")[0]}.pt'))

        #create second graph based on superpixel embedding similarity
        embeddings_np = slic_graph_position_similarity.x.cpu().detach().numpy()
        A = kneighbors_graph(embeddings_np, 5, mode='connectivity')
        edge_index = from_scipy_sparse_matrix(A)[0]
        slic_graph_embeddings_sim = Data(slic_graph_position_similarity.x, edge_index, y=flood_label)
        torch.save(slic_graph_embeddings_sim,
                   os.path.join(embeddings_similarity_save_dir, f'graph_{img_id.split(".")[0]}.pt'))


def create_SAG_graphs(image_folder, flood_label, save_dir):
    torch.cuda.empty_cache()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sam = sam_model_registry["default"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
    if torch.cuda.is_available():
        sam = sam.cuda()
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=32)

    for i, img_id in enumerate(os.listdir(image_folder)):
        print(i, img_id)

        image_path = os.path.join(image_folder, img_id)
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        masks_json_file = os.path.join(save_dir, img_id.split(".")[0] + ".json")
        segmentation_masks = json.dumps(masks, cls=NumpyEncoder)
        with open(masks_json_file, "w") as outfile:
            outfile.write(segmentation_masks)
        del segmentation_masks

        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, img_id))
        plt.close()


def create_graphs(args):

    root_dir = args.root_dir
    encoding_method = args.encoding_method
    images_root_path = os.path.join(root_dir, "data/Train/Labeled")

    for i, class_name in enumerate(class_names):
        images_folder = os.path.join(images_root_path, class_name, "image")
        graphs_dir = os.path.join(root_dir, encoding_method, class_name)
        if encoding_method == "SIFT":
            create_and_save_SIFT_graphs(images_folder, i, graphs_dir, num_keypoints=args.num_SLIC_segments)
        elif encoding_method == "SLIC":
            create_SLIC_graphs(images_folder, i, graphs_dir, n_segments=args.num_SLIC_segments)

        elif encoding_method == "SAM":
            create_SAG_graphs(images_folder, i, graphs_dir)



if __name__ == '__main__':

    args = parse_args()
    create_graphs(args)