import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc



def prepare_boxplot_data(xai_res_paths):
    xai_metrics_results_dfs = []
    for xai_res_path, model in xai_res_paths:
        xai_method_result_paths = [xai_method_path for xai_method_path in os.listdir(xai_res_path) if "attribution" in xai_method_path]
        for xai_method_path in xai_method_result_paths:
            method_name = xai_method_path.split("_")[0]
            method_name = method_name.capitalize()
            if method_name == "Ig":
                method_name = "IG"
            results_file = os.path.join(xai_res_path, xai_method_path, "explanation_metrics.csv")
            if os.path.exists(results_file):
                xai_method_metrics_df = pd.read_csv(results_file, index_col=0)
                xai_method_metrics_melted = xai_method_metrics_df.melt(id_vars='example_id', var_name='Metric', value_name='Value')
                # xai_method_metrics_melted = xai_method_metrics_melted[~xai_method_metrics_melted["Metric"].isin(["saliency", "gini_sparsity"])]
                xai_method_metrics_melted["Model"] = model
                xai_method_metrics_melted["Method"] = method_name
                # Melt the DataFrame to create the desired format
                xai_metrics_results_dfs.append(xai_method_metrics_melted)

    result = pd.concat(xai_metrics_results_dfs, ignore_index=True)
    return result


def summarize_results(xai_result_paths, dataset_name):
    xai_metrics = prepare_boxplot_data(xai_result_paths)
    print(xai_metrics)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    xai_metrics_sparsity = xai_metrics[xai_metrics["Metric"] == "pq_sparsity"]
    sns.boxplot(data=xai_metrics_sparsity, x="Model", y="Value", hue="Method", ax=ax)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Explanation Sparsity (\u2192)")
    fig.tight_layout()
    plt.savefig("{}_xai_metrics_sparsity.png".format(dataset_name), dpi=300)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    xai_metrics_infidelity = xai_metrics[xai_metrics["Metric"] == "infidelity"]
    print(xai_metrics_infidelity)
    # xai_metrics_infidelity["Value"] = np.log2(xai_metrics_infidelity["Value"])
    print(xai_metrics_infidelity["Value"].max(), xai_metrics_infidelity.min())
    sns.boxplot(data=xai_metrics_infidelity, x="Model", y="Value", hue="Method", ax=ax)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Explanation Infidelity (\u2190)")
    ax.set_yscale('log')
    fig.tight_layout()
    plt.savefig("{}_xai_metrics_infidelity.png".format(dataset_name), dpi=300)
    plt.close()


def get_auc_data(result_path, dataset_name):
    edge_addition_descending_path = os.path.join(result_path, "results_edges_added_descending.csv")
    edge_addition_ascending_path = os.path.join(result_path, "results_edges_added_ascending.csv")

    edge_addition_descending = pd.read_csv(edge_addition_descending_path, index_col=0)
    percentages_desc = edge_addition_descending[["edges_added"]].values.flatten()
    performance_desc = edge_addition_descending[["performance"]].values.flatten()
    # Min-max normalization
    performance_desc = (performance_desc - performance_desc.min()) / (performance_desc.max() - performance_desc.min())
    auc_descending = auc(percentages_desc, performance_desc)

    edge_addition_ascending = pd.read_csv(edge_addition_ascending_path, index_col=0)
    percentages_asc = edge_addition_ascending[["edges_added"]].values.flatten()
    performance_asc = edge_addition_ascending[["performance"]].values.flatten()
    # Min-max normalization
    performance_asc = (performance_asc - performance_asc.min()) / (performance_asc.max() - performance_asc.min())
    auc_ascending = auc(percentages_asc, performance_asc)

    return auc_descending - auc_ascending

def summarize_auc_results(xai_result_paths):

    auc_results = []
    for dataset_name, xai_approach, path in xai_result_paths:
        auc_diff = get_auc_data(path, dataset_name)
        auc_results.append({
            "Dataset": dataset_name,
            "xAI Approach": xai_approach,
            "AUC Difference": auc_diff
        })

    auc_results = pd.DataFrame(auc_results)

    print(auc_results)

    plt.figure(figsize=(3, 2.6))
    sns.barplot(data=auc_results, x="Dataset", y="AUC Difference", hue="xAI Approach", ci=None)
    plt.ylabel(r'Explanation $\Delta\ \mathrm{AUC}$' + ' (\u2192)')
    plt.xlabel("Dataset")
    #plt.legend(title="XAI Approach", loc="upper left", bbox_to_anchor=(0.98, 0.98), frameon=True, borderaxespad=0.)
    plt.tight_layout()
    out_fname = "auc_difference_barplot.png"
    plt.savefig(out_fname, dpi=300)
    plt.close()


def edge_quantitative_evaluation(result_path, dataset_name):
    insertion_metric_results_path = os.path.join(result_path, "results_insertion_descending.csv")
    deletion_metric_results_path = os.path.join(result_path, "results_deletion_descending.csv")

    insertion_results = pd.read_csv(insertion_metric_results_path, index_col=0)
    percentages_x = insertion_results[["insertion"]].values.flatten()
    performance_insertion = insertion_results[["performance"]].values.flatten()
    performance_insertion = (performance_insertion - performance_insertion.min()) / (performance_insertion.max() - performance_insertion.min())

    auc_insertion = auc(percentages_x, performance_insertion)
    print("AUC Descending: ", auc_insertion)
    insertion_results["Metric"] = "AUC: {:.2f}".format(auc_insertion)

    deletion_results = pd.read_csv(deletion_metric_results_path, index_col=0)
    percentages_x = deletion_results[["deletion"]].values.flatten()
    performance_deletion = deletion_results[["performance"]].values.flatten()
    performance_deletion = (performance_deletion - performance_deletion.min()) / (performance_deletion.max() - performance_deletion.min())
    auc_deletion = auc(percentages_x, performance_deletion)
    deletion_results["Metric"] = "AUC: {:.2f}".format(auc_deletion)
    min_performance_desc = insertion_results["performance"].min()
    min_performance_asc = deletion_results["performance"].min()
    shading_baseline = min(min_performance_desc, min_performance_asc) * 1.0

    ylabel = "Accuracy"
    if dataset_name == "Liveability":
        ylabel = "$R^2$ score"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.1, 1.8))
    
    color_desc = 'C0' # Blue
    color_asc = 'C1'  # Orange

    # Plot Descending curve and fill its AUC
    ax.plot(percentages_x, performance_deletion, label="Deletion", color=color_asc, linewidth=2)
    # ax.fill_between(percentages_x, performance_deletion, 0, color=color_asc, alpha=0.2)


# --- Insertion (Blue) ---
    ax.plot(percentages_x, performance_insertion, label="Insertion", color=color_desc, linewidth=2)
    # ax.fill_between(percentages_x, performance_insertion, 0, color=color_desc, alpha=0.2)

    # ax.text(0.43, 0.15, f"AUC: {auc_deletion:.2f}", 
    #     color=color_asc, fontsize=6, fontweight='bold', 
    #     transform=ax.transAxes, 
    #     horizontalalignment='right') # Keeps text from going right

# --- Insertion AUC (Blue) ---
# Positioned at 5% width, 85% height (top left)
    # ax.text(0.6, 0.8, f"AUC: {auc_insertion:.2f}", 
    #     color=color_desc, fontsize=6, fontweight='bold', 
    #     transform=ax.transAxes)

    ax.set_ylabel(ylabel)
    print("Ylabel: ", ylabel)
    ax.set_xlabel("Perc. of edges")
    ax.legend(fontsize=6, title_fontsize=8, title='Metric', loc='lower right', bbox_to_anchor=(1.0, 0.03)) # Adjust loc as needed
    ax.set_xlim(0, 1.00) # Extend slightly for visual clarity if needed
    #ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.savefig("{}_edge_attribution_eval.png".format(dataset_name), dpi=200, bbox_inches='tight')
    plt.close()
  



if __name__ == '__main__':
    # resisc_45_model_xai_res_paths = [
    #     ("/home/results/graph_image_understanding/resisc45/Isotropic_ViG_stem_VIG/2024-10-18_14.30.03/", "Isotropic ViG"),
    #     ("/home/results/graph_image_understanding/resisc45/Pyramid_ViG_stem_PVIG/2024-10-29_10.22.57/", "Pyramid ViG"),
    #     ("/home/results/graph_image_understanding/resisc45/WIGNN_default/2024-11-06_08.39.06/", "WiGNet"),
    #     ("/home/results/graph_image_understanding/resisc45/WIGNN_windows=4_4_gsat_subgraph_r=0.7_layers=4/2024-11-13_06.55.54/", "i-WiViG")]
    #
    # liveability_45_model_xai_res_paths = [
    #     ("/home/results/graph_image_understanding/Liveability/Isotropic_ViG_stem_VIG/2024-10-18_14.53.05/","Isotropic ViG"),
    #     ("/home/results/graph_image_understanding//Liveability/Pyramid_ViG_stem_PVIG/2024-10-31_17.48.59/", "Pyramid ViG"),
    #     ("/home/results/graph_image_understanding/Liveability/WIGNN_windows=8_8_8_8/2024-11-07_22.51.23/", "WiGNet"),
    #     ("/home/results/graph_image_understanding/Liveability/WIGNN_windows=4_4_gsat_subgraph_r=0.7_layers=4/2024-11-14_08.10.59/","i-WiViG")]
    #
    # summarize_results(liveability_45_model_xai_res_paths, "Liveability")
    #edge_quantitative_evaluation("/home/results/graph_image_understanding/resisc45/WIGNN_windows=8_8_8_gsat_subgraph_r=0.5_layers=4/2025-03-05_23.03.58/quantative_analysis/", "resisc45")
    
    resisc45_path ="/home/results/graph_image_understanding//resisc45/i-WiViG_encoder=WIGNN_windows=4_4_wo_overlapCONV=mr_dilation=2_GSAT_no_info_loss_no_learn_edge_att_GIN_CONV-3_layers/2026-03-02_15.43.56/quantative_analysis/gnn_explainer/" 
    sun397_path = "/home/results/graph_image_understanding/sun397/i-WiViG_encoder=WIGNN_windows=4_4_wo_overlapCONV=mr_dilation=2_GSAT_no_info_loss_no_learn_edge_att_GIN_CONV-3_layers/2026-03-02_15.42.29/quantative_analysis/gnn_explainer/"
    livebility_path = "/home/results/graph_image_understanding/Liveability/i-WiViG_encoder=WIGNN_windows=4_4_wo_overlapCONV=mr_dilation=2_GSAT_no_info_loss_no_learn_edge_att_GIN_CONV-3_layers/2026-03-02_15.43.55/quantative_analysis/gnn_explainer/"
    
    resisc_45_gsat_path = "/home/results/graph_image_understanding/resisc45/i-WiViG_encoder=WIGNN_windows=4_4_wo_overlapCONV=mr_dilation=2_GSAT_no_info_loss_GIN_CONV-3_layers/w_gsat_weights_variance_loss_0.01/sparsity_0.7/2025-11-02_18.11.15/quantative_analysis/gsat/"
    livebility_gsat_path = "/home/results/graph_image_understanding/Liveability/i-WiViG_encoder=WIGNN_windows=4_4_wo_overlapCONV=mr_dilation=2_GSAT_with_info_loss_GIN_CONV-3_layers/2025-10-13_09.14.03/quantative_analysis/gsat"
    sun397_gsat_path = "/home/results/graph_image_understanding/sun397/i-WiViG_encoder=WIGNN_windows=4_4_wo_overlapCONV=mr_dilation=2_GSAT_with_info_loss_GIN_CONV-3_layers/2025-11-09_11.01.10/quantative_analysis/gsat"

    #edge_quantitative_evaluation(resisc45_path, "resisc45")
    # edge_quantitative_evaluation(sun397_path, "sun397")
    edge_quantitative_evaluation(resisc_45_gsat_path, "resisc45")
    #edge_quantitative_evaluation(sun397_path, "sun397")
    #edge_quantitative_evaluation(livebility_path, "Liveability")



    # xai_result_paths = [
    #     ("Resisc45", "GSAT", resisc_45_gsat_path),
    #     ("Sun397", "GSAT", sun397_gsat_path),
    #     ("Liveability", "GSAT", livebility_gsat_path),
    #     ("Resisc45", "i-WiViG", resisc45_path),
    #     ("Sun397", "i-WiViG", sun397_path),
    #     ("Liveability", "i-WiViG", livebility_path)]
    # summarize_auc_results(xai_result_paths)











