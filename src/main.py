# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt
    import explainability.shap_radiomics as shap_radiomics

    PLT_PARAMS = {
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2
    }
    DPI = 100

    # Update the global pyplot parameters.
    plt.rcParams.update(PLT_PARAMS)

    class_names = ["Luminal A", "Luminal B"]
    max_display = 11
    clustering_cutoff = 0.5
    example_idx = 0
    show = True
    tvt = "train"

    file_task = "data/006_LumA_LumB_BBox_liminal_tvt_prior.json"

    file_model_rf = "models/random_forest/random_forest_model.pkl"
    file_scaler_rf = "models/random_forest/random_forest_scaler.pkl"
    file_hyperparams_rf = "models/random_forest/random_forest_hyperparams.json"

    for summation_mode in ["single", "sum"]:
        sr = shap_radiomics.SHAPRadiomics(
            file_task=file_task,
            model_id=None,
            tvt=tvt,
            summation_mode=summation_mode,
            file_weights=file_model_rf,
            file_scaler=file_scaler_rf,
            file_hyperparams=file_hyperparams_rf)
        sr.visual_summary(max_display=max_display,
                          class_names=class_names,
                          example_idx=example_idx,
                          show=show)

# TODO: Debug the demonstration of Grad-CAM.
# TODO: Add or edit the .yml files.
# TODO: Remove file main.py.
# TODO: Update README.md.
# TODO: Update the code availability section in the manuscript.

