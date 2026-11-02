# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import torchvision.transforms as tf

    import src.resnet_subtype.modality_ensemble as modality_ensemble
    import src.resnet_subtype.transform as transform

    TRANSFORM_VAL_2D = tf.Compose([
        transform.ResizeWithConstantRatio(target_size=128),
    ])
    TRANSFORM_VAL_3D = TRANSFORM_VAL_2D

    resnet_models = {"dce_mri":
                         {"channel_couple": [0, 1, 2, 3, 4, 5, 6],
                          "file_weights": "models/resnet/weights/dce_mri.pth",
                          "file_hyperparams":"models/resnet/hyperparams"
                                             "/hyperparams_dce_mri.json"},
                     "T2W":
                         {"channel_couple": [7],
                          "file_weights": "models/resnet/weights/t2w.pth",
                          "file_hyperparams":"models/resnet/hyperparams"
                                             "/hyperparams_t2w.json"},
                     "ADC":
                         {"channel_couple": [8],
                          "file_weights": "models/resnet/weights/adc.pth",
                          "file_hyperparams":"models/resnet/hyperparams"
                                             "/hyperparams_adc.json"},
                     "PET":
                         {"channel_couple": [9],
                          "file_weights": "models/resnet/weights/pet.pth",
                          "file_hyperparams":"models/resnet/hyperparams"
                                             "/hyperparams_pet.json"}
                     }
    file_task = "data/006_LumA_LumB_BBox_liminal_tvt_prior.json"
    dir_weights = "models/resnet/weights"
    dir_hyperparams = "models/resnet/hyperparams"

    ####

    alpha = 0.0
    C = 1.0
    eps_preds = 0
    eps_labels = 1e-2
    ensemble_selection = "reg_linear"

    criterion = ["patient", "bce"]
    axial_only = True
    transform_2D = TRANSFORM_VAL_2D
    transform_3D = TRANSFORM_VAL_3D

    ####

    se = modality_ensemble.SelectEnsemble(file_task=file_task,
                                          dir_base=None,
                                          criterion=criterion,
                                          modality_idxs_per_version=None,
                                          method_ensemble_selection=ensemble_selection,
                                          alpha=alpha,
                                          C=C,
                                          eps_preds=eps_preds,
                                          eps_labels=eps_labels,
                                          axial_only=axial_only,
                                          transform_2D=transform_2D,
                                          transform_3D=transform_3D,
                                          load_best_model_info=None,
                                          dict_models_light=resnet_models)
    se.predict_evaluate(tvt="test",
                        verbose=True)

# TODO: Debug the demonstration of explainability.
# TODO: Update the code availability section in the manuscript.
# TODO: Remove file main.py.
# TODO: Update README.md.
