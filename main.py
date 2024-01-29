import _pickle
import sys
import argparse
from pathlib import Path

from load_images import *
from load_label import *
from extraction_preprocess import *
from training import *


##Note that the whole process will take a long time... please be patient
def main(config):
    # Define the dataset and expression to spot
    dataset_name = config.dataset_name
    expression_type = config.expression_type
    train = config.train
    show_plot = config.show_plot
    image_size = config.image_size
    dataset_dir = config.dataset_dir

    print(' ------ Spotting', dataset_name, expression_type, '-------')

    # Load Images
    print('\n ------ Croping Images ------')
    # Can comment this out after completed on the dataset specified and intend to try on another expression_type
    # crop_images(dataset_name)
    print("\n ------ Loading Images ------")
    images, subjects, subjectsVideos = load_images(dataset_name)
    # images = pickle.load( open( dataset_name + "_images_crop.pkl", "rb" ) )

    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_images, final_videos, final_subjects, final_samples = load_gt(dataset_name, expression_type, images,
                                                                        subjectsVideos, subjects, codeFinal)
    print('\n ------ Computing k ------')
    k = cal_k(dataset_name, expression_type, final_samples)

    # Feature Extraction & Pre-processing
    print('\n ------ Feature Extraction & Pre-processing ------')
    # dataset = extract_preprocess(final_images, k)
    # with open(
    #     Path(
    #         dataset_dir,
    #         f"dataset_{expression_type}_features_{image_size}",
    #     ).with_suffix(".pkl"),
    #     "wb",
    # ) as pkl_file:
    #     _pickle.dump(dataset, pkl_file)
    #     pkl_file.close()

    with open(
        Path(
            dataset_dir,
            f"dataset_{expression_type}_features_{image_size}",
        ).with_suffix(".pkl"),
        "rb",
    ) as pkl_file:
        dataset = _pickle.load(pkl_file)
        pkl_file.close()

    # Pseudo-labeling
    print('\n ------ Pseudo-Labeling ------')
    pseudo_y = pseudo_labeling(final_images, final_samples, k)

    # LOSO
    print('\n ------ Leave one Subject Out ------')
    X, y, groupsLabel = loso(dataset, pseudo_y, final_images, final_samples, k)

    # Model Training & Evaluation
    print('\n ------ SOFTNet Training & Testing ------')
    TP, FP, FN, metric_fn = training(X, y, groupsLabel, expression_type, final_samples, k, dataset, train,
                                     show_plot)
    final_evaluation(TP, FP, FN, metric_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME_sq')  # Specify CASME_sq or SAMMLV only
    parser.add_argument('--expression_type', type=str,
                        default='micro-expression')  # Specify micro-expression or macro-expression only
    parser.add_argument('--train', type=bool, default=True)  # Train or use pre-trained weight for prediction
    parser.add_argument('--show_plot', type=bool, default=False)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--dataset_dir', type=str, default='/home/y/LGIFNet/CASME_sq')
    config = parser.parse_args()

    main(config)
