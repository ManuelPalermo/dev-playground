import os
import shutil
from argparse import ArgumentParser
from typing import List, Optional, Sequence, Tuple

from utils import (
    compare_frames_change_detection,
    draw_countours,
    find_folder_images,
    load_image_preprocess,
    preprocess_image_change_detection,
    split_list_in_chunks,
    visualize_image,
)


def filter_similar_images_in_dataset(  # noqa: C901
    dataset: Sequence[str],
    score_threshold: float = 100_000.0,
    chunk_size: Optional[int] = None,
    num_chunks: Optional[int] = None,
    gaussian_blur_radius_list: Optional[Sequence[int]] = (13,),
    min_contour_area: int = 250,
    black_mask: Tuple[int, int, int, int] = (5, 15, 5, 0),
    visualize_processing: bool = False,
) -> List[str]:
    """Selects the images which are different from the rest in the dataset.

    # TODO: possible to create a multiprocessing pool to calculate scores for each image (agains all the rest in chunk)
    # TODO: split large function into smaller funtions

    Args:
        dataset: images to evaluate.
        score_threshold: minimum score to consider images as different (images below are filtered out).
        chunk_size: blocks of images to compare between each other.
        num_chunks: number of chunks to process.
        gaussian_blur_radius_list: list of blur kernels to apply to the images before comparing them.
        min_contour_area: minimum area of a contour to be considered.
        black_mask: ignore unwanted areas of image (e.g. ceiling, etc...)
        visualize_processing: visualize the processing of the images for debugging (input, processed, countours).

    Returns:
        List of images which should be kept.

    """
    dataset_chunks = split_list_in_chunks(dataset, chunk_size)
    if num_chunks is not None:
        dataset_chunks = dataset_chunks[:num_chunks]

    imgs_to_remove: List[str] = []

    # compare each image in the chunk with each other to calculate its total score (O(n^2) :( )
    for chunk_idx, chunk in enumerate(dataset_chunks):
        for img_idx, img1 in enumerate(chunk):
            # skip images which have already been marked to remove
            if img1 in imgs_to_remove:
                continue

            removed_chunk: List[str] = []

            # load img1 preprocesses it to input representation (grayscale, 480x640, uint8, [0,255])
            img1_prep = load_image_preprocess(img1)

            # ignore images which cannot be loaded properly
            if img1_prep is None:
                removed_chunk.append(img1)
                continue

            img1_proc = preprocess_image_change_detection(
                img1_prep,
                gaussian_blur_radius_list=gaussian_blur_radius_list,
                black_mask=black_mask,
            )

            for img2 in chunk:
                if img1 != img2:
                    # load img2 preprocesses it to input representation (grayscale, 480x640, uint8, [0,255])
                    img2_prep = load_image_preprocess(img2)

                    # ignore images which cannot be loaded properly
                    if img2_prep is None:
                        removed_chunk.append(img2)
                        continue

                    img2_proc = preprocess_image_change_detection(
                        img2_prep,
                        gaussian_blur_radius_list=gaussian_blur_radius_list,
                        black_mask=black_mask,
                    )

                    # calculate score between img1 and img2
                    score, res_cnts, thresh = compare_frames_change_detection(
                        img1_proc, img2_proc, min_contour_area=min_contour_area
                    )

                    if visualize_processing:
                        img1_cnts = draw_countours(img1_prep, res_cnts)
                        visualize_image(
                            imgs=[
                                img1_prep,
                                img1_proc,
                                thresh,
                                img1_cnts,
                                img2_prep,
                                img2_proc,
                            ],
                            name=f"Vis: (img1, img1_proc, img1_thresh, img1_cnts, img2, img2_proc) "
                            f"| score: {score} "
                            f"| img1: {img1} "
                            f"| img2: {img2} ",
                        )

                    if score < score_threshold:
                        removed_chunk.append(img2)

            imgs_to_remove.extend(removed_chunk)

            print(
                f"Processed chunk: {chunk_idx+1}/{len(dataset_chunks)}  "
                f"| image: {img_idx+1}/{len(chunk)}  "
                f"| name: {img1}"
                f"| removed ({len(removed_chunk)}): {removed_chunk}  "
            )

    # return images which should be kept
    imgs_to_keep = list(set(dataset) - set(imgs_to_remove))
    return imgs_to_keep


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset containing images.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the selected images.",
        required=True,
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Number of chunks to process.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of images to compare between each other.",
    )
    parser.add_argument(
        "--clear_old",
        dest="clear_old",
        action="store_true",
        help="If the output folder should be cleared before saving the selected images.",
    )

    args = parser.parse_args()
    return args


def main():
    """Main function."""
    args = parse_args()
    print(args)

    # score images in dataset
    dataset = find_folder_images(path=args.dataset_path)

    # TODO: could add params to argparse
    debug = False  # visualize images and processing for debugging
    score_threshold = 100_000.0  # minimum score to consider images as different (images below are filtered out)
    gaussian_blur_radius_list = (11,)  # big blur kernel to ignore small differences (noise, etc...)
    min_contour_area = 250  # ignore small contours (e.g. noise, )
    black_mask = (5, 15, 5, 0)  # ignore unwanted areas of image (e.g. ceiling, etc...)

    selected = filter_similar_images_in_dataset(
        dataset,
        score_threshold=score_threshold,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
        gaussian_blur_radius_list=gaussian_blur_radius_list,
        min_contour_area=min_contour_area,
        black_mask=black_mask,
        visualize_processing=debug,
    )

    # create outputs folder
    if args.clear_old:
        shutil.rmtree(args.output_path, ignore_errors=True)
    os.makedirs(args.output_path, exist_ok=False)

    # save selected images to output path
    for img_path in selected:
        if debug:
            # visualize selected image for debugging
            visualize_image(
                imgs=[load_image_preprocess(img_path)],
                name=f"Selected img: {img_path}",
            )

        selected_output_path = os.path.join(args.output_path, os.path.basename(img_path))
        shutil.copy2(img_path, selected_output_path)
        print(f"Selected img: | name: {img_path} | save_path: {selected_output_path}")


if __name__ == "__main__":
    main()
