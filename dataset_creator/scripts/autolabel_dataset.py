import os
from itertools import chain

from ds_creator.autolabeling import AutoLabellerDepthAnything, AutoLabellerGroundingSAM
from ds_creator.utils import gather_images_and_labels, save_images_in_grid


def main():
    #src = "./data/images/"
    dst = "./data/output/search"

    # clear old data
    #clear_directory(
    #   directory=dst,
    #   clear_search=False,
    #   clear_patterns=OUTPUT_ARTIFACT_PATTERNS,
    #)

    # autolabel selected images (GroundingSAM: bbox2d + semseg masks | DepthAnything: depth)
    autolabel(
        src=dst,
        dst=dst,
        depth=True,
        instance_segmentation=True,
    )


def autolabel(
    src: str,
    dst: str,
    depth: bool = True,
    instance_segmentation: bool = True,
) -> None:
    if depth:
        # DepthAnything: depth image
        depthanything_model = AutoLabellerDepthAnything()
        depthanything_model.autolabel_directory(directory=src, save_path=None)
        del depthanything_model

    if instance_segmentation:
        # GroundingSAM: box2d + semseg
        class_onthology = {
            "cat": "cat",
            #
            "dog": "dog",
            #
            "exotic bird": "bird",
            "poultry bird": "bird",
            #
            "any animal in the image which is not a pet (exclude any: cat, dog, bird)": "animal",
            #
            "car": "vehicle",
            "bus": "vehicle",
            "motorbike": "vehicle",
            "vehicle": "vehicle",
            #
            "human": "human",
            "person": "human",
            "man": "human",
            "woman": "human",
            "child": "human",
            "a human doing any activity": "human",
            "a human doing sports": "human",
        }
        known_classes = str(tuple(set(class_onthology.values()))).replace("'", "")
        unknown_classes_prompt = (
            f"objects in the image which are salient but dont belong to known classes: {known_classes}"
        )
        class_onthology[unknown_classes_prompt] = "unknown"

        grounding_sam_model = AutoLabellerGroundingSAM(grounding_sam_class_onthology=class_onthology, image_shape=None)
        grounding_sam_model.autolabel_directory(directory=src, save_path=None)
        del grounding_sam_model

    # visualize original images and generated autolabels
    visu_dirs = {
        "autolabeling_all": dst,
        **{f"autolabeling_{subdir_class}": os.path.join(dst, subdir_class) for subdir_class in os.listdir(dst)},
    }
    for visu_name, visu_dir in visu_dirs.items():

        img_and_labels = gather_images_and_labels(
            directory=visu_dir,
            label_patterns=["_depth.png", "_instseg.png", "_semseg.png", "_caption.png"],
        )
        num_images = len(list(img_and_labels.values())[0])
        save_images_in_grid(
            list(chain.from_iterable(img_and_labels.values())),
            cols=num_images,
            save_path=f"./results/autolabeling/{visu_name}.png",
            title=visu_name.replace("_", ":  ").capitalize(),
        )


if __name__ == "__main__":
    main()
