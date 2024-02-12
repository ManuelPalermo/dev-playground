import os

from ds_creator.autolabelling import AutoLabellerDepthAnything, AutoLabellerGroundingSAM
from ds_creator.semantic_dataset_creator import SemanticDatasetCreator
from ds_creator.utils import OUTPUT_ARTIFACT_PATTERNS, clear_directory, save_images_in_grid


def main():
    src = "./data/images/"
    dst = "./data/output/search"

    # clear old data
    clear_directory(
        directory=dst,
        clear_search=True,
        clear_patterns=OUTPUT_ARTIFACT_PATTERNS,
    )
    # Select relevant data from directory (CLIP based search + Similarity filter)
    select_images(
        src=src,
        dst=dst,
        search=True,
        filter_similar=True,
        extract_metadata=True,
    )

    # Autolabel selected images (GroundingSAM: bbox2d + semseg masks | DepthAnything: depth)
    autolabel(
        src=dst,
        dst=dst,
        depth=True,
        instance_segmentation=True,
    )


def select_images(
    src: str,
    dst: str,
    search: bool = True,
    extract_metadata: bool = True,
    filter_similar: bool = True,
):
    # initialize dataset creator with CLIP based model
    semantic_search = SemanticDatasetCreator()

    if search:
        # search best matches in directory for a given text query
        ###
        query = "petting: a real image with a human petting an animal (e.g. cat, dog)"
        query_save_path = os.path.join(dst, query.split(":")[0])
        semantic_search.text_search_directory(
            directory=src,
            save_path=query_save_path,
            text_query=query,
            top_k=5,
        )
        save_images_in_grid(
            images_or_directory=query_save_path,
            cols=5,
            save_path=f"./results/semantic_search/text_search_{query.split(':')[0]}.png",
            title=query,
        )

        ###
        query = "humans: a complex scene with humans"
        query_save_path = os.path.join(dst, query.split(":")[0])
        semantic_search.text_search_directory(
            directory=src,
            save_path=query_save_path,
            text_query=query,
            top_k=10,
        )
        save_images_in_grid(
            images_or_directory=query_save_path,
            cols=5,
            save_path=f"./results/semantic_search/text_search_{query.split(':')[0]}.png",
            title=query,
        )

        ###
        query = "cars: a scene with multiple visible cars, possibly with humans nearby"
        query_save_path = os.path.join(dst, query.split(":")[0])
        semantic_search.text_search_directory(
            directory=src,
            save_path=query_save_path,
            text_query=query,
            top_k=5,
        )
        save_images_in_grid(
            images_or_directory=query_save_path,
            cols=5,
            save_path=f"./results/semantic_search/text_search_{query.split(':')[0]}.png",
            title=query,
        )

        # search best matches in directory for a given image
        query = "./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png"
        query_save_path = os.path.join(dst, "bird")
        semantic_search.image_search_directory(
            directory=src,
            save_path=query_save_path,
            image_query="./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png",
            top_k=4,
        )
        save_images_in_grid(
            images_or_directory=query_save_path,
            cols=4,
            save_path="./results/semantic_search/text_search_bird.png",
            title=query,
        )

    if filter_similar:
        # filter images which are very similar (in CLIP embedding space)
        semantic_search.filter_similar_samples(
            directory=dst,
            similarity_threshold=0.90,
        )

    if extract_metadata:
        # process image directory and compute metadata (image embeddings, captions)
        semantic_search.compute_metadata_and_embedding_directory(
            directory=dst,
            save_path=None,
            compute_embedding=True,
            compute_caption=True,
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


if __name__ == "__main__":
    main()
