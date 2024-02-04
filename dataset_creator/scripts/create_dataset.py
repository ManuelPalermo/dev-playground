from ds_creator.autolabelling import AutoLabellerDepthAnything, AutoLabellerGroundingSAM
from ds_creator.semantic_dataset_creator import SemanticDatasetCreator
from ds_creator.utils import OUTPUT_ARTIFACT_PATTERNS, clear_directory


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
        extract_metadata=True,
        filter_similar=True,
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
        semantic_search.text_search_directory(
            directory=src,
            save_path=dst,
            text_query="a real image with a human petting an animal (e.g. cat, dog)",
            top_k=5,
        )
        semantic_search.text_search_directory(
            directory=src,
            save_path=dst,
            text_query="a complex image with humans",
            top_k=10,
        )
        semantic_search.text_search_directory(
            directory=src,
            save_path=dst,
            text_query="a scene with multiple visisble cars, possibly with humans nearby",
            top_k=5,
        )
        # search best matches in directory for a given image
        semantic_search.image_search_directory(
            directory=src,
            save_path=dst,
            image_query="./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png",
            top_k=3,
        )

    if extract_metadata:
        # process image directory and compute metadata (image embeddings, captions)
        semantic_search.compute_metadata_and_embedding_directory(
            directory=dst,
            save_path=dst,
            compute_embedding=True,
            compute_caption=True,
        )

    if filter_similar:
        # filter images which are very similar (in CLIP embedding space)
        semantic_search.filter_similar_samples(
            directory=dst,
            similarity_threshold=0.90,
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
        depthanything_model.autolabel_directory(directory=src, save_path=dst)
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
        grounding_sam_model.autolabel_directory(directory=src, save_path=dst)
        del grounding_sam_model


if __name__ == "__main__":
    main()
