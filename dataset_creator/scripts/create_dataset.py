from ds_creator.autolabelling import AutoLabellerDepthAnything, AutoLabellerGroundingSAM
from ds_creator.semantic_dataset_creator import SemanticDatasetCreator
from ds_creator.utils import clear_directory


def main():
    src = "./data/images/"
    dst = "./data/output/search"

    # clear old data
    clear_directory(
        directory=dst,
        clear_search=False,
        clear_patterns=["_metadata.json", "_box2*", "_semseg*", "_depth*"],
    )
    # Select relevant data from directory (CLIP based search + Similarity filter)
    select_images(src=src, dst=dst)

    # Autolabel selected images (GroundingSAM: bbox2d + semseg masks | DepthAnything: depth)
    autolabel(src=dst, dst=dst)


def select_images(src: str, dst: str):
    # initialize dataset creator with CLIP based model
    semantic_search = SemanticDatasetCreator()

    # search best matches in directory for a given text query
    semantic_search.text_search_directory(
        directory=src,
        save_path=dst,
        text_query="a playfull animal",
        top_k=5,
    )
    semantic_search.text_search_directory(
        directory=src,
        save_path=dst,
        text_query="a scene with multiple humans",
        top_k=5,
    )
    # search best matches in directory for a given image
    semantic_search.image_search_directory(
        directory=src,
        save_path=dst,
        image_query="./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png",
        top_k=3,
    )

    # process image directory and compute metadata (image embeddings, captions)
    semantic_search.compute_metadata_and_embedding_directory(
        directory=dst,
        save_path=dst,
        compute_embedding=True,
        compute_caption=True,
    )

    # filter images which are very similar (in CLIP embedding space)
    semantic_search.filter_similar_samples(
        directory=dst,
        similarity_threshold=0.90,
    )


def autolabel(src: str, dst: str) -> None:
    # GroundingSAM: box2d + semseg
    class_onthology = {
        "cat": "cat",
        "dog of any breed": "dog",
        "colorful bird": "bird",
        "person of any race or age (human), walking or standing": "human",
    }
    grounding_sam_model = AutoLabellerGroundingSAM(grounding_sam_class_onthology=class_onthology)
    grounding_sam_model.autolabel_directory(directory=src, save_path=dst)
    del grounding_sam_model

    # DepthAnything: depth image
    depthanything_model = AutoLabellerDepthAnything()
    depthanything_model.autolabel_directory(directory=src, save_path=dst)
    del depthanything_model


if __name__ == "__main__":
    main()
