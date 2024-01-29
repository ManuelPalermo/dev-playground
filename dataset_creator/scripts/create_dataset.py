from ds_creator.semantic_dataset_creator import SemanticDatasetCreator


def main():
    # initialize dataset creator with CLIP and Grounding-SAM models
    semantic_search = SemanticDatasetCreator(
        clip_model_name_weights=("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k"),
        grounding_sam_model=True,
        device="cuda",
        image_shape=(224, 224),
    )
    # clear previous search results
    semantic_search.clear_directory(
        directory="./data/output/search",
        clear_search=True,
        clear_metadata_and_labels=True,
    )

    # ################## Select relevant data from directory ##################
    # search best matches in directory for a given text query
    semantic_search.text_search_directory(
        directory="./data/images/",
        save_path="./data/output/search",
        text_query="a playfull animal",
        top_k=10,
    )
    semantic_search.text_search_directory(
        directory="./data/images/",
        save_path="./data/output/search",
        text_query="a scene with multiple humans",
        top_k=5,
    )
    # search best matches in directory for a given image
    semantic_search.image_search_directory(
        directory="./data/images/",
        save_path="./data/output/search",
        image_query="./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png",
        top_k=3,
    )

    # ################# Compute metadata and generate Labels ##################
    # process image directory and compute metadata (image embeddings, captions, bbox_2d, semseg)
    semantic_search.compute_autolabel_directory(
        directory="./data/output/search",
        save_path="./data/output/search",
        compute_embedding=True,
        compute_caption=True,
        compute_instance_labels_onthology={
            "cat": "cat",
            "dog of any breed": "dog",
            "colorful bird": "bird",
            "person of any race or age (human), walking or standing": "human",
        },
    )


if __name__ == "__main__":
    main()
