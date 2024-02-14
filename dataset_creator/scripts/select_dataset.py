import os

from ds_creator.semantic_dataset_creator import SemanticDatasetCreator
from ds_creator.utils import clear_directory, save_images_in_grid


def main():
    src = "./data/images/"
    dst = "./data/output/search"

    # clear old data
    clear_directory(
        directory=dst,
        clear_search=True,
        clear_patterns=(),
    )
    # select relevant data from directory (CLIP based search + Similarity filter)
    select_images(
        src=src,
        dst=dst,
        search=True,
        filter_similar=True,
        extract_metadata=True,
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
        query = "humans: a scene with real humans (men, women) doing activities outdoors"
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
            save_path=f"./results/selection/text_search_{query.split(':')[0]}.png",
            title=query,
        )

        ###
        query = "petting: a real image with a human petting an animal"
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
            save_path=f"./results/selection/text_search_{query.split(':')[0]}.png",
            title=query,
        )

        ###
        query = "cars: a scene with multiple visible cars, possibly with pedestrians nearby"
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
            save_path=f"./results/selection/text_search_{query.split(':')[0]}.png",
            title=query,
        )

        # search best matches in directory for a given image
        query = "./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png"
        query_save_path = os.path.join(dst, "bird")
        semantic_search.image_search_directory(
            directory=src,
            save_path=query_save_path,
            image_query="./data/images/birds/b86ab31a85c9d98991b99dd73283326d.png",
            top_k=5,
        )
        save_images_in_grid(
            images_or_directory=query_save_path,
            cols=5,
            save_path="./results/selection/image_search_bird.png",
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


if __name__ == "__main__":
    main()
