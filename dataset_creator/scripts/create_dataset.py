from ds_creator.clip_image_search import ClipImageSearch


def main():
    semantic_search = ClipImageSearch()

    # search best matches in directory for a given text query
    semantic_search.text_search_directory(
        directory="./data/images/",
        save_path="./data/output/search",
        text_query="an orange and white cat",
        top_k=3,
    )
    # search best matches in directory for a given image
    semantic_search.image_search_directory(
        directory="./data/images/",
        save_path="./data/output/search",
        image_query="./data/images/bird/7.png",
        top_k=3,
    )

    # process image directory and compute metadata (image embeddings, captions, etc..)
    semantic_search.compute_metadata_directory(
        directory="./data/output/search",
        save_path="./data/output/search",
        write_embedding=True,
        write_caption=True,
    )
    # search.clear_metadata_directory(directory="./data/output/search")


if __name__ == "__main__":
    main()
