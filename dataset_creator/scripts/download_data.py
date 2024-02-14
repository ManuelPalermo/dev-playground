"""Utility script to download image samples from the internet for a given text description."""

import hashlib
import os

import cv2
import numpy as np
import requests
from ds_creator.utils import glob_images, save_images_in_grid
from tqdm import tqdm

PIXABAY_API = "https://pixabay.com/api/?key="
PIXABAY_API_KEY = ""  # NOTE: login and add you key here


def download_images(keywords: list[str], save_path: str = "./data/images/", num: int = 5) -> None:
    """Automatically download images given keywords."""
    pixabay_api = (
        PIXABAY_API + PIXABAY_API_KEY + "&q=" + "+".join(keywords).lower() + "&image_type=photo&safesearch=true"
    )

    response = requests.get(pixabay_api, timeout=10)
    print("Fetching images with request:", pixabay_api, "response code: ", response)
    output = response.json()

    for each in tqdm(output["hits"][:num], desc="Downloading images"):
        imageurl = each["webformatURL"]
        response = requests.get(imageurl, timeout=10)

        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        file_hash = hashlib.md5(image.tobytes()).hexdigest()

        outfile = f"{save_path}/{file_hash}.png"
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(filename=outfile, img=image)
        tqdm.write(f"Downloaded file from: '{each['userImageURL']}' to {outfile}")


def plot_input_images(search_path: str, title: str, save_path: str):
    image_list = glob_images(search_path)
    save_images_in_grid(
        image_list,
        cols=5,
        save_path=save_path,
        title=title,
    )


def main():
    """Run main entrypoint."""
    queries = (
        ("./data/images/humans", ["humans", "playing"], 20),
        ("./data/images/humans", ["humans", "sport"], 20),
        ("./data/images/humans", ["humans", "crowd"], 10),
        ("./data/images/cars", ["cars", "traffic"], 30),
        ("./data/images/dogs", ["dogs", "playful"], 10),
        ("./data/images/cats", ["cats", "playful"], 10),
        ("./data/images/birds", ["birds", "colorful", "nature"], 10),
        ("./data/images/petting", ["humans", "petting", "animal"], 20),
    )

    for save_path, keywords, num in queries:
        download_images(
            keywords=keywords,
            save_path=save_path,
            num=num,
        )

        title = os.path.split(save_path)[-1]
        plot_input_images(
            search_path=save_path, title=title.capitalize(), save_path=f"./results/download/input_data_grid_{title}.png"
        )


if __name__ == "__main__":
    main()
