"""Utility script to download image samples from the internet for a given text description."""

import hashlib
import os

import cv2
import numpy as np
import requests
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


def main():
    """Run main entrypoint."""
    queries = (
        ("./data/images/humans", ["humans", "crowd", "many", "activities"], 20),
        ("./data/images/dogs", ["dogs"], 10),
        ("./data/images/cats", ["cats"], 10),
        ("./data/images/birds", ["birds"], 10),
        ("./data/images/mixed", ["humans", "animals", "petting", "outdoors"], 10),
    )

    for save_path, keywords, num in queries:
        download_images(
            keywords=keywords,
            save_path=save_path,
            num=num,
        )


if __name__ == "__main__":
    main()