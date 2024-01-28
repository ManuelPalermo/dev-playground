"""Utility script to download image samples from the internet for a given text description."""
import os

import cv2
import numpy as np
import requests

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

    for idx, each in enumerate(output["hits"][:num]):
        imageurl = each["webformatURL"]
        response = requests.get(imageurl, timeout=10)

        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        outfile = f"{save_path}/{str(idx)}.png"
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(filename=outfile, img=image)
        print(f"Downloaded file from: '{each['userImageURL']}' to {outfile}")


def main():
    """Run main entrypoint."""
    download_images(
        keywords=["bird"],
        save_path="./data/images/bird",
        num=10,
    )


if __name__ == "__main__":
    main()
