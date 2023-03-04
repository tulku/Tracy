#!/usr/bin/env python
"""
Transforms a series of pictures in a folder given the april 
that can be detected on a provided image.
"""
import argparse
import os
import sys

import cv2

from camera_calibration import (
    calculate_transform,
    find_screen_corners,
    open_image,
    transform_frame,
)


def get_transform(april_tag_image: str, image_size: int):
    tag_image = open_image(april_tag_image)
    corners = find_screen_corners(tag_image)
    if corners is None:
        sys.exit(f"Cannot detect tag in image {april_tag_image}")
    return calculate_transform(corners, image_size)


def create_directory(output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)


def write_image(image, path: str):
    cv2.imwrite(path, image)


def transform_directory(input_dir, output_dir, transform, target_size: int):
    for image in os.listdir(input_dir):
        output_name = os.path.join(output_dir, image)
        frame = open_image(os.path.join(input_dir, image))
        transformed_frame = transform_frame(frame, transform, target_size)
        write_image(transformed_frame, output_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("april_tag_image", help="Path to the image with the apriltag.")
    parser.add_argument(
        "input_dir", help="Path to the directory with the images to convert"
    )
    parser.add_argument("output_dir", help="Path to the output directory.")
    parser.add_argument(
        "--size",
        default=640,
        type=int,
        help="Size of the image side in pixels. Output is always square.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    transform = get_transform(args.april_tag_image, args.size)
    create_directory(args.output_dir)
    transform_directory(args.input_dir, args.output_dir, transform, args.size)


if __name__ == "__main__":
    main()
