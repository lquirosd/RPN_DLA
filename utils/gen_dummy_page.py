import os
import glob
import argparse
import json
from tqdm import tqdm
import re
from typing import List
import datetime
import warnings
import numpy as np
from PIL import Image
from xmlPAGE import pageData

def get_paths(
    img_dir_path: str = None,
    ext: str = "jpg",
    img_paths_list_path: str = None,
) -> List[str]:
    # If use paths list
    if img_paths_list_path is not None:
        with open(img_paths_list_path, 'r') as fh:
            img_paths = [line.rstrip('\n') for line in fh]
        return img_paths

    # If use dir path
    ext_with_dot = "." + ext if ext != "" else ""
    if img_dir_path is not None:
        img_paths = glob.glob(os.path.join(img_dir_path, "*" + ext_with_dot))
        return img_paths

def main():
    parser = argparse.ArgumentParser(
        description="This script support converting PAGE format XML to COCO format json"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="path to image files directory. It is not need when use --img_list",
    )
    parser.add_argument(
        "--img_ext",
        type=str,
        default="jpg",
        help='Extensiong of image files. Default "jpg"',
    )
    parser.add_argument(
        "--img_list",
        type=str,
        default=None,
        help="List of paths to img files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output.json",
        help="path to output json file",
    )
    args = parser.parse_args()
    img_paths = get_paths(args.img_dir, args.img_ext, args.img_list)
    for in_file in tqdm(img_paths):
        #print("Working on: {}".format(in_file))
        img_file = os.path.basename(in_file) 
        img_name = os.path.splitext(img_file)[0]
        with Image.open(in_file) as fh:
            (width, height) = fh.size
        page = pageData(os.path.join(args.output,img_name + ".xml"))
        page.new_page(img_file, str(height), str(width))
        page.save_xml()


if __name__ == "__main__":
    main()
