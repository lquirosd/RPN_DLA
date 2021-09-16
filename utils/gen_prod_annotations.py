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


def get_image_info(img):
    img_name = os.path.basename(img)
    with Image.open(img) as fh:
        (width, height) = fh.size

    image_info = {
        "file_name": img_name,
        "height": height,
        "width": width,
        #'id': img_id,
        "date_captured": datetime.datetime.today().strftime("%Y/%m/%d"),
    }
    return image_info


def imgs2cocojson(args):
    # --- cocoJson format: https://cocodataset.org/#format-data
    date = datetime.datetime.today()
    coco_dic = {
        "info": {
            "year": date.year,
            "version": args.version,
            "description": args.description,
            "contributor": args.contributor,
            "url": args.main_url,
            "date_created": date.strftime("%Y/%m/%d"),
        },
        "type": "instances",
        "licences": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    # --- handle licences
    if args.licences is not None:
        for l in args.licences:
            coco_dic["licences"].append(l)
    else:
        coco_dic["licences"].append(
            {"url": args.main_url, "id": 1, "name": "Check URL for details",}
        )
    img_paths = get_paths(args.img_dir, args.img_ext, args.img_list)
    img_id = 0
    ann_id = 1
    for idx,item in enumerate(args.categories):
        sc, c = item.split(":")
        coco_dic["categories"].append(
            {
                "supercategory": sc,
                "id": idx,
                "name": c,
            }
        )

    for in_file in tqdm(img_paths):
        #print("Working on: {}".format(in_file))
        img_id += 1
        img_info = get_image_info(in_file)
        img_info["licence"] = 1
        img_info["id"] = img_id
        coco_dic["images"].append(img_info)
    return coco_dic


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
        "--categories",
        type=str,
        default=None,
        nargs="+",
        help="List of categories to be used, default=None. Format 'Superclass1:class1' 'Superclass2:class2' ...",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output.json",
        help="path to output json file",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0",
        help="Version of the dataset. Default 1.0",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Image list to COCO Json converted dataset",
        help="Description of the dataset.",
    )
    parser.add_argument(
        "--contributor",
        type=str,
        default="https://github.com/lquirosd/gen_prod_annotations",
        help="Dataset contributor[s].",
    )
    parser.add_argument(
        "--main_url",
        type=str,
        default="https://github.com/lquirosd/gen_prod_annotations",
        help="Default URL to dataset and dataset info",
    )
    parser.add_argument(
        "--licences", type=str, default=None, help="licences data."
    )
    args = parser.parse_args()

    data = imgs2cocojson(args)
    with open(args.output, "w") as f:
        output_json = json.dumps(data)
        f.write(output_json)


if __name__ == "__main__":
    main()
