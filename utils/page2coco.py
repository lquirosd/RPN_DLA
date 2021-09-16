import os
import glob
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
from typing import List
from xmlPAGE import pageData
import datetime
import warnings
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from PIL import Image

def get_poly_area(poly):
    """ 
    Compute polygon area using https://en.wikipedia.org/wiki/Shoelace_formula
    """
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_paths(
    page_dir_path: str = None,
    ext: str = "xml",
    page_paths_list_path: str = None,
) -> List[str]:
    # If use paths list
    if page_paths_list_path is not None:
        return page_paths_list_path

    # If use dir path
    ext_with_dot = "." + ext if ext != "" else ""
    if page_dir_path is not None:
        page_paths = glob.glob(os.path.join(page_dir_path, "*" + ext_with_dot))
        return page_paths


def get_image_info(page, force_name=False, force_size=False):
    if force_name:
        #--- TODO: allow ext by parameter
        img_name = page.name+'.jpg'
    else:
        img_name = page.get_image_name()
    # img_id = int(page.gen_check_sum(), 16) % (10 ** 8)
    if force_size:
        im = Image.open(page.get_image_path())
        (width, height) = im.size
    else:
        (width, height) = page.get_size()

    image_info = {
        "file_name": img_name,
        "height": height,
        "width": width,
        #'id': img_id,
        "date_captured": page.get_metadata(value="Created"),
    }
    return image_info


def pagexml2cocojson(args):
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
    page_paths = get_paths(args.page_dir, args.page_ext, args.page_list)
    img_id = 0
    ann_id = 1
    categories = {}
    for in_file in tqdm(page_paths):
        #print("Working on: {}".format(in_file))
        img_id += 1
        page = pageData(in_file)
        page.parse()
        img_info = get_image_info(page, force_name=args.force_filename, force_size=args.force_image_size)
        img_info["licence"] = 1
        img_info["id"] = img_id
        coco_dic["images"].append(img_info)
        # --- get annotations
        # --- handle baselines
        #if "TextLine" in args.include:
        #    t_name = "textline"
        #    if args.classes is not None:
        #        if t_name in args.classes:
        #            categories[t_name] = args.classes.index(t_name) + 1
        #        else:
        #            raise ValueError("Class {} found in file {} is not in the list of valid classes. See --classes argument.".format(t_name,in_file))
        #    else:
        #        categories[t_name] = len(categories) + 1
        #    coco_dic["categories"].append(
        #        {
        #            "supercategory": t_name,
        #            "id": categories[t_name],
        #            "name": t_name,
        #        }
        #    )
        #    for node in page.get_region("TextLine"):
        #        bl = page.get_coords(node)
        #        x_min = float(bl[:, 0].min())
        #        x_max = float(bl[:, 0].max())
        #        y_min = float(bl[:, 1].min())
        #        y_max = float(bl[:, 1].max())
        #        ann = {
        #            "segmentation": [[int(x) for x in bl.flatten()]],
        #            "area": float(get_poly_area(bl)),
        #            "iscrowd": 0,
        #            "image_id": img_id,
        #            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        #            "category_id": categories[t_name],
        #            "id": ann_id,
        #        }
        #        ann_id += 1
        #        coco_dic["annotations"].append(ann)
        if "Table" in args.include:
            args.include.remove("Table")
            #warnings.warn("Tabless are not suported yet!!!")
            tables = page.get_region("TableRegion")
            c_sup = "TableRegion"
            #--- rows and colums are added here, cells are leavet to general extractor bellow
            if tables is not None:
                for table in tables:
                    cells = page.get_childs("TableCell", parent=table)
                    rows = {}
                    cols = {}
                    for cell in cells:
                        r, rs, c ,cs, corners = page.get_cell_data(cell)
                        c_coords = page.get_coords(cell)
                        pC = Polygon(c_coords)
                        #--- handle basic case
                        for i in range(r,r+rs):
                            if i not in rows:
                                rows[i] = []
                            rows[i].append(pC)
                        for j in range(c,c+cs):
                            if j not in cols:
                                cols[j] = []
                            cols[j].append(pC)
                    for row,cells in rows.items():
                        c_name = "row"
                        pR = cascaded_union(cells)
                        coords = np.array(pR.exterior.coords)
                        #--- add row to coco data
                        if c_sup + c_name not in categories.keys():
                            if args.classes is not None:
                                if c_name in args.classes:
                                    categories[c_sup + c_name] = args.classes.index(c_name) + 1
                                else:
                                    raise ValueError("Class {} found in file {} is not in the list of valid classes. See --classes argument.".format(c_name,in_file))
                            else:
                                categories[c_sup + c_name] = len(categories) + 1
                            coco_dic["categories"].append(
                                {
                                    "supercategory": c_sup,
                                    "id": categories[c_sup + c_name],
                                    "name": c_name,
                                }
                            )
                            x_min = float(coords[:, 0].min())
                            x_max = float(coords[:, 0].max())
                            y_min = float(coords[:, 1].min())
                            y_max = float(coords[:, 1].max())
                            ann = {
                                "segmentation": [[int(x) for x in coords.flatten()]],
                                "area": float(get_poly_area(coords)),
                                "iscrowd": 0,
                                "image_id": img_id,
                                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                                "category_id": categories[c_sup + c_name],
                                "id": ann_id,
                            }
                            ann_id += 1
                            coco_dic["annotations"].append(ann)
                    for col,cells in cols.items():
                        c_name = 'column'
                        pC = cascaded_union(cells)
                        coords = np.array(pC.exterior.coords)
                        if c_sup + c_name not in categories.keys():
                            if args.classes is not None:
                                if c_name in args.classes:
                                    categories[c_sup + c_name] = args.classes.index(c_name) + 1
                                else:
                                    raise ValueError("Class {} found in file {} is not in the list of valid classes. See --classes argument.".format(c_name,in_file))
                            else:
                                categories[c_sup + c_name] = len(categories) + 1
                            coco_dic["categories"].append(
                                {
                                    "supercategory": c_sup,
                                    "id": categories[c_sup + c_name],
                                    "name": c_name,
                                }
                            )
                            x_min = float(coords[:, 0].min())
                            x_max = float(coords[:, 0].max())
                            y_min = float(coords[:, 1].min())
                            y_max = float(coords[:, 1].max())
                            ann = {
                                "segmentation": [[int(x) for x in coords.flatten()]],
                                "area": float(get_poly_area(coords)),
                                "iscrowd": 0,
                                "image_id": img_id,
                                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                                "category_id": categories[c_sup + c_name],
                                "id": ann_id,
                            }
                            ann_id += 1
                            coco_dic["annotations"].append(ann)

        for element in args.include:
            c_sup = element
            nodes = page.get_region(element)
            if nodes is not None:
                for node in nodes:
                    # --- get element type
                    if element != "TextLine":
                        c_name = page.get_region_type(node) 
                        if c_name == None: 
                            warnings.warn( 
                                    "Type undefined for element {} on {} using SuperClass name".format(
                                    page.get_id(node), page.full_name
                                )
                            )
                            c_name = element
                    else:
                        c_name = "textline"

                    if c_name == None:
                        warnings.warn(
                            "Type undefined for element {} on {}".format(
                                page.get_id(node), page.full_name
                            )
                        )
                        continue
                    else:
                        if c_sup + c_name not in categories.keys():
                            if args.classes is not None:
                                if c_name in args.classes:
                                    categories[c_sup + c_name] = args.classes.index(c_name) + 1
                                else:
                                    raise ValueError("Class {} found in file {} is not in the list of valid classes. See --classes argument.".format(c_name,in_file))
                            else:
                                categories[c_sup + c_name] = len(categories) + 1
                            coco_dic["categories"].append(
                                {
                                    "supercategory": c_sup,
                                    "id": categories[c_sup + c_name],
                                    "name": c_name,
                                }
                            )
                        # --- get element coords
                        coords = page.get_coords(node)
                        # coords = (coords * np.flip(scale_factor, 0)).astype(np.int)
                        try:
                            x_min = float(coords[:, 0].min())
                        except:
                            print(page.get_id(node), page.name)
                        x_max = float(coords[:, 0].max())
                        y_min = float(coords[:, 1].min())
                        y_max = float(coords[:, 1].max())
                        ann = {
                            "segmentation": [[int(x) for x in coords.flatten()]],
                            "area": float(get_poly_area(coords)),
                            "iscrowd": 0,
                            "image_id": img_id,
                            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                            "category_id": categories[c_sup + c_name],
                            "id": ann_id,
                        }
                        ann_id += 1
                        coco_dic["annotations"].append(ann)
                            
    tmp = sorted(coco_dic["categories"], key=lambda k: k['id'])
    cats = [x["name"] for x in tmp]
    sup_cats = set([x["supercategory"] for x in tmp])
    print("Used categories: {}".format(cats).replace(',',''))
    print("Used Supercategories: {}".format(sup_cats).replace(',',''))
    return coco_dic


def main():
    parser = argparse.ArgumentParser(
        description="This script support converting PAGE format XML to COCO format json"
    )
    parser.add_argument(
        "--page_dir",
        type=str,
        default=None,
        help="path to page-xml files directory. It is not need when use --page_list",
    )
    parser.add_argument(
        "--page_ext",
        type=str,
        default="xml",
        help='Extensiong of PAGE-XML files. Default "xml"',
    )
    parser.add_argument(
        "--page_list",
        type=str,
        default=None,
        nargs="+",
        help="List of paths to PAGE-XML files.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="*",
        nargs="+",
        help="List of elements in PAGE-XML to be exported to COCO, default=all",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        nargs="+",
        help="List classes to be used, default=None. If None classes will be extracted from data",
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
        default="PAGE format XML to COCO Json convrted dataset",
        help="Description of the dataset.",
    )
    parser.add_argument(
        "--contributor",
        type=str,
        default="https://github.com/lquirosd/page-xml2coco",
        help="Dataset contributor[s].",
    )
    parser.add_argument(
        "--main_url",
        type=str,
        default="https://github.com/lquirosd/page-xml2coco",
        help="Default URL to dataset and dataset info",
    )
    parser.add_argument(
        "--licences", type=str, default=None, help="licences data."
    )
    parser.add_argument(
        "--force_filename",
        action='store_true',
        help="Force to use image-name equal to page file name, instead of the one defined on the PAGE metadata.",
    )
    parser.add_argument(
        "--force_image_size",
        action='store_true',
        help="Force to use image real size, instead of the one defined on the PAGE metadata.",
    )
    args = parser.parse_args()

    data = pagexml2cocojson(args)
    with open(args.output, "w") as f:
        output_json = json.dumps(data)
        f.write(output_json)


if __name__ == "__main__":
    main()
