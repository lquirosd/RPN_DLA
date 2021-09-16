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
import pycocotools.mask as mask_util
import cv2
import string
import random


def rle_to_poly(rle_data):
    mask = mask_util.decode(rle_data)
    #--- force copy array, so CV2 can access it on correct format
    mask = mask.copy()
    res_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #--- check len to support CV versions
    if len(res_) == 2:
        contours, hierarchy = res_
    else:
        _, contours, hierarchy = res_
    if len(contours) > 1:
        print("Warning: more than one polygon found on RLE mask, only first is used")
    if len(contours) == 0:
        return np.array([])
    else:
        return contours[0] 

def get_image_by_id(dataset, idx):
    for img in dataset["images"]:
        if img["id"] == idx:
            return img
    return None

def get_class_by_id(dataset,idx):
    for cls in dataset["categories"]:
        if cls["id"] == idx:
            return cls

    return None

def get_image_by_name(dataset, name):
    for img in dataset["images"]:
        if img["file_name"] == name:
            return img
    return None

def arrange_instances_by_sample(data, dataset, score_th):
    sample_dic = {}
    for instance in data:
        if instance["score"]< score_th:
            continue
        if instance["image_id"] not in sample_dic.keys():
            sample_dic[instance["image_id"]] = {"regs":[], "lines":[]}
        if get_class_by_id(dataset, instance["category_id"])["supercategory"] == "TextLine":
            sample_dic[instance["image_id"]]["lines"].append(instance)
        else:
            sample_dic[instance["image_id"]]["regs"].append(instance)
    #img = get_image_by_name(dataset, "170025120000003,0086.tif")
    #print(dataset["categories"])
    #return {img["id"]:sample_dic[img["id"]]}
    return sample_dic

def lines_to_regions(data):
    #--- add lines key to all regions
    for sample_id, sample in data.items():
        for y in sample["regs"]:
            y["lines"] = []
        for x in sample["lines"]:
            max_iou = 0
            reg = None
            for i,y in enumerate(sample["regs"]):
                l_iou = mask_util.iou([x["segmentation"]],[y["segmentation"]],np.array([1]))[0][0]
                if l_iou >= max_iou:
                    max_iou = l_iou
                    reg = i
            if reg is not None:
                sample["regs"][reg]["lines"].append(x)
    return data


def remove_overlapping(data, th, allowed_overlap=[]):
    for sample_id,sample in data.items():
        #iou = np.zeros((len(sample["regs"]),len(sample["regs"])))
        clean_data = []
        merged = []
        for x,sx in enumerate(sample["regs"]):
            if x in merged:
                continue
            if sx["category_id"] in allowed_overlap:
                #print("sx added of type {}".format(sx["category_id"]))
                clean_data.append(sx)
                merged.append(x)
                continue
            for y,sy in enumerate(sample["regs"][x+1:]):
                #if sy["category_id"] in allowed_overlap:
                #    print("sy added of type {}".format(sy["category_id"]))
                #    clean_data.append(sy)
                #    merged.append(x+y+1)
                #    print("in allowed:", merged)
                #    continue
                iou = mask_util.iou([sx["segmentation"]], [sy["segmentation"]], np.array([1]))[0][0]
                if iou > th:
                    merged.append(x+y+1)
                    print("in iou:", merged)
                    union = mask_util.merge([sx["segmentation"],sy["segmentation"]])
                    #tmp = sx.copy()
                    sx["segmentation"] = union
                    #--- TODO: use real posterior prob of each element instead of this "average"
                    if sx["score"] < sy["score"]:
                        sx["category_id"] = sy["category_id"]
                    sx_area = mask_util.area(sx["segmentation"])
                    sy_area = mask_util.area(sy["segmentation"])
                    sx["score"] = (sx["score"]*sx_area + sy["score"]*sy_area)/(sx_area + sy_area)
            #print("sx added end of type {}".format(sx["category_id"]))
            clean_data.append(sx)
        sample["regs"] = clean_data
        #print(merged)
    return data

def json_to_page(dataset, data, out_dir, th, score_th, allowed_overlap=[]):
    validValues = string.ascii_uppercase + string.ascii_lowercase + string.digits
    page_dic = {}
    data = arrange_instances_by_sample(data,dataset,score_th)
    allowed_cls_id = []
    for cls in dataset["categories"]:
        if cls["name"] in allowed_overlap:
            allowed_cls_id.append(cls["id"])
    data = remove_overlapping(data,th, allowed_overlap=allowed_cls_id)
    data = lines_to_regions(data) 
    for sample_id, sample in tqdm(data.items(), desc="Processing data"):
        for hyp in sample["regs"]:
            #--- Ignore TextRegions without lines
            #if len(hyp["lines"]) == 0:
            #    continue
            if hyp["score"] < score_th:
                continue
            if hyp["image_id"] not in page_dic.keys():
                img = get_image_by_id(dataset, hyp["image_id"])
                if img == None:
                    print("Error: No image found for id {}".format(hyp["image_id"]))
                    continue
                img_name = os.path.splitext(img["file_name"])[0]
                page_dic[hyp["image_id"]] = pageData(os.path.join(out_dir,
                    img_name + ".xml"))
                page_dic[hyp["image_id"]].new_page(img["file_name"],
                        str(img["height"]), str(img["width"]))
            #--- add new element
            cls = get_class_by_id(dataset, hyp["category_id"])
            r_class = cls["supercategory"]
            r_type = cls["name"]
            r_id = ''.join(random.choice(validValues) for _ in range(4))
            r_score = str(hyp["score"])
            r_coords = ""
            for x in rle_to_poly(hyp["segmentation"]).reshape(-1,2):
                r_coords = r_coords + " {},{}".format(x[0], x[1])
            reg_node = page_dic[hyp["image_id"]].add_element(r_class, r_id, r_type, r_coords.strip(), score=r_score)
            for line in hyp["lines"]:
                l_class = "TextLine"
                l_type = r_type
                l_id = ''.join(random.choice(validValues) for _ in range(4))
                l_score = str(line["score"])
                l_coords = ""
                for x in rle_to_poly(line["segmentation"]).reshape(-1,2):
                    l_coords = l_coords + " {},{}".format(x[0], x[1])
                page_dic[hyp["image_id"]].add_element(l_class,
                        l_id, 
                        l_type, 
                        l_coords.strip(), 
                        score=l_score,
                        parent=reg_node)
    #--- add dummy page for images without instances
    for img in dataset['images']:
        if img['id'] not in page_dic.keys():
            print("No instances found for image {}. Generating a dummy xml file...".format(img["file_name"]))
            img_name = os.path.splitext(img["file_name"])[0]
            page_dic[img["id"]] = pageData(os.path.join(out_dir,img_name + ".xml"))
            page_dic[img["id"]].new_page(img["file_name"], str(img["height"]), str(img["width"]))


    for _,page in tqdm( page_dic.items(), desc="Saving generated files"):
        #print("Saving {} ...".format(page.name))
        page.save_xml()


def json2page(dataset, data, out_dir, score_th):
    validValues = string.ascii_uppercase + string.ascii_lowercase + string.digits
    page_dic = {}
    full_page = {}
    for hyp in tqdm(data, desc="Processing data"):
        if hyp["score"] < score_th:
            continue
        if hyp["image_id"] not in page_dic.keys():
            img = get_image_by_id(dataset, hyp["image_id"])
            if img == None:
                print("Error: No image found for id {}".format(hyp["image_id"]))
                continue
            img_name = os.path.splitext(img["file_name"])[0]
            page_dic[hyp["image_id"]] = pageData(os.path.join(out_dir,
                img_name + ".xml"))
            page_dic[hyp["image_id"]].new_page(img["file_name"],
                    str(img["height"]), str(img["width"]))
            #--- add "dummy" fullpage region
            #--- TODO: remove when Line to region code is ready
            full_page[hyp["image_id"]] = page_dic[hyp["image_id"]].add_element("TextRegion",
                    "fullpage", 
                    "fullpage", 
                    "0,0 "+ str(img["width"]) + ",0 " +  str(img["width"]) + ","+ str(img["height"]) + " 0," + str(img["height"]))

        #--- add new element
        cls = get_class_by_id(dataset, hyp["category_id"])
        r_class = cls["supercategory"]
        r_type = cls["name"]
        r_id = ''.join(random.choice(validValues) for _ in range(4))
        r_score = str(hyp["score"])
        r_coords = ""
        for x in rle_to_poly(hyp["segmentation"]).reshape(-1,2):
            r_coords = r_coords + " {},{}".format(x[0], x[1])

        if r_class == "TextLine":
            #--- add to a "fullpage" region
            #--- TODO: add to overlapping region
            #--- TODO: get baseline from this "line"
            page_dic[hyp["image_id"]].add_element(r_class,
                    r_id, 
                    r_type, 
                    r_coords.strip(), 
                    score=r_score,
                    parent=full_page[hyp["image_id"]])
        else:
            page_dic[hyp["image_id"]].add_element(r_class, r_id, r_type, r_coords.strip(), score=r_score)


    for _,page in tqdm( page_dic.items(), desc="Saving generated files"):
        #print("Saving {} ...".format(page.name))
        page.save_xml()




def main():
    parser = argparse.ArgumentParser(
        description="This script support converting Detectron2 COCO JSON format to PAGE XML format"
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="path to Detectron2 output json file.",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        default=None,
        help="path to Dataset json file.",
    )
    parser.add_argument(
        "--only_bl",
        type=bool,
        default=False,
        help="Gen a PAGE-XML only with baselines",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="path to output directory",
    )
    parser.add_argument(
        "--score_th",
        type=float,
        default=0.5,
        help="Keep only hypotesis with score hier than this",
    )
    parser.add_argument(
        "--overlap_th",
        type=float,
        default=0.5,
        help="Maximum overlap alllowed between reagions",
    )
    parser.add_argument(
        "--allow_overlap",
        type=str,
        nargs="+",
        default=[],
        help="List of regions that allow overlap with another region",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0",
        help="Version of the data. Default 1.0",
    )
    args = parser.parse_args()
    with open(args.dataset_json) as fh:
        dataset = json.load(fh)

    with open(args.results_json) as fh:
        results = json.load(fh)
    if args.only_bl:
        json2page(dataset, results, args.output, args.score_th)
    else:
        json_to_page(dataset, 
            results, 
            args.output, 
            args.overlap_th, 
            args.score_th,
            allowed_overlap=args.allow_overlap)





if __name__ == "__main__":
    main()
