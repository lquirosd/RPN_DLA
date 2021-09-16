import os
import glob
import argparse
import json
import pycocotools.mask as maskUtils


#--- expected format:
#   [{"image_id":int, "category_id":int, "bbox":list, "score": float, "segmentation":{"size":list, "counts":RLE}},{...}]

def list2dic(json_data, key="file_name"):
    to_return = {}
    for img in json_data["images"]:
        name = img[key]
        to_return[name] = img
    return to_return

#def map_cats(gt_cats,hyp_cats):
#    for c in gt_cats:


def dataset2result(gt_data, hyp_data):
    gt_ids = list2dic(gt_data,key="id")
    hyp_ids = list2dic(hyp_data, key="id")
    gt_name = list2dic(gt_data)
    hyp_name = list2dic(hyp_data)
    res_data = []
    for obj in hyp_data["annotations"]:
        o_img_id = obj["image_id"]
        gt_img_id = gt_name[hyp_ids[o_img_id]["file_name"]]["id"]
        h=gt_ids[gt_img_id]["height"]
        w=gt_ids[gt_img_id]["width"]
        if "score" in obj.keys():
            o_score = obj["score"]
        else:
            o_score = 1
        try:
            rles = maskUtils.frPyObjects(obj["segmentation"], h, w)
        except:
            print(obj["id"])
            continue
        rle = maskUtils.merge(rles)
        rle["counts"] = rle["counts"].decode("utf-8")

        #--- assuming thet cats are the same for both
        res_data.append({
            "image_id":gt_img_id,
            "category_id":obj["category_id"],
            "bbox":obj["bbox"],
            "score":o_score,
            "segmentation":rle
            })

    return res_data

def main():
    parser = argparse.ArgumentParser(
        description="This script support converting dataset COCO JSON to result COCO JSON"
    )
    parser.add_argument(
        "--gt_json",
        type=str,
        default=None,
        help="path to dataset JSON",
    )
    parser.add_argument(
        "--hyp_json",
        type=str,
        default=None,
        help='path to results JSON in dataset format',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output.json",
        help="path to output json file",
    )
    args = parser.parse_args()
    
    with open(args.gt_json,'r') as fh:
        gt_data = json.load(fh)
    with open(args.hyp_json,'r') as fh:
        hyp_data = json.load(fh)

    data = dataset2result(gt_data, hyp_data)

    with open(args.output, "w") as f:
        output_json = json.dumps(data)
        f.write(output_json)

if __name__ == "__main__":
    main()
