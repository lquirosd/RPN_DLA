# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_dataset_config(cfg: CN):
    """
    Add config for additional dataset options
     - dataset name
     - dataset json file (for train and test)
     - dataset path (for train and test)
    """
    _C = cfg
    _C.DATASETS.DLA = CN()
    _C.DATASETS.DLA.REGISTER_NEW_DATASET = False
    _C.DATASETS.DLA.TRAIN = CN()
    _C.DATASETS.DLA.TRAIN.NAME = "my_dataset_train"
    _C.DATASETS.DLA.TRAIN.JSON = "instances_train.json"
    _C.DATASETS.DLA.TRAIN.PATH = "./datasets/train/"
    _C.DATASETS.DLA.TEST = CN()
    _C.DATASETS.DLA.TEST.NAME = "my_dataset_test"
    _C.DATASETS.DLA.TEST.JSON = "instances_test.json"
    _C.DATASETS.DLA.TEST.PATH = "./datasets/test/"

