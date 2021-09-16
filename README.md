# RPN_DLA
RPN approach to DLA problem

# Requirements
* Linux (OSX may work, but untested.).
* Python (3.7 under conda virtual environment is recomended)
* Pytorch (1.5.1)
* [Detectron2](https://github.com/facebookresearch/detectron2)

# Installation

1. Install python dependencies using [requirements file](rpndla_env.yml)
2. Install detectron2

# Usage

1. Input data must follow the folder structure `data_tag/page`, where images must be into the `data_tag` folder and xml files into `page`. For example:
```bash
mkdir -p data/{train,val,test,prod}/page;
tree data;
```

```
data
├── prod
│   ├── page
│   │   ├── prod_0.xml
│   │   └── prod_1.xml
│   ├── prod_0.jpg
│   └── prod_1.jpg
├── test
│   ├── page
│   │   ├── test_0.xml
│   │   └── test_1.xml
│   ├── test_0.jpg
│   └── test_1.jpg
├── train
│   ├── page
│   │   ├── train_0.xml
│   │   └── train_1.xml
│   ├── train_0.jpg
│   └── train_1.jpg
└── val
    ├── page
    │   ├── val_0.xml
    │   └── val_1.xml
    ├── val_0.jpg
    └── val_1.jpg
```

2. Convert PAGE-XML data into COCO-JSON 
```bash 
python utils/page2coco.py --page_dir <> --include <> --classes <> --output <>
```

3. Run the tool
```bash
python train_net.py --num-gpus <1> --config-file <configs/> SOLVER.IMS_PER_BATCH <4> SOLVER.BASE_LR <0.01>
```
An example config file can be found in [configs](configs) folder

4. Convert results fron JSON to PAGE-XML
```bash
python utils/detectronCoco2page.py --results_json <> --dataset_json <> --output <>
```

5. Gen the baselines if required
```bash
python utils/baselines/baselines.py --img_dir <> --page_dir <> --out_dir <>
```

# License

Apache 2.0 license.
See [LICENSE](LICENSE) to see the full text.
