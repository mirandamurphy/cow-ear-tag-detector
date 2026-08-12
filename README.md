# Cow Ear Tag Detector 🐮

Fine-tuned YOLO26n model that detects ear tags on dairy cows, developed as a proof-of-concept for automated missing tag detection on farms.

**Trained Model (on Hugging Face):** [Cow Ear Tag Detector YOLO26n](https://huggingface.co/mirandamurphy/cow-ear-tag-detector-yolo26n)

## Overview
Canada's DairyTrace program requires every female dairy cow to be identified with both a visual and RFID tag for life.
Tags can fall off for many reasons, including, environmental factors (e.g., cows rubbing against objects) or weather-related damage. 
To comply with Canada's national dairy cattle traceability program, DairyTrace, producers have to manually inspect their herd and prompty order missing tags so the cow can be re-tagged. 

The goal of this project is to explore the potential of using object detection to automatically detect ear tags in dairy cows and flag animals that may require further inspection. 

### Why object detection, not classification?
Object detection is required to meet the project's goals:
- Detects and differentiates multiple individual cows and their tags within a single image
- Precisely localizes ear tags using bounding box coordinates  
- Makes it easy to visualize and validate results
- Allows cropping of detected regions for future OCR of ID numbers in future development
## Project Structure 

```
cow-ear-tag-detector
├── datasets                                
│   └── yolo
│       ├── images
│       │   ├── train
│       │   ├── test
│       │   └── val
│       ├── labels
│       │   ├── train
│       │   ├── test
│       │   └── val
│       └── data.yaml
├── models
│   └── cow_ear_tag_detector_yolo26n.pt
├── notebooks
│   ├── 01_train_yolo26n.ipynb
│   └── 02_test_yolo26n.ipynb
├── scripts
│   ├── calc_label_counts.py
│   ├── preprocess.py
│   ├── step01_remove_duplicates.py
│   ├── step_02_remove_blurry.py
│   └── step_03_split_dataset.py
├── utils
│   ├── constants.py
│   ├── definitions.py
│   └── logger_config.py
├── LICENSE
└── README.md


```

### Dataset
This project uses the [CEID-D dataset](https://www.kaggle.com/datasets/fandaoerji/cow-eartag-detection-dataset/data), a publicly available dataset on Kaggle. The dataset contains images of dairy cattle captured under a wide range of real-world conditions, on a dairy farm in China. 

This repository does not redistribute CEID-D images or original annotations.

Prior to training, the dataset was pre-processed by running the `preprocess.py` script. This ran some pre-processing steps and organized the dataset into separate training, validation, and test datasets using a 70/15/15 split.

| Split | # of images |
|-------|-------------|
| Train | 1537        |
| Val   | 329         |
| Test  | 330         |

### Results

| Metric             | Value |
|--------------------|-------|
| mAP50              | 0.915 |
| mAP50-95           | 0.403 |
| Precision          | 0.892 |
| Recall             | 0.865 |
| Inference (ms/img) | 6.7ms |

### Limitations 
- Ear tag sizes vary globally; the model trained only on Chinese cow images may not generalize elsewhere.
- Mostly black and white Holsteins; breed diversity underrepresented. 
- Heifers and adult cows mainly represented; very young calves underrepresented.
- Data from one farm's barn and outdoor pens; limited environment variation. 
- Tested only on static images.
- No automated tests yet

### Future Development
- Automated testing
- Combine with OCR to read ear tag numbers 
- More diverse data collection 

### License
AGPL-3.0

### Acknowledgements

Base model (Ultralytics YOLO26):
```
Ultralytics YOLO26n (Base Model)
@misc{jocher2026ultralyticsyolo26unifiedrealtime,
  title = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
  year = {2026},
  eprint = {2606.03748},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2606.03748},
  url = {https://arxiv.org/abs/2606.03748},
}
```
CEID-D Dataset:
```
@Article{s24072194,
AUTHOR = {Gao, Tianhong and Fan, Daoerji and Wu, Huijuan and Chen, Xiangzhong and Song, Shihao and Sun, Yuxin and Tian, Jia},
TITLE = {Research on the Vision-Based Dairy Cow Ear Tag Recognition Method},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {7},
ARTICLE-NUMBER = {2194},
URL = {https://www.mdpi.com/1424-8220/24/7/2194},
PubMedID = {38610405},
ISSN = {1424-8220},
DOI = {10.3390/s24072194}
}
```


