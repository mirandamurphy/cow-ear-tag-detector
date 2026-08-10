# Dairy Cow Ear Tag Detector (Fine-Tuned YOLO26n)

## Overview 
The Dairy Cow Ear Tag Detector is a computer vision proof-of-concept designed to automatically detect ear tags in images of dairy cattle. By identifying animals that may be missing an ear tag, the system has the potential to reduce reliance on manual herd inspections and support timely re-tagging.

The project evaluates the feasibility of using object detection models and standard on-farm images to automate a component of dairy cattle traceability management.

## Background
Dairy cattle traceability is a vital, federally regulated component of biosecurity in Canada. Through Canada's national dairy cattle traceability program, DairyTrace, producers are required to report cattle identities and movements throughout an animal's lifetime. Accurate location data from these records allow rapid and precise responses 
during animal disease outbreaks through effective traceback procedures. 

As part of these requirements, all female dairy cattle must be identified with both a visual ear tag and an electronic RFID tag at birth. These identifiers must remain with the animal throughout its lifetime to ensure continuous traceability.

Ear tags can be lost due to environmental factors, such as rubbing against objects or weather-related damage. When a tag is lost, producers must identify the affected animal and replace the tag promptly, with the same identification number, to maintain accurate traceability records.
## Problem Statement
There is currently no practical automated method for identifying dairy cattle that have lost their ear tags using standard on-farm camera systems. As a result, producers must rely on manual herd inspections to identify missing tags and initiate re-tagging. Delays in identifying missing tags can create gaps in traceability records and
increase the burden of maintaining compliance.

This project evaluates whether computer vision can be used to automatically detect ear tags in dairy cow images and flag animals that may require further inspection.

## Project Goal
The objective of this project is to develop and evaluate an object detection model capable of identifying ear tags within dairy cow ear regions. The results will be used to assess the feasibility of automated ear tag monitoring in future development.

## Why Object Detection?
While it may seem like a simple classification task, object detection is required to meet the project's goals.

- **Multiple cows per image:** Classification can only indicate if a tag exists somewhere on the image; object detection can identify individual tags associated with specific animals.
- **Spatial localization:**  The goal is to detect ear tags in the cow’s ear, not just confirm their presence. Object detection provides bounding box coordinates.  
- **Future OCR development:** Precise tag locations will allow cropping for optical character recognition (OCR) of identification numbers.  
- **Validation:** Detection results can be visualized directly, making it easier to view predictions.

## Dataset
This project uses the **CEID-D**, a publicly available dataset hosted on Kaggle. The dataset contains images of dairy cattle captured under a wide range of real-world conditions, on a dairy farm in China.

Source: [https://www.kaggle.com/datasets/fandaoerji/cow-eartag-detection-dataset](https://www.kaggle.com/datasets/fandaoerji/cow-eartag-detection-dataset)

**The dataset includes:**
- Dairy cows at different life stages: calves, heifers, and milking cows.
- Indoor (freestall) and outdoor (open pen) environments with different lighting conditions.
- Images with one cow or multiple cows.
- Cows heads inside and outside headgates. 
- Cows at various angles and distances from the camera.
- Ear tags annotated with bounding boxes for object detection.

### Dataset Preparation
Prior to training, the dataset was organized into separate training, validation, and test datasets using a 70/15/15 split.

| Dataset | # of images |
|---------|-------------|
| Train   | 1537        |
| Val     | 329         |
| Test    | 330         |


## Future Development
Future work will expand beyond tag detection to include automated ear tag number recognition. By combining object detection with OCR, the system could support animal identification, tracking, and traceability record management through automatic extraction of tag information from images.