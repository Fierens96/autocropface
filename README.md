# autocropface
face auto cropping with yolo and opencv

This module combines YOLO object detection and OpenCV image processing, aiming to quickly and automatically crop individuals in images and provide a positional offset feature. 
The output images are square, and the aesthetics of cropping are subjective. 
Based on the cropping and scaling settings, the typical output consists of 9 images commonly.

Please note that if facial features are dispersed, the algorithm retains the central facial region near the square boundary, emphasizing the distance between the central facial region and the square boundary to preserve facial proportions in the image to the greatest extent possible.
