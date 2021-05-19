# Automatic License Plate Detection from Videos using YOLOv3 and Tesseract OCR Engine

## INTRODUCTION
<p align='justify'> For the given challenge, I have considered the use of YOLOv3 Object Detection Algorithm and Tesseract OCR Engine for extraction of license plate numbers from the video. The entire project has been divided into three modules namely, 

1) Detection and Localization of License Plates in each frame using YOLOv3. 

2) Preprocessing the license plate images from each frame using OpenCV for text extraction. 

3) Extraction of license plate numbers using Tesseract OCR Engine. </p>

<p align='justify'> Out of various object detection models, like Faster-RCNN, SSD, YOLO, etc., YOLOv3 achieves a better Mean Average Precision (mAP) while maintaining a commendable Frame Rate on the Microsoft Common Objects in Context (MS COCO) dataset (based on the original YOLO v3 paper at https://arxiv.org/pdf/1804.02767.pdf). Tesseract Optical Character Recognition (OCR) Engine was chosen as it is one of leading, accurate, and industry grade OCR engine that is available. Moreover, it can produce better results with minimum hardware requirement, unlike other models like EasyOCR which works better only with a GPU enabled environment. The architecture/flow diagram of the proposed method is given in Figure 1. </p>

![alt text](https://github.com/Vigneshbaalaji/Automatic_License_Plate_Detection_from_Videos_using_YOLOv3_and_Tesseract_OCR_Engine/blob/09aa26dd58eaf56f21a3f2af02379be6c0a926f1/Architecture:Flow%20Diagram.png?raw=true)
 
## DETECTION AND LOCALIZATION USING YOLO-V3: 
 
<p align='justify'> Training YOLOv3 for this specific problem could be a tedious task, as YOLOv3 requires not only the images of license plate but also the location of license plates in the given image (annotations) as a text file. As this process of annotation has to be made manually, using graphical image annotation tools like LabelIMG, for each and every image of the dataset, it is exhaustive. Hence, pre-trained YOLOv3 was used for addressing this problem. The respective links for downloading the configuration file, weights file and the class label file is given below, </p>
 
Config File: https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector?select=darknet-yolov3.cfg

Weights File: https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector?select=lapi.weights 

Labels File: https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector?select=classes.names 
 
<p align='justify'> As this model is applied on a video, a frame-based approach is followed, where the frames from the video are extracted and each frame is passed on to the YOLOv3 model that is trained for License Plate Detection. After detection and localization, using the coordinates of the bounding boxes given by YOLOv3 the License Plate Regions of Interest (ROIs) are cropped from that frame for text extraction. </p>
 
## PROCESSING OF LICENSE PLATE IMAGES USING OPENCV: 
 
<p align='justify'> The images of number plates are converted from BGR format to Grayscale format, and Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to enhance the image. After this, Blackhat morphological operation is performed to reveal dark characters (letters and digits) against light backgrounds (the license plate itself). The kernel defined for this operation has a rectangular shape of 13 pixels wide x 5 pixels tall, which corresponds to the shape of a typical international license plate. At last, thresholding is applied on this Blackhat image, to get the final processed images of the license plate, that is ready for text extraction. <p>
 
## LICENSE PLATE NUMBER EXTRACTION USING TESSERACT OCR ENGINE: 
 
<p align='justify'> The processed license plate images that are ready for text extraction is passed onto the Tesseract OCR Engine for extraction of License Plate Numbers. To produce efficient results, duplicate values are filtered out by constructing a set from the result list. Further, regular expressions are employed to filter out meaningless and incorrect predictions. Finally, the processed result set along with the resultant video from the application of YOLOv3 License Plate Detector on the input video is displayed as final results. Even after filtration with regular expressions, some unwanted results do exist. One way to eliminate this problem while employing the model in real-time is that, we can compare the obtained results with the registered license plate numbers in the database and filter out the values that are not present in the database, as they have a higher probability of being wrong. </p>
 
## CONCLUSION AND FUTURE WORKS: 
 
<p align='justify'> Thus, a model for Automatic License Plate Recognition and text extraction is constructed successfully using YOLOv3 Object Detection Algorithm and Tesseract Optical Character Recognition Engine. Further, if we consider employing such automatic license plate recognition models on videos captured from CCTV cameras that are placed high from ground-level/from the subject, then RetinaNet can be considered as it can perform well for detecting and localizing subjects that are smaller in size, in the given image. </p>
