# DeepGlobeRoadExtraction
![thumbnail2](https://user-images.githubusercontent.com/42332501/175374360-74082f58-bae8-4a84-9fe5-d35ad92ef88b.png)

## 1. Introduction
In disaster zones, especially in developing countries, maps and accessibility information are crucial for crisis response. Deep Globe Road Extraction Challenge poses the challenge of automatically extracting roads and street networks from satellite images.

## 2. Business problem
It is basically a computer vision based deep learning problem. Using satellite images provided in the dataset and the state of the art segmentation based algorithms we should be able to extract the road from the satellite images.

## 3. Mapping the Business problem to Deep learning
It is Computer vision based Semantic segmentation problem. Here we have 2 classes one is road and remaining objects in image will be other than road like buildings , grass etc. It is binary image segmentation task.

## 4. Data set analysis
We have total of 14796 satellite and mask images

* Training satellite images : 6226
* Training mask images : 6226
* Validation satellite images :1243
* Test satellite images :1101

We need to predict mask images for validation and test image datasets.

Kaggle link :
https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset?select=class_dict.csv

## 5. Performance metric

**Intersection over union (IOU):**
* The Intersection-Over-Union (IoU), also known as the Jaccard Index, is one of the most commonly used metrics in semantic segmentation. The IoU is a very straight forward metric that’s extremely effective.
* IoU is the area of overlap between the predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth.

![IOU](https://user-images.githubusercontent.com/42332501/175777592-ab6d5e14-3a3e-4a86-9e9e-c377b61b027f.png)

## 6.OpenCV experimentation
* **working examples:**

![opencv_working](https://user-images.githubusercontent.com/42332501/175778194-d54b396b-7591-47f4-aaac-c05b95dc5e53.png)

* **Failing examples:**

![opencv_failing](https://user-images.githubusercontent.com/42332501/175778374-c7fa72a6-4780-4277-abba-52f391b35224.png)

## 7. Advanced Deep learning modelling
Few techniques that we tried in this advanced deep learning modelling:
* **Transfer learning**

It has the benefit of decreasing the training time for a neural network model and can result in lower generalization error.The weights in re-used layers may be used as the starting point for the training process and adapted in response to the new problem. This usage treats transfer learning as a type of weight initialization scheme.
This may be useful when the first related problem has a lot more labeled data than the problem of interest and the similarity in the structure of the problem may be useful in both contexts. So in our U-Net models that we are using we shall use encoder freeze for backbone architectures.

* **Online or Real-time Augmentation**

Deep learning Convolutional Neural Networks(CNN) need a huge number of images for the model to be trained effectively. This helps to increase the performance of the model by generalizing better and thereby reducing overfitting. As the name suggests, augmentation is applied in real time. This is usually applied for larger data sets as we do not need to save the augmented images on the disk. In this case, we apply transformations in mini-batches and then feed it to the model. Online augmentation model will see different images at each epoch. The model generalizes better with online augmentation as it sees more samples during training with online data augmentation.We will be using imgaug class for demonstrating Image Augmentation.

* **Custom Data Generator**

You probably encountered a situation where you try to load a dataset but there is not enough memory in your machine. As the field of machine learning progresses, this problem becomes more and more common. Today this is already one of the challenges in the field of vision where large datasets of images and video files are processed.
The ImageDataGenerator is an easy way to load and augment images in batches for image classification tasks. But! What if you have a segmentation task? For that, we need to build a custom data generator .To build a custom data generator, we need to inherit from the Sequence class.

* **Multiple Back bone architectures in U-Net**

The backbones used are often Vanilla CNNs such as VGG16, ResNet34, InceptionV3, Densenet201 etc which performs encoding and downsampling by itself. These networks are taken and their counter parts are built to perform decoding and up sampling to form the final Unet. We shall use Unet architecture by using encoder as different types of backbones as mentioned above. By the using the above state of the art architectures along with transfer learning will be helpful in achieving desired performance metric very quickly.

![U-Net](https://user-images.githubusercontent.com/42332501/175778749-231b87af-e15c-481c-99da-82bb5d81c976.png)

## No Augmentation vs Augmentation : IOU Plots
![Iou_metrics](https://user-images.githubusercontent.com/42332501/175778825-1d5ef7fb-a3e3-4209-9ea9-87c9f2d09370.png)

## No Augmentation vs Augmentation : Loss Plots
![Loss_metrics](https://user-images.githubusercontent.com/42332501/175778893-ac44e708-1dd7-414d-89df-d673fb6c364e.png)

## 8.Results

![Results](https://user-images.githubusercontent.com/42332501/175779006-ef5659f4-cc2f-490a-ae2e-d8c496cc0b02.png)

**Extraction of road path**

![final_image_1](https://user-images.githubusercontent.com/42332501/175779080-9463099f-fb6f-424e-91ed-7714b226cd5c.png)

![final_image_2](https://user-images.githubusercontent.com/42332501/175779087-3a25503f-9d29-487f-8300-7ee494c00d48.png)

![final_image_3](https://user-images.githubusercontent.com/42332501/175779095-6e893284-8257-4101-add6-7b829f227db5.png)

## 9.Video Demonstration

https://user-images.githubusercontent.com/42332501/175374418-d05fe88d-0561-4ba7-8d99-a13c6c2aeee4.mp4

## 10.Medium link
https://medium.com/@yeshwanth2424/deep-globe-road-extraction-dd309c076af9

## 11. LinkedIn profile link
https://www.linkedin.com/in/yeshwanth-sai-278b59186/
