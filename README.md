# flowers

*What flower are you looking at?*

---

### Dataset
The dataset comes from the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
* There are 8,189 images from 102 flower categories with 40–258 images per category of various image sizes.
* I followed the data specification to split the data set into train, validation, and test sets. The train set has 10 images for each flower category, meaning 1020 images in total. Same with the validation set. The remaining 6,149 images belong to the test set.
* More details about how I did data exploration can be found in the  [Data_Exploration_and_Preparation Notebook](https://github.com/hanh-nguyen/flowers/blob/main/Data_Exploratory_and_Preparation.ipynb).


### Model Development

* __Flower Detection__:  
`#TODO`: we need a model to identify if the image contains a flower or not.

* __Flower Classification__:   
    * I trained a Convolutional Neural Network (CNN) model to predict the flower category. 

    * I built a [CNN model from scratch](https://github.com/hanh-nguyen/flowers/blob/main/CNN_from_scratch.ipynb). This model had 3 convolutional blocks, each block consisted of a convolutional layer with ReLu, followed by dropout, batch normalization and max pooling layers. Flattening and fully connected layers were added for classification. Its test accuracy rate was 15.69% (16 times better than a baseline model making random predictions from 102 flower categories).   

    * I used transfer learning to build an [enhanced model](https://github.com/hanh-nguyen/flowers/blob/main/ResNet_weights.ipynb). This model used the pre-trained Resnet-50 model as a fixed feature extractor, where the last convolutional output of Resnet50 is fed as input to the model. I added an average pooling layer followed by dropout, batch normalization, and fully connected layers. The test accuracy rate is 80.96%. This result is within range when compared to other models below.

    * [A. Gurnani et al](https://arxiv.org/abs/1708.03763) developed two models that used CNN GoogleNet and AlexNet. Their accuracies are 47.15% and 43.39% respectively.

    * [H. Hiary et al](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-cvi.2017.0155) proposed a two step approach. The first step was to localize the flower by detecting the minimum bounding box around it, which used a fully convolutional network initialized by the VGG-16 model. The second step developed a CNN model initialized by the first model. They also used data augmentation. The accuracy with and without augmentation were 99% and 95.4% respectively.


|          Model           | Train accuracy | Val accuracy | Test accuracy |
| :----------------------: | :------------: | :----------: | :-----------: |
| Random model             |      0.98%     |     0.98%    |     0.98%     |
| CNN from scratch         |     84.80%     |    17.75%    |    14.07%     |
| CNN with Resnet-50       |       100%     |    84.71%    |    82.60%     |

* __Weaknesses and improvement for the model with transfer learning from Resnet50__
    * The model has an accuracy rate of 81%. In terms of categorical level, 20 categories have accuracy rate less than 75% and the lowest individual category accuracy rate is 35%.

    * I looked into three flower categories that had the lowest model accuracy: canterbury bells, siam tulip, and petunia, which were predicted mostly incorrectly as monkshood, sweet pea, azalea respectively. Their images present challenges that could be difficult for human eyes as well as the models.
        1. A flower category can have different colors. For example, canterbury bells can be purple, pink, or white; sweet pea can have a mix of white, pink, purple and orange.
        2. Two different categories can share similar colors or shapes. For example, canterbury bells and monkshood, or petunia and azalea.
        3. Surrounding objects such as grass or leaves

    * The model shows some signs of overfitting. We want to introduce more **regularization**.

    * To improve model accuracy, we can collect more images or try **data augmentation**, meaning applying random transformations such as rotation, flip, crop, brightness or contrast on the training data. However, given the similarity among flowers, it is helpful to have some flower expertise to choose useful transformations.

    * The model used Resnet-50 for transfer learning. Resnet-50 was trained on the ImageNet dataset which has very few flower types. We could use a **different pre-trained network**, for example Inception V3, which was trained on the iNaturalist (iNat) 2017 dataset of over 5,000 different species of plants and animals.

    * In this model, we choose to freeze all the layers from Resnet-50. We could try to **fine-tuning**, which consists of unfreezing the entire model we obtained above (or part of it), and re-training it on the new data with a very low learning rate. This can potentially achieve meaningful improvements, by incrementally adapting the pretrained features to the new data.
    
    * The model is used to classify a flower category from an image that has one flower in it. We need a **detection model** to identify if an image has a flower in it, or better yet, a model to be able to tell how many flowers are in an image. Being able to detect a flower also helps to detect the region around the flower in an image, and then uses the cropped images to build the classifier.

