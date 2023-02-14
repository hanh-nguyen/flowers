# flowers

*What flower are you looking at?*

---

### Dataset
The dataset comes from the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
* There are 8,189 images for 102 flower categories. Some categories have more images than the others. There are a minimum of 40 images for each category.
* I followed the data specification to split the data set into train, validation, and test sets. The train set has 10 images for each flower category, meaning 1020 images in total. Same with the validation set. The remaining 6,149 images belong to the test set.
More details about how I did data exploration can be found in the  [Data_Exploration_and_Preparation Notebook](https://github.com/hanh-nguyen/flowers/blob/main/Data_Exploratory_and_Preparation.ipynb).


### Model Development

* __Flower Detection__:  
#TODO: we need a model to identify if the image contains a flower or not.

* __Flower Classification__:   
    * I trained a Convolutional Neural Network (CNN) model to predict the flower category. 

    * I built a CNN model from scratch. This model had 3 convolutional blocks, each block used convolution, dropout, batch normalization and max pooling layers. Flattening and fully connected layers were added for classification. Its test accuracy rate was 15.69% (16 times better than a baseline model making random predictions from 102 flower categories).   

    * I used transfer learning to build an enhanced model. This model used the pre-trained `Resnet50` model as a fixed feature extractor, where the last convolutional output of `Resnet50` is fed as input to the model. I added an average pooling layer followed by dropout, batch normalization, and fully connected layers. The test accuracy rate is 80.96%. This result is great when compared to other models below.

    * [A. Gurnani](https://arxiv.org/abs/1708.03763) et al developed two models that used CNN GoogleNet and AlexNet. Their accuracies are 47.15% and 43.39% respectively.

    * 
|          Model           | Train accuracy | Val accuracy | Test accuracy |
| :----------------------: | :------------: | :----------: | :-----------: |
| Random model             |      0.98%     |     0.98%    |     0.98%     |
| CNN from scratch         |     15.69%     |    15.69%    |    15.69%     |
| CNN with Resnet50          |       100%     |    84.71%    |    82.60%     |

* __Weaknesses and improvement for the model with transfer learning from Resnet50__
    * The model is used to classify a flower category from an image that has one flower in it. We need a model to identify if an image has a flower in it, or better yet, a model to be able to tell how many flowers are in an image.
    * The model has an accuracy rate of 81%. In terms of categorical level, 20 categories have accuracy rate less than 75% and the lowest individual category accuracy rate is 35%.
    * I looked into three flower categories that had the lowest model accuracy: canterbury bells, siam tulip, and petunia, which were predicted mostly incorrectly as monkshood, sweet pea, azalea respectively. Their images present challenges that could be difficult for human eyes.
        1. A flower category can have different colors. For example, canterbury bells can be purple, pink, or white; sweet pea can have a mix of white, pink, purple and orange.
        2. Two different categories can share similar colors or shapes. For example, canterbury bells and monkshood, or petunia and azalea.
    * To improve model accuracy, we can try data augmentation, meaning applying random transformations such as rotation, flip, crop, brightness or contrast on the training data. However, given the similarity among flowers, it is helpful to have some flower expertise to choose useful transformations.
    * The model used Resnet50 for transfer learning. Resnet50 was trained on the ImageNet dataset which has very few flower types. We could use a different network, for example Inception V3, which was trained on the iNaturalist (iNat) 2017 dataset of over 5,000 different species of plants and animals.
    * The model shows some signs of overfitting. We want to introduce more regularization, including dropout.
