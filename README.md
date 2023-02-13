# flowers

*What flower are you looking at?*

---

### Dataset
The dataset comes from the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
* There are 8,189 images for 102 flower categories. Some categories have more images than the others. There are a minimum of 40 images for each category.
* I followed the data specification to split the data into train, validation, and test sets. The train set has 10 images for each flower category, meaning 1020 images in total. Same with the validation set. The remaining 6,149 images belong to the test set.
More details about how I did data exploration can be found in this [notebook](https://github.com/hanh-nguyen/flowers/blob/main/Data_Exploratory_and_Preparation.ipynb).


### Model Development

* __Flower Detection__:  
#TODO: we need a model to identify if the image contains a flower or not.

* __Flower Classification__:   
    * I trained Convolutional Neural Network (CNN) to predict flower category. 

    * I built a CNN model from scratch. This model had 3 convolutional blocks, each block used convolution, dropout, batch normalization and max pooling layers. Flattening and fully connected layers were added for classification. Its test accuracy rate was 15.69% (16 times better than a baseline model making random predictions from 102 flower categories).   

    * I used transfer learning to build models. This model used the pre-trained `Resnet50` model as a fixed feature extractor, where the last convolutional output of `Resnet50` is fed as input to the model. I added an average pooling layer followed by dropout, batch normalization, and fully connected layers. The test accuracy rate is 80.96%. This result is great comparing to other models below.

    * [A. Gurnani](https://arxiv.org/abs/1708.03763) et al developed two models that used CNN GoogleNet and AlexNet. Their accuracies are 47.15% and 43.39% respectively.

    * 
|          Model           | Train accuracy | Val accuracy | Test accuracy |
| :----------------------: | :------------: | :----------: | :-----------: |
| Random model             |      0.98%     |     0.98%    |     0.98%     |
| CNN from scratch         |     15.69%     |    15.69%    |    15.69%     |
| CNN w/ Resnet50          |     80.96%     |    80.96%    |    80.96%     |

* __Weaknesses and improvement for the model with transfer learning from Resnet50__
    * The model is used to classify a flower category from an image that has one flower in it. We need a model to identify if an image has a flower in it, or better yet, a model to be able to tell how many flowers in an image.
    * The model shows some signs of overfitting. We want to introduce more regularization, including dropout.
    * The model has an accuracy rate of 81%. In terms of categorical level, 26 categories have accuracy rate less than 75% and the lowest accuracy rate is 30%. We can try data augmentation, meaning applying random transformations such as rotation, flip, crop, brightness or contrast on the training data.
    * 
