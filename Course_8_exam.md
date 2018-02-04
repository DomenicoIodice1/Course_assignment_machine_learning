# Practical Machine Learning Course Project
Domenico Iodice  
03 febbraio 2018  



###Background

#####Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.

###Data loading


```r
if (!file.exists("pml-training.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
}
testing <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA"))
training <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA"))
```

#####Exploring data, it is possible to see many columns with "NA" values in both data sets. They will be removed. Test and training data sets will be uniformed to be analyzed.


```r
variables <- names(testing[,colSums(is.na(testing)) == 0])[8:59]
training <- training[,c(variables,"classe")]
testing <- testing[,c(variables,"problem_id")]
```

#####To verify the fitting accurancy of the model which should be used to predict the initial testing data, we create a partition of the cleaned data. 75% of observation are usually used.


```r
set.seed(916)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.4.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
Validation_set <- createDataPartition(training$classe, p = 0.75, list = F)
training2 <- training[Validation_set,]
testing2 <- training[-Validation_set,]
```

###Random Forest Model


```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.4.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
Random_forest_fit <- randomForest(classe~., data = training2)
Random_forest_fit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training2) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.46%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    3    0    0    0 0.0007168459
## B   12 2830    6    0    0 0.0063202247
## C    0   12 2552    3    0 0.0058433970
## D    0    0   21 2389    2 0.0095356551
## E    0    0    1    7 2698 0.0029563932
```

#####OOB eximated error rate of Random Forest model is 0.46% on training data. Applying this model on the partition, it can be possible to use confusion matrixes to explore the accurancy, and correlated error, of the model.


```r
Random_forest_prediction_test <- predict(Random_forest_fit, testing2, type = "class")
confusionMatrix(testing2$classe, Random_forest_prediction_test)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    4  945    0    0    0
##          C    0    6  847    2    0
##          D    0    0    7  797    0
##          E    0    0    0    4  897
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9953        
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.2853        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9941        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9971   0.9937   0.9918   0.9925   1.0000
## Specificity            1.0000   0.9990   0.9980   0.9983   0.9990
## Pos Pred Value         1.0000   0.9958   0.9906   0.9913   0.9956
## Neg Pred Value         0.9989   0.9985   0.9983   0.9985   1.0000
## Prevalence             0.2853   0.1939   0.1741   0.1637   0.1829
## Detection Rate         0.2845   0.1927   0.1727   0.1625   0.1829
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9986   0.9963   0.9949   0.9954   0.9995
```

```r
Random_forest_prediction_train <- predict(Random_forest_fit, training2, type = "class")
confusionMatrix(training2$classe, Random_forest_prediction_train)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

#####The estimated accuracy of the model is 99.53% and the estimated out-of-sample error is 0.47% (100-99.53). The model after this preliminary test will be used on the original cleaned data.


```r
prediction_RF <- predict(Random_forest_fit, testing, type = "class")
prediction_RF
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

