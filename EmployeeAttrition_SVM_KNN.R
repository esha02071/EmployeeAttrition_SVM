#Canterra Employee Attrition




options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("vctrs")
library(caret) #For confusionMatrix(), training ML models, and more
library(class) #For knn()
library(dplyr) #For some data manipulation and ggplot
library(pROC)  #For ROC curve and estimating the area under the ROC curve

library(ggplot2)
library(tidyr)
library(ROSE) #For balancing the target 

library(e1071) #For svm() function
library(pROC)  #For ROC curve and estimating the area under the ROC curve
library(fastDummies) #To create dummy variable (one hot encoding)
library(tidyr)

#library(dplyr)
#library(corrplot)
#library(tidyverse)
# library(scales)

#The dataset has 4410obs of 18 columns, 17 predictors and the last column as target: "Attrition"
#Categorical variables are read asFactors
edata = read.csv("EmployeeData.csv", stringsAsFactors = TRUE)
str(edata)
#summary(edata)


#Distinct Row Check
duplicates <- edata[which(duplicated(edata)), ]
if ((nrow(duplicates)) == 0) {
  print("No duplicates detected") 
}

#Check for columns/variables with missing values
sapply(edata, function(x){sum(is.na(x))})
sum(is.na(edata))


#Target variable has a moderate degree of imbalance NO:83.87% and YES:16.12%
(table(edata$Attrition))
prop.table(table(edata$Attrition))


#Retrive Categorical/Factor Variables
cat_vars <- edata %>% select_if(is.factor)

#Distributions of Categorical Variables
prop.table(table(edata$BusinessTravel))
prop.table(table(edata$Gender))
prop.table(table(edata$MaritalStatus))


#Include Attrition in Corr
#library(tidyr)
ex_att <- as.numeric(edata$Attrition) - 1
edata_corr <- edata
edata_corr$Attrition <- as.numeric(edata$Attrition) - 1

corr_df <- edata_corr %>% select_if(is.numeric)

#Creates dummy variable, removes the column used to generate the dummy columns and first dummy of the variable
edata_dummies = dummy_cols(edata, select_columns = c('BusinessTravel'),
                           remove_selected_columns = T,
                           remove_first_dummy = T)

#Remove variables that we will not use for distance calculation. Dataset to be used for data partition is: employee
employee = 
  edata_dummies %>% select(-c(EmployeeID, StandardHours, Gender, MaritalStatus, Education, JobLevel, Age))


# Create data partition
set.seed(123)  # Set a seed for reproducibility
index = sample(nrow(employee),0.7*nrow(employee))

train_data = employee[index, ]
test_data = employee[-index, ]


#Resampling to address class imbalance 

# library(ROSE)
set.seed(123)
train_data = ovun.sample(Attrition ~ ., data = train_data, method = "over")$data
prop.table(table(train_data$Attrition))



# Plot new variable: we now have a better ratio of quality category
employee  %>% ggplot(aes(x = Attrition))+ geom_bar(width = 0.2, fill="#61679a")+
  labs(title = "Attrition Distribution", x = "Attrition", y = "Count")

#Check the ratio of yes and no in the target variable
prop.table(table(employee$Attrition)) #Target is unbalanced


####Include code in APPENDIX######
#Check for columns/variables with missing values
sapply(train_data, function(x){sum(is.na(x))})
sapply(test_data, function(x){sum(is.na(x))})

sum(is.na(train_data))
sum(is.na(test_data))

#what fraction is missing in target ?
sum(is.na(train_data$Attrition))/nrow(train_data)
sum(is.na(test_data$Attrition))/nrow(test_data)

#No missing data in the train_data.
#Replace missing values in test_data with the mean in train_data
test_data[is.na(test_data$TotalWorkingYears),'TotalWorkingYears'] = mean(train_data$TotalWorkingYears,na.rm=TRUE) 
test_data[is.na(test_data$NumCompaniesWorked),'NumCompaniesWorked'] = mean(train_data$NumCompaniesWorked,na.rm=TRUE) 
test_data[is.na(test_data$JobSatisfaction),'JobSatisfaction'] = mean(train_data$JobSatisfaction,na.rm=TRUE) 
test_data[is.na(test_data$EnvironmentSatisfaction),'EnvironmentSatisfaction'] = mean(train_data$EnvironmentSatisfaction,na.rm=TRUE) 


#Using the classical approach to perform Scaling and use it in the first kNN() model.

#We create different datasets for predictors and the target variable.
#We use information from train set to scale test dataset.
train_x = scale(train_data[, -1]) 
test_x = scale(test_data[,-1],center = apply(train_data[,-1],2,mean),
               scale = apply(train_data[,-1],2,sd))

#Create the target variable in data train and data test
train_y = train_data$Attrition 
test_y = test_data$Attrition 


#Common practice to begin the `k` value using the formula
#train_x is the scaled dataset
k = sqrt(nrow(train_x)) 
k 

#Using function knn() from the package 'class' to run the model
#For KNN, prediction is automatically indicated in modeling
#Scaling is required to use the knn(). Data scaled in lines 275-276.

set.seed(123) #Seed must be set in order to ensure reproducibility of results and R will randomly break the tie. 
model_knn = knn(train=train_x, test=test_x, cl=train_y, k=71) 
model_knn #to see the predicted classes for the test dataset

#knn() Model Evaluation
#ConfusionMatrix() function from the 'caret' package
confusionMatrix(data = model_knn, 
                reference = test_y,
                positive = "Yes")

####Accuracy for different values of k

#Create output vector with some rows and columns (to save output of different values of k)
#Create a loop to go through different values of k and capture the accuracy from the confusion matrix and store results in output
#Let's do it for values of k between 1 and 50, total of 50 values
output = matrix(ncol=2, nrow=50)

for (k_val in 1:50){
  
  #Storing predicted values for each run of the loop (i.e., each value of k)
  set.seed(123)
  temp_pred = knn(train = train_x
                  , test = test_x
                  , cl = train_y
                  , k = k_val)
  
  #Calculate performance measures for the given value of k
  temp_eval = confusionMatrix(temp_pred, test_y) 
  temp_acc = temp_eval$overall[1]
  
  #Add the calculated accuracy as a new row in the output matrix
  output[k_val, ] = c(k_val, temp_acc) 
  
}

#Convert the output to a data frame and plot the results
output = as.data.frame(output)
names(output) = c("K_value", "Accuracy")

# Plot to find out that value of k that gives the highest accuracy
ggplot(data=output, aes(x=K_value, y=Accuracy, group=1)) +
  geom_line(color="#61679a")+
  geom_point(color="#61679a")+
  theme_bw()

#Run the model and store in model_1. Print OUTPUT
#To get the predicted class/category for each data in test
set.seed(123)
model_1 = knn(train=train_x, test=test_x, cl=train_y, k=71)

#To get the predicted probability of belonging to each class/category
set.seed(123)
model_1_probs = attributes(knn(train=train_x, test=test_x, cl=train_y, k=71, prob=TRUE))$prob


#Check model evaluation
confusionMatrix(data = model_1, 
                reference = test_y, 
                positive = "Yes")

#Using CARET Package
# Cross validation parameters using trainControl(). Specify 10-fold cross validation
ctrl = trainControl(method="cv",number=10)

# CV KNN model
# train() is a general function from caret package that can handle many methods
# Here we're using "knn" based on the CV parameters that we set earlier
# tuneLength=55 means the function automatically runs/tests for 55 different values of k
set.seed(123)
knn_cv = train(
  Attrition ~ ., data = train_data,  #Using train data because it is not scaled. 
  method = "knn", trControl = ctrl, 
  preProcess = c("center","scale"), 
  tuneGrid = expand.grid(k = 1:20),
  tuneLength = 38)  #

# assess results
knn_cv

# Plot accuracy versus different values of k used by tuneLength
plot(knn_cv, main = "Model 2: kNN caret Performance")


# Plot most important variables
plot(varImp(knn_cv), 5, main = "Most Important Variables for kNN caret (Model 2)") 

#To get the predicted class/category for each data in test
model_2 = predict(knn_cv, test_data, type="raw")

#To get the predicted probability of belonging to each class/category   ###NOT GETTING PROBABILITY
model_2_probs = predict(knn_cv, test_data, type="prob")[,2]

confusionMatrix(data = model_2, 
                reference = test_y, 
                positive = "Yes")

# Tuning SVM Model (searching for the best parameters, running like a loop)

#The e1071 library includes a built-in function, tune(), to perform cross validation. 
#By default, tune() performs ten-fold cross-validation on a set of models of interest. 
#In order to use this function, we pass in relevant information about the set of models that are under consideration. 
#Setting the probability to TRUE provides probability predictions to later use for ROC curve.
set.seed(123)
tune.out = tune(svm, Attrition~., data=train_data, kernel ="linear", probability=TRUE,
                ranges=list(cost=c(0.001,0.01,0.1,1,5,10))) #list of values to search for cost parameters

summary(tune.out)


#The tune() function stores the best model obtained, which can be accessed as follows:
bestmod = tune.out$best.model  #best model, save it to use for prob predictions.
SVM_pred = predict(bestmod, newdata = test_data[-1]) #obtaining predicted classes. Needed for Confusion matrix function.

#Obtaining predicted probabilities (to use in creating ROC curves to compare different models)
SVM_prob = attributes(predict(bestmod, newdata=test_data[-1], probability=TRUE))$probabilities[,1]

# Check model evaluation
confusionMatrix(data = SVM_pred, 
                reference = test_data$Attrition, 
                positive = "Yes")

#Metrics:  Sensitivity : 0.6791, Sensitivity/Recall (how many were correctly classified as true positives?)         
#          Specificity : 0.6245, Specificity is focused on negative class.(how many were correctly classified as true negative?)
#Accuracy : 0.6334   

#e1071
#We can perform cross-validation using tune() to select the best choice of cost and gamma for an SVM with a radial kernel
set.seed(123)
tune.out2 = tune(svm, Attrition ~., data=train_data, kernel ="radial", probability=TRUE,
                 ranges=list(cost=c(0.01,0.1,1,5,10), gamma=c(0.5,1,2,3))) #searching for cost and gamma.
summary(tune.out2)

bestmod2 = tune.out2$best.model
SVMR_pred = predict(bestmod2, newdata = test_data[-1]) #obtaining predicted classes

#Obtaining predicted probabilities
SVMR_prob = attributes(predict(bestmod2, newdata=test_data[-1], probability=TRUE))$probabilities[,1]

#Check model evaluation
confusionMatrix(data = SVMR_pred, 
                reference = test_data$Attrition, 
                positive = "Yes")


# Accuracy : 0.9735
# Sensitivity : 0.8512          
# Specificity : 0.9973 

#The levels of the target variables need to change to "Yes/No" instead of "1/0"!
train_data_copy = train_data
test_data_copy = test_data

levels(train_data_copy$Attrition) = c("No","Yes")
levels(test_data_copy$Attrition) = c("No","Yes")

# We can set our cross validation parameters using trainControl()
# Specify 10-fold cross validation
# classProbs=T makes sure probabilities are also calculated and can be accessed later
ctrl = trainControl(method="cv",number=10,classProbs=TRUE)

# train() is a general function from caret package that can handle many methods
# Here we're using "svmRadial" based on the CV parameters that we set earlier. Other methods are "svmLinear" and "svmPoly"
# By setting tuneLength=10, the function tests 10 different value of C
# Specified 10-fold cross validation

set.seed(123)
SVMR_caret = train(
  Attrition ~ ., data = train_data_copy,
  method = "svmRadial", trControl = ctrl, 
  preProcess = c("center","scale"), tuneLength = 10) #Searching over cost, fixes sigma or gamma.

SVMR_caret
plot(SVMR_caret, main="Support Vector Machine Model 4")

#To get the predicted class/category for each data in test
#Note: When using the train() function from "caret" package, the model automatically chooses the best value of k to use in prediction
SVMR_pred2 = predict(SVMR_caret, test_data_copy[,-1], type="raw")

#To get the predicted probability of belonging to each class/category
SVMR_prob2 = predict(SVMR_caret, test_data_copy[,-1], type="prob")[,2]

confusionMatrix(data = SVMR_pred2, 
                reference = test_data_copy$Attrition, 
                positive = "Yes")


# Accuracy : 0.932  
# Sensitivity : 0.8605         
# Specificity : 0.9458 

#Comparing classifiers

#Using plot.roc() and auc() functions from "pROC" package:

plot.roc(test_y,model_1_probs,col="#ba60ae",legacy.axes=T)
plot.roc(test_y,model_2_probs,add=TRUE,col="#74c07a")
plot.roc(test_data$Attrition,SVM_prob,add=TRUE,col="#257ca3", legacy.axes=T)
plot.roc(test_data$Attrition,SVMR_prob,add=TRUE,col="#e27272",lty=5,legacy.axes=T)
plot.roc(test_data_copy$Attrition,SVMR_prob2,add=TRUE,col="#a060ba")
legend("bottomleft",legend=c("kNN","kNN.CV","SVM(linear)","SVM(radial)-e1071","SVM(radial)-caret"),
       col=c("#ba60ae","#74c07a","#257ca3","#e27272","#a060ba"),lty=c(1,2,3,4,5),cex=0.8,
       text.width = 0.5,  # Adjust the text.width to make the legend text smaller
       text.col = "black")  # Set the text color to black)


#Calculate the area under the ROC curve for the classifiers
auc_knn <- auc(test_y,model_1_probs) #kNN
auc_knn.cv <- auc(test_y,model_2_probs) #kNN caret
auc_SVM.l <- auc(test_data$Attrition,SVM_prob) #SVM Linear
auc_SVM.r <- auc(test_data$Attrition,SVMR_prob) #SVM Radial (good combination radial and cost)
auc_SVM.r.cv <- auc(test_data_copy$Attrition,SVMR_prob2) #SVM Radial "caret"

cat("AUC for kNN:", auc_knn, "\n")
cat("AUC for kNN.caret:", auc_knn.cv, "\n")
cat("AUC for SVM.Linear:", auc_SVM.l, "\n")
cat("AUC for SVM.Radial-e1071:", auc_SVM.r, "\n")
cat("AUC for SVM.Radial.caret:", auc_SVM.r.cv, "\n")

# Area under the curve: 0.5239
# Area under the curve: 0.942
# Area under the curve: 0.7088
# Area under the curve: 0.9802
# Area under the curve: 0.9134

