library(e1071)
library(lubridate)
library(FSAdata)
library(FSA)
library(magrittr)
library(dplyr)
library(tidyr)
library(tidyverse)
library(data.table)
library(plyr)
library(ggplot2)
library(psych)
library(Hmisc)
library(dplyr)
library(corrplot)
library(caret)
library(olsrr)
library(performance)
library(Ecdat)
library(leaps)
library(lmtest)
library(visdat)
library(inspectdf)
library(skimr)
library(ggplot2)
library(ggcorrplot)
library(car)
library(DataExplorer)
library(ISLR)
library(plotrix)
library(glmnet)
library(cat)
library(mltools)
library(olsrr)
library(MASS)

options(max.print = 20000000)
pd <- read.csv(file.choose( ), header = TRUE)
head(pd)

#investigate on satisfaction and customer type
#factors that influence satisfaction
#Type of travel preferred
#class of travel preferred
#factor that influences satisfaction(satisfied) and customer type(disloyal)
#flight and airport having max delays

#Questions 
#1. What is the average satisfaction level of customers?
#2. What is the average customer loyalty ratio of customers?
#3. What are the top three factors affecting the customer satisfaction?
#4. What are the top three factors affecting the customer loyalty?
#5. What type of travel is frequently preferred by the customers?
#6. What type of class is frequently preferred by the customers?
#7. To investigate the delay parameters and find out the one affecting the most delays.

#EDA
summary(pd)
dim(pd)
glimpse(pd)
ls(pd)
vis_dat(pd, palette = "cb_safe", warn_large_data = FALSE)
skim(pd)

#Showing Mission Rows
x <-inspect_imb(pd)   
show_plot(x)  

#Check NAs values
z <-inspect_na(pd) 
show_plot(z) 

#Removing rows with na values
pd1 <- pd                                    
pd1[pd1 == ""] <- NA
pd1 <- na.omit(pd1)
pd1
head(pd1)
dim(pd1)
summary(pd1)


#Question 1 : Average satisfaction
satisfaction <- pd1$satisfaction
satisfaction
y = 0
z = 0
for (x in satisfaction) {
  if(x == "satisfied"){
    y = y+1
  } else {
    z=z+1
  }
}
y
z

satisfied_cust <- round(y/(y+z) * 100,2)
satisfied_cust
dissatisfied_cust <- round(z/(y+z) * 100,2)
dissatisfied_cust

#Visualization 
slices <- c(satisfied_cust,dissatisfied_cust)
lbls <- c("Satisfied - 54.74%", "Dissatisfied - 45.26%")
# pie3D(slices, labels = lbls, main="Pie Chart of Satisfaction level of customers",explode=0.1, col=c("cornflowerblue", "gray"))

#Question 2 : Average Loyalty
loyalty <- pd1$Customer.Type
loyalty
a = 0
b = 0
for (x in loyalty) {
  if(x == "Loyal Customer"){
    a = a+1
  } else {
    b = b+1
  }
}
a
b

loyal_cust <- round(a/(a+b) * 100,2)
loyal_cust
disloyal_cust <- round(b/(a+b) * 100,2)
disloyal_cust

#********************REGULARIZATION************************************************************
#Splitting the dataset

x=0
for (x in 1:129487) {
  if(pd1$satisfaction[x] == "satisfied"){
    pd1$satisfaction[x] = 1
  } else {
    pd1$satisfaction[x] = 0
  }
}

pd1$satisfaction
pd1$satisfaction <- as.integer(pd1$satisfaction)
glimpse(pd1)
head(pd1$satisfaction)


set.seed(1)
Sample <- sample(nrow(pd1), nrow(pd1)*.7) 
trainData <- pd1[Sample, ] 
head(trainData)
testData <- pd1[-Sample, ] 
head(testData)


X <- model.matrix(satisfaction ~.,trainData)[,-1]
X
X1 <- model.matrix(satisfaction ~.,testData)[,-1]
X1
Y <- trainData$satisfaction
Y1 <- testData$satisfaction
Y1

# RIDGE -- start
#Defining Predictor and Response Varibales
#Training Model
set.seed(150)
lambda <- 10^seq(2,-2, by = -.1)
ridgeModelTrain <- cv.glmnet(X, Y, alpha = 0, lambda = lambda) 
trainLamda.Min1 <- ridgeModelTrain$lambda.min 
trainLambda.1se1 <- ridgeModelTrain$lambda.1se 
data.frame(trainLamda.Min1,trainLambda.1se1 )

# Displaying regression coefficients
coef(ridgeModelTrain)
trainingDataPrediction <- predict(ridgeModelTrain, s = lambda, newx = X) 

plot(ridgeModelTrain)
abline(v=ridgeModelTrain$lambda.min, col = "blue", lty=2)
abline(v=ridgeModelTrain$lambda.1se, col="grey", lty=2)

#Calculating RMSE - Training Data
RMSEtrainingData <- sqrt(mean((Y - trainingDataPrediction)^2)) 
RMSEtrainingData

#Testing Model
set.seed(250)
lambda <- 10^seq(2,-2, by = -.1)
ridgeModelTest <- cv.glmnet(X1, Y1, alpha = 0, lambda = lambda) 
testLamda.Min1 <- ridgeModelTest$lambda.min 
testLambda.1se1 <- ridgeModelTest$lambda.1se 
data.frame(testLamda.Min1,testLambda.1se1 )

# Displaying regression coefficients
coef(ridgeModelTest)
testingDataPrediction <- predict(ridgeModelTest, s = lambda, newx = X1) 

plot(ridgeModelTest)
abline(v=ridgeModelTest$lambda.min, col = "blue", lty=2)
abline(v=ridgeModelTest$lambda.1se, col="green", lty=2)

#Calculating RMSE - Training Data
RMSEtestingData <- sqrt(mean((Y1 - testingDataPrediction)^2)) 
RMSEtestingData

# RIDGE -- End

# LASSO -- start
#Training Model
set.seed(150)
lambda1 <- 10^seq(2,-2, by = -.1)
LassoModelTrain <- cv.glmnet(X, Y, alpha = 1, lambda = lambda) 
trainLamda.Min1_Lasso <- LassoModelTrain$lambda.min 
trainLambda.1se1_Lasso <- LassoModelTrain$lambda.1se 
data.frame(trainLamda.Min1_Lasso,trainLambda.1se1_Lasso )

# Displaying regression coefficients
coef(LassoModelTrain)
plot(LassoModelTrain)
abline(v=LassoModelTrain$lambda.min, col = "yellow", lty=2)
abline(v=LassoModelTrain$lambda.1se, col="red", lty=2)

# Train set predictions using "lambda.1se"
trainingDataPrediction1 <- predict(LassoModelTrain, s = lambda, newx = X) 

#Calculating RMSE - Training Data
RMSEtrainingData1 <- sqrt(mean((Y - trainingDataPrediction1)^2)) 
RMSEtrainingData1

#Testing Model
set.seed(150)
lambda1 <- 10^seq(2,-2, by = -.1)
LassoModelTest <- cv.glmnet(X1, Y1, alpha = 1, lambda = lambda) 
testLamda.Min1_Lasso <- LassoModelTest$lambda.min 
testLambda.1se1_Lasso <- LassoModelTest$lambda.1se 
data.frame(testLamda.Min1_Lasso,testLambda.1se1_Lasso )

# Displaying regression coefficients
coef(LassoModelTest)

plot(LassoModelTest)
abline(v=LassoModelTest$lambda.min, col = "orange", lty=2)
abline(v=LassoModelTest$lambda.1se, col="green", lty=2)

# Test set predictions using "lambda.1se"
testingDataPrediction1 <- predict(LassoModelTest, s = lambda, newx = X1) 

#Calculating RMSE - Testing Data
RMSEtestingData1 <- sqrt(mean((Y1 - testingDataPrediction1)^2)) 
RMSEtestingData1

# LASSO -- End
#*#******************************************************************************************

#Visualization 
slices <- c(loyal_cust,disloyal_cust)
lbls <- c("Loyal - 81.69%", "Disloyal - 18.31%")
pie(slices, labels = lbls, main="Pie Chart of Loyalty of Customers", col=c("aquamarine3", "cadetblue3"))


#Question 3 : Top 3 factors affecting customer satisfaction
x=0
for (x in 1:129487) {
  if(pd1$satisfaction[x] == "satisfied"){
    pd1$satisfaction[x] = 1
  } else {
    pd1$satisfaction[x] = 0
  }
}

pd1$satisfaction
pd1$satisfaction <- as.integer(pd1$satisfaction)
glimpse(pd1)
pd1$satisfaction


X <- data.matrix(pd1[, variableNames]) 
head(X)
Y <- pd1[, "satisfaction"] 
head(Y)
set.seed(150)
lambda <- 10^seq(2,-2, by = -.1)
ridgeModelTrain <- cv.glmnet(X, Y, alpha = 0, lambda = lambda) 
Lamda.Min1 <- ridgeModelTrain$lambda.min 
Lambda.1se1 <- ridgeModelTrain$lambda.1se 
data.frame(Lamda.Min1,Lambda.1se1 )

pd1_cor <- cor(pd1[sapply(pd1, is.numeric)])
pd1_cor
#Visualization
par(mfrow=c(1,1))
corrplot(pd1_cor, title= "Correlation Matrix for Invistico Airlines", order="hclust", mar=c(0,0,1,0), addrect = 1)
col<- colorRampPalette(c("cadetblue1", "white", "cadetblue4"))(20)
heatmap(pd1_cor, col = col,symm = TRUE,main="Heatmap for Invistico Airlines")

#Top 3 
#1. Inflight.entertainment
#2. Ease.of.Online.booking
#3. Online.support
#Last 3
#1.Arrival.Delay.in.Minutes
#2.Departure.Delay.in.Minutes
#3.Flight.Distance


#Question 4 : Top 3 factors affecting customer loyalty
x=0
for (x in 1:129487) {
  if(pd1$Customer.Type[x] == "Loyal Customer"){
    pd1$Customer.Type[x] = 1
  } else {
    pd1$Customer.Type[x] = 0
  }
}
pd1$Customer.Type
pd1$Customer.Type <- as.integer(pd1$Customer.Type)
glimpse(pd1)

pd1_cor <- cor(pd1[sapply(pd1, is.numeric)])
pd1_cor

#Top 3
#1.Age 
#2.Inflight.entertainment
#3.satisfaction
#Last 3
#1. Flight.Distance
#2. Arrival.Delay.in.Minutes
#3. Departure.Delay.in.Minutes

#Visualization
par(mfrow=c(1,1))
corrplot(pd1_cor, title= "Correlation Matrix for Invistico Airlines", order="hclust", mar=c(0,0,1,0), addrect = 1)
col<- colorRampPalette(c("cadetblue1", "white", "cadetblue4"))(20)
heatmap(pd1_cor, col = col,symm = TRUE,main="Heatmap for Invistico Airlines")

#Question 8 : Linear Regression
pd1model <- lm(satisfaction ~ Inflight.entertainment+Ease.of.Online.booking+Online.support, data = pd1)
pd1model

summary(pd1model)
summary(pd1model)$coefficient

vif(pd1model)

#Question 5 : Type of Travel mostly preferred
df_uniq <- unique(pd1$Type.of.Travel)
length(df_uniq)

s=0
f=0
g=0
for (s in 1:129487) {
  if(pd1$Type.of.Travel[s] == "Personal Travel"){
    f = f+1
  } else {
    g = g+1
  }
}
f
g
#Business travel > Personal Travel

#Question 6 : Class preferred by customers
s=0
d=0
e=0
f=0
for (s in 1:129487) {
  if(pd1$Class[s] == "Eco"){
    d = d+1
  } else if(pd1$Class[s] == "Business"){
    e = e+1
  }
  else
  { f = f+1}
}
d
e
f
#Business > Eco > Eco Plus

#Question 7 : Hypothesis testing
s=0
a=0
b=0
c=0
d=0
for (s in 1:129487) {
  if(pd1$Customer.Type[s] == 0 && pd1$satisfaction[s] == 0){
    
    a= a+ 1
  }
  else  if(pd1$Customer.Type[s] == 0 && pd1$satisfaction[s] == 1){
    b= b+ 1
  }
  else  if(pd1$Customer.Type[s] == 1 && pd1$satisfaction[s] == 1){
    c= c+ 1
  }
  else  {
    d=d+ 1
  }
}
a
b
c
d

t_loyal <- c+d
t_loyal
t_disloyal <- a+b
t_disloyal
t_satisfied <- c+b
t_satisfied
t_dissatisfied <- a+d
t_dissatisfied
total <- t_satisfied + t_dissatisfied
total
exp_satisfiedL <- (t_loyal/total)*t_satisfied
exp_satisfiedL
exp_satisfiedD <- (t_disloyal/total)*t_satisfied
exp_satisfiedD
exp_dissatisfiedL <- (t_loyal/total)*t_dissatisfied
exp_dissatisfiedL
exp_dissatisfiedD <- (t_disloyal/total)*t_dissatisfied
exp_dissatisfiedD

obs = c(65194,5688,40579,18026)
exp = c(exp_satisfiedL,exp_satisfiedD,exp_dissatisfiedL,exp_dissatisfiedD)
obs
exp

#DF
df <- data.frame(obs,exp)
df

#Critical Value
z3 <- qchisq(0.05,3,lower.tail = FALSE)
z3

# Step 3: Calculating test-value:p-value: 
Test1 <- chisq.test(df) 
Test1
Test1$statistic
Test1$p.value
Test1$parameter
