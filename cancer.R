rm(list=ls())
setwd("C:/Users/Garima/Documents/R/cancer")
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('grid') # visualisation
library('gridExtra') # visualisation
library('corrplot') # visualisation
library('ggfortify') # visualisation
library('ggraph') # visualisation
library('igraph') # visualisation
library('dplyr') # data manipulation
library('readr') # data input
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('tidytext') # text mining
library('SnowballC') # text analysis
library('wordcloud') # test visualisation
library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(stringr)
library(tm)
library(syuzhet) 
library(sqldf)
library(caret)
library(e1071)
# Load Data ---------------------------------------------------------------
memory.limit(size=5000000)
path <- "C:/Users/Garima/Documents/R/cancer"
data <- fread(file.path(path, "data.csv"))
train_text <- do.call(rbind,strsplit(readLines('training_text'),'||',fixed=T))
train_text <- as.data.table(train_text)
train_text=train_text[-1,]
test_text <- do.call(rbind,strsplit(readLines('test_text'),'||',fixed=T))
test_text <- as.data.table(test_text)
test_text=test_text[-1,]
training_variants <- fread(file.path(path, "training_variants"))
test_variants <- fread(file.path(path, "test_variants"))
test_variants$Class=0
colnames(train_text) <- c("ID", "Text")
colnames(test_text) <- c("ID", "Text")
test_text$ID=as.integer(test_text$ID)
train_text$ID=as.integer(train_text$ID)
train <- merge(training_variants,train_text,by="ID")
test <- merge(test_variants,test_text,by="ID")
data <- rbind(train,test)
rm(test_text,test_variants,training_variants,train_text)
gc()

# Basic text features
#level-wise
Class1=sqldf("Select * from data where Class='1'")
Class2=sqldf("Select * from data where Class='2'")
Class3=sqldf("Select * from data where Class='3'")
Class4=sqldf("Select * from data where Class='4'")
Class5=sqldf("Select * from data where Class='5'")
Class6=sqldf("Select * from data where Class='6'")
Class7=sqldf("Select * from data where Class='7'")
Class8=sqldf("Select * from data where Class='8'")
Class9=sqldf("Select * from data where Class='9'")
write.csv(Class1, file = 'class1.csv')
write.csv(Class2, file = 'class2.csv')
write.csv(Class3, file = 'class3.csv')
write.csv(Class4, file = 'class4.csv')
write.csv(Class5, file = 'class5.csv')
write.csv(Class6, file = 'class6.csv')
write.csv(Class7, file = 'class7.csv')
write.csv(Class8, file = 'class8.csv')
write.csv(Class9, file = 'class9.csv')
write.csv(data, file = 'data.csv')

summary(data)
dt=data[data$nwords==1]
dt

# TF-IDF
txt <- Corpus(VectorSource(data$Text))
txt <- tm_map(txt, stripWhitespace)
txt <- tm_map(txt, content_transformer(tolower))
txt <- tm_map(txt, removePunctuation)
txt <- tm_map(txt, removeWords, c(stopwords("english"),"fig", "figure", "et", "al", "table",
                                  "observ","interact","signific","type","express","associ","known",
                                  "describ","chang","site","suggest","contain","data", "analysis", "analyze", "study",
                                  "method", "result", "conclusion", "author",
                                  "find", "found", "show", "perform",
                                  "demonstrate", "evaluate", "discuss","use","also","may","shown","howev"))
txt <- tm_map(txt, stemDocument, language="english")
txt <- tm_map(txt, removeWords, c("fig", "figure", "et", "al", "table",
                                  "observ","interact","signific","type","express","associ","known",
                                  "describ","chang","site","suggest","contain","data", "analysis", 
                                  "analyze", "study",
                                  "method", "result", "conclusion", "author",
                                  "find", "found", "show", "perform",
                                  "demonstrate", "evaluate", "discuss","use","also","may","shown","howev"))
txt <- tm_map(txt, removeNumbers)

dtm <- DocumentTermMatrix(txt)
dtm1=dtm
dtm2=dtm
dtms <- removeSparseTerms(dtm, .90)
dtma <- colSums(as.matrix(dtms))
#dt=as.matrix(dtms)
#dtm <- as.matrix(dtm)
write.csv(dtma, file = 'matrix.csv')
#write.csv(dt, file = 'matrix1.csv')
wf <- data.frame(word = names(dtma), freq = dtma)
chart <- ggplot(subset(wf, freq >80000), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart
assocs=findAssocs(dtm, c('mutat','cell'), corlimit=0.30)
set.seed(142)
wordcloud(names(dtma), dtma, min.dtma = 2500, scale = c(6, .1), colors = brewer.pal(4, "BuPu"))
wordcloud(names(dtma), dtma, max.words = 5000, scale = c(6, .1), colors = brewer.pal(6, 'Dark2'))
newsparse <- as.data.frame(as.matrix(dtms))
dim(newsparse)
colnames(newsparse) <- make.names(colnames(newsparse))
newsparse$Class <- as.factor(c(train$Class, rep('7', nrow(test))))
newsparse$Gene=as.factor(data$Gene)
newsparse$Variation=as.factor(data$Variation)
newsparse$Gene=as.factor(newsparse$Gene)
newsparse$Variation=as.factor(newsparse$Variation)
mytrain1 <- newsparse[1:3000,]
mytest1 <- newsparse[3001:3321,]
write.csv(newsparse, file = 'newsparse.csv')

#Divide training data into test and train
test2 <- train[3001:3321,]
#test1=rbind(test1,test)
ctrain1 <- xgb.DMatrix(Matrix(data.matrix(mytrain1[,!colnames(mytrain1) %in% c('Class')])), label = as.numeric(mytrain1$Class)-1)
dtest1 <- xgb.DMatrix(Matrix(data.matrix(mytest1[,!colnames(mytest1) %in% c('Class')]))) 
rm(fin,final_submit1,submit_match11,submit_match12,submit_match13)

xgbmodel1 <- xgboost(data = ctrain1, objective = "multi:softmax",eta=0.3,max.depth=25,verbose = 1,num_class = 20, nround = 200)
xgbmodel12 <- xgboost(data = ctrain1, nrounds = 250,  objective = "multi:softmax",eta= 0.2, max_depth = 20,verbose = 1,num_class = 20)
xgbmodel13 <- xgboost(data = ctrain1,objective= "multi:softmax",eta= 0.1,max_depth= 20,verbose= 2,min_child_weight= 2,num_class= 20,gamma= 2, nround = 250)

#predict 1
xgbmodel.predict1 <- predict(xgbmodel1, newdata = data.matrix(mytest1[, !colnames(mytest1) %in% c('Class')]))
xgbmodel.predict.text1 <- levels(mytrain1$Class)[xgbmodel.predict1 + 1]
#predict 2
xgbmodel.predict12 <- predict(xgbmodel12, newdata = data.matrix(mytest1[, !colnames(mytest1) %in% c('Class')])) 
xgbmodel.predict2.text1 <- levels(mytrain1$Class)[xgbmodel.predict12 + 1]
#predict 3
xgbmodel.predict13 <- predict(xgbmodel13, newdata = data.matrix(mytest1[, !colnames(mytest1) %in% c('Class')])) 
xgbmodel.predict3.text1 <- levels(mytrain1$Class)[xgbmodel.predict13 + 1]

#data frame for predict 1
submit_match11 <- cbind(as.data.frame(test2$ID), as.data.frame(xgbmodel.predict.text1))
colnames(submit_match11) <- c('ID','Class')
submit_match11 <- data.table(submit_match11, key = 'ID')
#data frame for predict 2
submit_match12 <- cbind(as.data.frame(test2$ID), as.data.frame(xgbmodel.predict2.text1))
colnames(submit_match12) <- c('ID','Class')
submit_match12 <- data.table(submit_match12, key = 'ID')
#data frame for predict 3
submit_match13 <- cbind(as.data.frame(test2$ID), as.data.frame(xgbmodel.predict3.text1))
colnames(submit_match13) <- c('ID','Class')
submit_match13 <- data.table(submit_match13, key = 'ID')

sum(diag(table(mytest1$Class, xgbmodel.predict1)))/nrow(mytest1) 
sum(diag(table(mytest1$Class, xgbmodel.predict12)))/nrow(mytest1)
sum(diag(table(mytest1$Class, xgbmodel.predict13)))/nrow(mytest1)

#ensembling 
submit_match13$Class2 <- submit_match12$Class 
submit_match13$Class1 <- submit_match11$Class

#function to find the maximum value row wise
Mode <- function(x) {
  u <- unique(x)
  u[which.max(tabulate(match(x, u)))]
}
x1 <- Mode(submit_match13[,c("Class","Class2","Class1")])
y1 <- apply(submit_match13,1,Mode)
final_submit1 <- data.frame(id= submit_match13$ID, Class = submit_match13$Class2)
write.csv(final_submit1, 'ensembletry.csv', row.names = FALSE)
fin=final_submit1[1:321,]
x <- as.factor(fin$Class)
y <- test2$Class
l <- union(x, y)
Table2 <- table(factor(x, l), factor(y, l))
xtab=table(observed=test2$Class, predicted=fin$Class)
confusionMatrix(Table2)


#Predicting data for test data
mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]
ctrain <- xgb.DMatrix(Matrix(data.matrix(mytrain[,!colnames(mytrain) %in% c('Class')])), label = as.numeric(mytrain$Class)-1)
dtest <- xgb.DMatrix(Matrix(data.matrix(mytest[,!colnames(mytest) %in% c('Class')]))) 

xgbmodel <- xgboost(data = ctrain, objective = "multi:softmax",eta=0.3,max.depth=25,verbose = 1,num_class = 20, nround = 200)
xgbmodel2 <- xgboost(data = ctrain, nrounds = 250,  objective = "multi:softmax",eta= 0.2, max_depth = 20,verbose = 1,num_class = 20)
xgbmodel3 <- xgboost(data = ctrain,objective= "multi:softmax",eta= 0.1,max_depth= 20,verbose= 2,min_child_weight= 2,num_class= 20,gamma= 2, nround = 250)

#predict 1
xgbmodel.predict <- predict(xgbmodel, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('Class')]))
xgbmodel.predict.text <- levels(mytrain$Class)[xgbmodel.predict + 1]
#predict 2
xgbmodel.predict2 <- predict(xgbmodel2, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('Class')])) 
xgbmodel.predict2.text <- levels(mytrain$Class)[xgbmodel.predict2 + 1]
#predict 3
xgbmodel.predict3 <- predict(xgbmodel3, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('Class')])) 
xgbmodel.predict3.text <- levels(mytrain$Class)[xgbmodel.predict3 + 1]

#data frame for predict 1
submit_match1 <- cbind(as.data.frame(test$ID), as.data.frame(xgbmodel.predict.text))
colnames(submit_match1) <- c('ID','Class')
submit_match1 <- data.table(submit_match1, key = 'ID')
#data frame for predict 2
submit_match2 <- cbind(as.data.frame(test$ID), as.data.frame(xgbmodel.predict2.text))
colnames(submit_match2) <- c('ID','Class')
submit_match2 <- data.table(submit_match2, key = 'ID')
#data frame for predict 3
submit_match3 <- cbind(as.data.frame(test$ID), as.data.frame(xgbmodel.predict3.text))
colnames(submit_match3) <- c('ID','Class')
submit_match3 <- data.table(submit_match3, key = 'ID')

sum(diag(table(mytest$Class, xgbmodel.predict)))/nrow(mytest) 
sum(diag(table(mytest$Class, xgbmodel.predict2)))/nrow(mytest)
sum(diag(table(mytest$Class, xgbmodel.predict3)))/nrow(mytest)

#ensembling 
submit_match3$Class2 <- submit_match2$Class 
submit_match3$Class1 <- submit_match1$Class

#function to find the maximum value row wise
Mode <- function(x) {
  u <- unique(x)
  u[which.max(tabulate(match(x, u)))]
}
x <- Mode(submit_match3[,c("Class","Class2","Class1")])
y <- apply(submit_match3,1,Mode)
final_submit <- data.frame(id= submit_match3$ID, Class = submit_match3$Class)
write.csv(final_submit, 'ensemble4.csv', row.names = FALSE)
#view submission file
data.table(final_submit)
final_submit$class1=0
final_submit$class2=0
final_submit$class3=0
final_submit$class4=0
final_submit$class5=0
final_submit$class6=0
final_submit$class7=0
final_submit$class8=0
final_submit$class9=0

for(i in 1:5668) {0
  if(final_submit$Class[i] == 1){
    final_submit$class1[i]=1
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 2){
    final_submit$class1[i]=0
    final_submit$class2[i]=1
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 3){
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=1
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 4){
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=1
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 5){
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=1
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 6){
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=1
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 8){
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=1
    final_submit$class9[i]=0
  }
  else if(final_submit$Class[i] == 7){
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=1
    final_submit$class8[i]=0
    final_submit$class9[i]=0
  }
  else{
    final_submit$class1[i]=0
    final_submit$class2[i]=0
    final_submit$class3[i]=0
    final_submit$class4[i]=0
    final_submit$class5[i]=0
    final_submit$class6[i]=0
    final_submit$class7[i]=0
    final_submit$class8[i]=0
    final_submit$class9[i]=1
  }
}
  
final_submit1=final_submit
final_submit$Class=NULL
  #final submission
write.table(final_submit, "submission4.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

