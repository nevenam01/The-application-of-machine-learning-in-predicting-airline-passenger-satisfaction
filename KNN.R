library(e1071)
library(caret)
library(ROSE)
library(pROC)
library(dplyr)
source("Util.R")

data <- readRDS(file = "AnalizaISelekcija.RDS")
levels(data$Satisfaction) <- make.names(levels(data$Satisfaction))
levels(data$Class) <- make.names(levels(data$Class))
levels(data$Customer.Type) <- make.names(levels(data$Customer.Type))
str(data)
summary(data)
# pretvaranje ordinalnih faktorskih promenljivih u numeričke
data <- data %>%
  mutate(Class = as.numeric(factor(Class, levels = c("Economy", "Economy.Plus", "Business"))),
    Customer.Type = as.numeric(factor(Customer.Type, levels = c("First.time", "Returning"))),
    Type.of.Travel = as.numeric(factor(Type.of.Travel, levels = c("Business", "Personal"))))

# provera tipova podataka
sapply(train.data, class)
# nasumično uzorkovanje dela podataka, radi brzeg treniranja modela
set.seed(123)  
sample_indices <- sample(seq_len(nrow(data)), size = 0.2 * nrow(data))
data <- data[sample_indices, ]

#trening i test
train.indices <- createDataPartition(data$Satisfaction, p = 0.8, list = FALSE)
train.data <- data[train.indices, ]
test.data <- data[-train.indices, ]

# standardizacija numeričkih podataka
preProcess_range <- preProcess(train.data[, -19], method = c("center", "scale"))
train.data[, -19] <- predict(preProcess_range, train.data[, -19])
test.data[, -19] <- predict(preProcess_range, test.data[, -19])
summary(train.data)
summary(test.data)

tr_ctrl <- trainControl(method = "cv", classProbs = TRUE,
                        number=10)

# definisanje raspona za broj suseda
tuning_grid_knn <- expand.grid(.k = seq(from=3,to=15,by=2))

# treniranje KNN modela sa različitim tehnikama balansiranja
#original##########################################################################################
tr_ctrl$sampling <- NULL
set.seed(123)
knn.original <- train(Satisfaction ~ ., 
                      data = train.data, 
                      method = "knn", 
                      trControl = tr_ctrl, 
                      tuneGrid = tuning_grid_knn)
knn.original
##down############################################################################################
tr_ctrl$sampling <- "down"
set.seed(123)
knn.down <- train(Satisfaction ~ ., 
                  data = train.data, 
                  method = "knn", 
                  trControl = tr_ctrl, 
                  tuneGrid = tuning_grid_knn)
knn.down
##up##################################################################################
tr_ctrl$sampling <- "up"
set.seed(123)
knn.up <- train(Satisfaction ~ ., 
                data = train.data, 
                method = "knn", 
                trControl = tr_ctrl, 
                tuneGrid = tuning_grid_knn)
knn.up
##ROSE#########################################################################################
tr_ctrl$sampling <- "rose"
set.seed(123)
knn.rose <- train(Satisfaction ~ ., 
                  data = train.data, 
                  method = "knn", 
                  trControl = tr_ctrl, 
                  tuneGrid = tuning_grid_knn)
knn.rose
#kreiranje i evaluacija liste modela--------------------------------------------------------------------------------------------
models_knn <- list(original = knn.original,
                   down = knn.down,
                   up = knn.up,
                   rose = knn.rose)
resampling_knn <- resamples(models_knn)
summary(resampling_knn)

#Call:
#  summary.resamples(object = resampling_knn)

#Models: original, down, up, rose 
#Number of resamples: 10 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#original 0.9133782 0.9178296 0.9205967 0.9202635 0.9238450 0.9254090    0
#down     0.9100096 0.9174687 0.9239654 0.9215147 0.9253156 0.9273340    0
#up       0.9104909 0.9166266 0.9232435 0.9209854 0.9238542 0.9302214    0
#rose     0.8946102 0.8997834 0.9020929 0.9030846 0.9060395 0.9157844    0

#Kappa 
#              Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#original 0.8218385 0.8313086 0.8370855 0.8363390 0.8437450 0.8469030    0
#down     0.8158830 0.8314217 0.8444919 0.8395572 0.8475657 0.8514369    0
#up       0.8165761 0.8293813 0.8430838 0.8384784 0.8443909 0.8575141    0
#rose     0.7860502 0.7962501 0.8015301 0.8030102 0.8087722 0.8284111    0

#Model sa down sampling metodom ima najbolje performanse
#PRVI MODEL: model sa down balansiranjem##########################################################
best_knn_model <- models_knn$down

# predikcija na test skupu
knn.prob <- predict(best_knn_model, newdata = test.data, type = "prob")[,2]
knn.pred <- ifelse(knn.prob > 0.5, "Satisfied", "Neutral.or.Dissatisfied")


cm.knn <- table(actual = test.data$Satisfaction, predicted = knn.pred)
cm.knn
#                                  predicted
#actual                    Neutral.or.Dissatisfied Satisfied
#Neutral.or.Dissatisfied                    2772       148
#Satisfied                                   264      2011
eval.knn <- compute_eval_metrics(cm.knn)
eval.knn
#accuracy precision    recall        F1 
#0.9206930 0.9314497 0.8839560 0.9070816

#DRUGI MODEL: model sa promenom praga (closest topleft metoda) ############################################################################
knn.roc <- roc(response = as.integer(test.data$Satisfaction), predictor = knn.prob)
knn.roc$auc  #0.9682
plot.roc(knn.roc, print.thres = 'best', print.thres.best.method = "closest.topleft")

closesttopleft.coords <- coords(knn.roc, x = 'best', best.method = "closest.topleft", ret = c("threshold", "sensitivity"))
closesttopleft.threshold <- closesttopleft.coords[1,1]
closesttopleft.threshold #0.3888889

knn.pred.closesttopleft <- ifelse(knn.prob > closesttopleft.threshold, "Satisfied", "Neutral.or.Dissatisfied")
knn.pred.closesttopleft<-as.factor(knn.pred.closesttopleft)
cm.knn.closesttopleft<- table(actual = test.data$Satisfaction, predicted = knn.pred.closesttopleft)
cm.knn.closesttopleft
#predicted
#actual                    Neutral.or.Dissatisfied Satisfied
#Neutral.or.Dissatisfied                    2681       239
#Satisfied                                   209      2066

eval.knn.closesttopleft <- compute_eval_metrics(cm.knn.closesttopleft)
eval.knn.closesttopleft
#accuracy precision    recall        F1 
#0.9137632 0.8963124 0.9081319 0.9021834 

#Za Youden se dobije 0.5 kao optimalni prag, sto je isto kao i kod podrazumevanog modela

data.frame(rbind(eval.knn,eval.knn.closesttopleft),row.names = paste0("KNN model_",1:2))
#             accuracy   precision  recall    F1
#KNN model_1 0.9206930 0.9314497 0.8839560 0.9070816
#KNN model_2 0.9137632 0.8963124 0.9081319 0.9021834