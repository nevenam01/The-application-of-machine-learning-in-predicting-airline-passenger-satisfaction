library(caret)
library(ROSE)
library(rpart)
library(rpart.plot)
source("Util.R")
data <- readRDS(file = "AnalizaISelekcija.RDS")

str(data)

#Da bi Neutral or Dissatisfied pretvorilo u Neutral.or.Dissatisfied
levels(data$Satisfaction) <- make.names(levels(data$Satisfaction))

#Trening i Test
set.seed(123)
train.indices <- createDataPartition(data$Satisfaction, p = 0.8, list = FALSE)
train.data <- data[train.indices, ]
test.data <- data[-train.indices, ]

#Stablo 1 - nebalansirano, bez podesavanja hiper parametara##########################
set.seed(123)

tree1 <- rpart(Satisfaction ~ ., data = train.data)
rpart.plot(tree1)

tree1.pred <- predict(tree1, newdata = test.data, type = "class")

cm1 <- table(true=test.data$Satisfaction,
             predicted=tree1.pred)
cm1
#                                       predicted
#true                      Neutral or Dissatisfied Satisfied
# Neutral or Dissatisfied                   12735      1955
# Satisfied                                  1451      9834

tree1.eval <- compute_eval_metrics(cm1)
tree1.eval

#accuracy precision    recall        F1 
#0.8688739 0.8341674 0.8714222 0.8523880 

#rebalansiranje podataka koriscenjem kros-validacije - down-sampling metoda

tr_ctrl <- trainControl(method = "repeatedcv", repeats = 5, 
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        sampling = "down")

cp_grid <- expand.grid(.cp = seq(0.001, 0.01, 0.0025))

set.seed(123)
down <- train(x = train.data[,-19],
              y = train.data$Satisfaction,
              method = "rpart",
              metric = "ROC",
              trControl = tr_ctrl,
              tuneGrid = cp_grid)

down$bestTune$cp # 0.001

#rebalansiranje podataka koriscenjem kros-validacije - up-sampling metoda

tr_ctrl$sampling <- "up"

set.seed(123)
up <- train(x = train.data[,-19],
            y = train.data$Satisfaction,
            method = "rpart",
            metric = "ROC",
            trControl = tr_ctrl,
            tuneGrid = cp_grid)
up 

up$bestTune$cp# 0.001

#rebalansiranje podataka koriscenjem kros-validacije - rose metoda

tr_ctrl$sampling <- "rose"

set.seed(123)
rose <- train(x = train.data[,-19],
              y = train.data$Satisfaction,
              method = "rpart",
              metric = "ROC",
              trControl = tr_ctrl,
              tuneGrid = cp_grid)

rose$bestTune$cp #0.001


#rebalansiranje podataka koriscenjem kros-validacije - default metoda

tr_ctrl$sampling <- NULL

set.seed(123)
original <- train(x = train.data[,-19],
                  y = train.data$Satisfaction,
                  method = "rpart",
                  metric = "ROC",
                  trControl = tr_ctrl,
                  tuneGrid = cp_grid)

original$bestTune$cp # 0.001

optimal_cp <- 0.001

models <- list(original = original,
               down = down,
               up = up,
               rose = rose)

resampling <- resamples(models)
summary(resampling, metric = "ROC")


#################################################################################
#Call:
#  summary.resamples(object = resampling, metric = "ROC")

#Models: original, down, up, rose 
#Number of resamples: 50 

#ROC 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#original 0.9791078 0.9807675 0.9811362 0.9812498 0.9816632 0.9838333    0
#down     0.9774626 0.9799698 0.9805358 0.9805941 0.9812135 0.9828674    0
#up       0.9774626 0.9799698 0.9805358 0.9805941 0.9812135 0.9828674    0
#rose     0.9518936 0.9567727 0.9613492 0.9605595 0.9635583 0.9687106    0  
# ################################################################################
#Uporedjivanjem vrednosti ROC AUC metrike, odlucujemo da koristimo default(original) model

#Stablo 2 - nebalansirano stablo uz koriscenje cp parametra
original$finalModel

rpart.plot(original$finalModel)
tree2.pred <- predict(object = original$finalModel, newdata = test.data, type = "class")

#Kreiranje matrice konfuzije
cm2 <- table(true=test.data$Satisfaction, predicted=tree2.pred)
cm2

                                        #predicted
#true                      Neutral.or.Dissatisfied Satisfied
#Neutral.or.Dissatisfied                   14284       406
#Satisfied                                  1035     10250

tree2.eval <- compute_eval_metrics(cm2)
tree2.eval

#accuracy precision    recall        F1 
#0.9445236 0.9618994 0.9082853 0.9343239 

#poredjenje stabala
data.frame(rbind(tree1.eval, tree2.eval),
           row.names = c(paste("Stablo_", 1:2, sep = "")))

#         accuracy precision    recall        F1    
#Stablo_1 0.8688739 0.8341674 0.8714222 0.8523880
#Stablo_2 0.9445236 0.9618994 0.9082853 0.9343239

#Vidimo da stablo 2 ima bolje evalucione metrike

# Izracunavanje znacajnosti atributa
importance <- varImp(original$finalModel, scale = TRUE)
str(importance)
# Nalazenje max vrednosti znacajnosti i prevodjenje na opseg 0-100
max_importance <- max(importance$Overall, na.rm = TRUE)
importance$Normalized <- (importance$Overall / max_importance) * 100

print(importance)

