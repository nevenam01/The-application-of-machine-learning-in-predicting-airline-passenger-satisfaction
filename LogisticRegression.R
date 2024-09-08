library(caret)
library(ROSE)
library(glmnet)
library(pROC)
source("Util.R")

data <- readRDS(file = "AnalizaISelekcija.RDS")
levels(data$Satisfaction) <- make.names(levels(data$Satisfaction))
levels(data$Class) <- make.names(levels(data$Class))
levels(data$Customer.Type) <- make.names(levels(data$Customer.Type))
str(data)
summary(data)

# nasumično uzorkovanje dela podataka, radi brzeg treniranja modela
set.seed(123)  
sample_indices <- sample(seq_len(nrow(data)), size = 0.2 * nrow(data))
data <- data[sample_indices, ] #sada data ima 25976 nasumicno odabranih promenljivih

#podela na trening i test
set.seed(123)
train.indices <- createDataPartition(data$Satisfaction, p = 0.8, list = FALSE)
train.data <- data[train.indices, ]
test.data <- data[-train.indices, ]

#pretvaranje varijabli u opsegu 1-5 u faktorske(takodje je opcija i njih normaizovati, ali posto je opseg samo 1-5 mislila sam da za tim nema potrebe)
categorical_vars <- c("Ease.of.Online.Booking", "Check.in.Service", "Online.Boarding",
                      "Gate.Location", "On.board.Service", "Seat.Comfort",
                      "Leg.Room.Service", "Cleanliness", "Food.and.Drink", "In.flight.Service",
                      "In.flight.Wifi.Service", "In.flight.Entertainment", "Baggage.Handling")
train.data[categorical_vars] <- lapply(train.data[categorical_vars], as.factor)
test.data[categorical_vars] <- lapply(test.data[categorical_vars], as.factor)

#standardizacija numeričkih promenljivih(Age i Flight.Distance)
numericCols <- sapply(train.data, is.numeric)
preProcValues <- preProcess(train.data[, numericCols], method = c("center", "scale"))

train.data[, numericCols] <- predict(preProcValues, train.data[, numericCols])
test.data[, numericCols] <- predict(preProcValues, test.data[, numericCols])
summary(train.data)
summary(test.data)

# PRVI MODEL: default logistic regression model (nebalansiran i bez metode regularizacije)###################################################
lr1 <- glm(Satisfaction ~ ., data = train.data, family = 'binomial')
lr1.prob <- predict(lr1, newdata = test.data, type = "response")
lr1.pred <- ifelse(lr1.prob > 0.5, "Satisfied", "Neutral.or.Dissatisfied")
cm1 <- table(actual = test.data$Satisfaction, predicted = lr1.pred)
cm1
#                                   predicted
#actual                    Neutral.or.Dissatisfied Satisfied
#Neutral.or.Dissatisfied                    2731       189
#Satisfied                                   237      2038
eval1 <- compute_eval_metrics(cm1)
eval1
#accuracy precision    recall        F1 
#0.9179981 0.9151325 0.8958242 0.9053754 

# Lasso, Ridge, Elastic Net sa razlicitim metodama uzorkovanja (down, up i ROSE)##################
methods <- list(lasso = 1, ridge = 0, elasticnet = 0.5)
sampling_methods <- c("down", "up", "rose")

# kreiramo prazne liste
models <- list()
eval_metrics <- list()

for (method_name in names(methods)) {
  alpha_value <- methods[[method_name]]
  cv_model <- cv.glmnet(x = model.matrix(Satisfaction ~ .,
                                         data = train.data)[, -1],
                        y = train.data$Satisfaction, 
                        alpha = alpha_value, 
                        family = "binomial")
  tuning_grid <- expand.grid(alpha = alpha_value, lambda = c(cv_model$lambda.min, cv_model$lambda.1se))
  
  #smanjila sam broj repeats sa 5 na 2 radi brzeg izvrsavanja
  for (sampling in sampling_methods) {
    tr_ctrl <- trainControl(method = "repeatedcv", repeats = 2, classProbs = TRUE, summaryFunction = twoClassSummary)
    tr_ctrl$sampling <- sampling
    
    set.seed(123)
    model_name <- paste0(method_name, "_", sampling)
    print(paste0("Trening modela: ", model_name))
    
    # treniranje modela koristeći ROC kao metriku za optimizaciju modela
    models[[model_name]] <- train(Satisfaction ~ ., data = train.data, method = "glmnet", family = "binomial",
                                  metric = "ROC", trControl = tr_ctrl, tuneGrid = tuning_grid)
    
    # cuvanje najbolje lambda vrednosti za svaki model
    best_lambda <- models[[model_name]]$bestTune$lambda
    final_model <- models[[model_name]]$finalModel
    
    # pravljenje predikcije na osnovu najboljeg modela
    prob_pred <- predict(final_model, newx = model.matrix(Satisfaction ~ ., data = test.data)[, -1], s = best_lambda, type = "response")
    
    pred_labels <- ifelse(prob_pred > 0.5, "Satisfied", "Neutral.or.Dissatisfied")
    pred_labels <- as.factor(pred_labels)
    cm <- table(actual = test.data$Satisfaction, predicted = pred_labels)
    eval_metrics[[model_name]] <- compute_eval_metrics(cm)
  }
}

# poredjenje evalucionih metrika medju modelima
eval_comparison <- do.call(rbind, eval_metrics)
row.names(eval_comparison) <- names(eval_metrics)
eval_comparison


#                 accuracy precision    recall        F1
#lasso_down      0.9156882 0.8991743 0.9094505 0.9042832
#lasso_up        0.9160731 0.8992618 0.9103297 0.9047619
#lasso_rose      0.9012512 0.8870826 0.8874725 0.8872775
#ridge_down      0.9095284 0.8939328 0.9002198 0.8970653
#ridge_up        0.9091434 0.8945295 0.8984615 0.8964912
#ridge_rose      0.8970164 0.8819140 0.8830769 0.8824951
#elasticnet_down 0.9151107 0.8993902 0.9076923 0.9035222
#elasticnet_up   0.9160731 0.8996089 0.9098901 0.9047203
#elasticnet_rose 0.9010587 0.8866930 0.8874725 0.8870826

#Uporedjivanjem evalucionih metrika, odlucujemo da koristimo model koji koristi lasso metodu regularizacije i up metodu balansiranja
# DRUGI MODEL: model logisticke regresije primenom lasso regularizacije i up balansiranja #################################
lr2 <- models[["lasso_up"]]
lr2
best_lambda <- lr2$bestTune$lambda
best_lambda

lr2.prob <- predict(lr2, newdata = test.data, type = "prob")
lr2.prob.satisfied <- lr2.prob[, 2]

lr2.pred <- ifelse(lr2.prob.satisfied > 0.5, "Satisfied", "Neutral.or.Dissatisfied")
lr2.pred <- as.factor(lr2.pred)

cm2 <- table(actual = test.data$Satisfaction, predicted = lr2.pred)
cm2
#predicted
#actual                    Neutral.or.Dissatisfied Satisfied
#Neutral.or.Dissatisfied                 2688       232
#Satisfied                               204       2071
eval2 <- compute_eval_metrics(cm2)
eval2
# accuracy precision    recall        F1 
#0.9160731 0.8992618 0.9103297 0.9047619 

#Napomena: evalucione metrike za ovaj model smo mogli da dobijemo i direktno iz eval_comparison
eval2 <- eval_comparison[2, ]
eval2

#TRECI MODEL: model sa promenom praga (youden metoda) ############################################################################################
lr3.roc <- roc(ifelse(test.data$Satisfaction == "Satisfied", 1, 0),predictor =  lr2.prob.satisfied)
lr3.roc$auc
# Area under the curve: 0.9667, vrlo blizu 1, odlicno

youden.coords <- coords(lr3.roc,
                        best.method = "youden",
                        ret=c("threshold","sensitivity"),
                        x='best',
                        transpose=FALSE)

youden.threshold <- youden.coords[1,1]
youden.threshold # 0.5684613


lr3.pred <- ifelse(test=lr2.prob.satisfied > youden.threshold, yes="Satisfised",no="Neutral.or.Dissatisfied")
lr3.pred <- as.factor(lr3.pred)

cm3 <- table(actual = test.data$Satisfaction, 
             predicted = lr3.pred)
cm3
#predicted
#actual                    Neutral.or.Dissatisfied   Satisfised
#Neutral.or.Dissatisfied              2744            176
#Satisfied                            240             2035

eval3 <-  compute_eval_metrics(cm3)
eval3
# accuracy  precision    recall        F1
# 0.9197305 0.9184685 0.8962637 0.9072303 

#CETVRTI MODEL: model sa promenom praga (closest.topleft metoda) #############################
ctl.coords <- coords(lr3.roc,
                     best.method = "closest.topleft",
                     ret=c("threshold","sensitivity"),
                     x='best',
                     transpose=FALSE)

ctl.threshold <- ctl.coords[1,1]
ctl.threshold #0.5140023


lr4.pred <- ifelse(test=lr2.prob.satisfied > ctl.threshold, yes="Satisfised",no="Neutral.or.Dissatisfied")
lr4.pred <- as.factor(lr4.pred)

cm4 <- table(actual = test.data$Satisfaction, 
             predicted = lr4.pred)
cm4
#predicted
#actual                    Neutral.or.Dissatisfied Satisfised
#Neutral.or.Dissatisfied                    2703        217
#Satisfied                                   211       2064

eval4 <-  compute_eval_metrics(cm4)
eval4
# accuracy precision    recall      F1 
#0.9176131 0.9048663 0.9072527 0.9060579 

#poredjenje evalucionih metrika medju modelima
data.frame(rbind(eval1,eval2,eval3,eval4),row.names = paste0("model_",1:4))
#                            accuracy  precision    recall    F1
#model_1(default model)     0.9179981 0.9151325 0.8958242 0.9053754
#model_2(lasso, up)         0.9160731 0.8992618 0.9103297 0.9047619
#model_3(youden)            0.9199230 0.9203980 0.8945055 0.9072671
#model_4(clossest.topleft)  0.9176131 0.9048663 0.9072527 0.9060579

#Na osnovu evalucionih metrika lasso up model sa primenom Youden thresholda je najbolji je najbolji
