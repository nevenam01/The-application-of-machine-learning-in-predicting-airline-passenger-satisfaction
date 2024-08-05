compute_eval_metrics <- function(cm){
  
  TP <- cm[2,2]
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]
  
  acc <- (TP + TN)/sum(cm)
  prec <- TP/(TP + FP)
  rec <- TP/(TP + FN)
  F1 <- 2*prec*rec/(rec+prec)
  
  c(accuracy=acc, precision=prec, recall=rec, F1=F1) 
}