library(caret)

#install.packages('brio')
library(devtools)
library(roxygen2)
library(ROCR)

source_url("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/DrawConfusionMatrix.R")
source ('DrawConfusionMatrix.R')

library(rpart)
library(rpart.plot)


data.df = read.csv("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/IBM-HR-Employee-Attrition.csv")
data <- subset(data.df, select = -c(EmployeeCount,StandardHours,Over18,EmployeeNumber) )

# eliminamos los valores nulos
data.df = na.omit(data.df)

# Fijamos semilla para inicializar datos aleatorios, así podremos obtener 
#repetitividad en los experimentos
set.seed(123)

# Particionamiento de los datos en conjuntos de entrenamiento y test
train_split_idx <- caret::createDataPartition(data$Attrition, p = 0.8, list = FALSE)
train <- data[train_split_idx, ]
test <- data[-train_split_idx, ]



#############################
# KNN

# Estrategia de control
knn_control <- trainControl(method = "repeatedcv", # validaciones cruzadas con repetición
                            number = 5, 
                            repeats = 10,
                            classProbs = TRUE, 
                            # muestreo adicional que se realiza después del remuestreo 
                            # (normalmente para resolver los desequilibrios de clase).
                            # sub-conjunto aleatorio de todas las clases en el conjunto de entrenamiento 
                            # para que sus frecuencias de clase coincidan con la clase menos prevalente
                            sampling = "down", 
                            summaryFunction = twoClassSummary,
                            savePredictions = TRUE)

# proceso de entrenamiento
knn_model <- caret::train(Attrition ~ ., 
                         data=train,  
                         method="knn",
                         trControl = knn_control,	
                         preProcess = c("range"),			
                         metric = "Sens",    
                         tuneGrid = expand.grid(k = 1:50))

# Mostramos la gráfica para eliegir el número de los K venidos 
plot(knn_model)

# predicción de los datos de test usando el modelo knn_model
knn_NN_preds <- predict(knn_model, newdata=test, type="raw")
knn_confussion_matrix <- caret::confusionMatrix(as.factor(knn_NN_preds), as.factor(test$Attrition),positive="Yes")

# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = knn_confussion_matrix, caption = "Matriz de confusión KNN")

# KNN
#############################



#############################
# Decision Trees

xtest <- subset(test, select = -c(Attrition))
ytest <- as.data.frame(test$Attrition)
ytest.factor <- as.factor(unlist(ytest))


#tree_control = caret::trainControl(method = "repeatedcv", number=10, repeats=3)
tree_control = caret::trainControl(method = "cv", number=10)

tree_formula <- as.formula("Attrition ~ .")
tree_model <- caret::train(tree_formula,  data = train, method = "ctree", metric = "Accuracy", trControl = tree_control)
#tree_model <- caret::train(tree_formula,  data = train, method = "rf", metric = "Accuracy", trControl = tree_control)

# Decision Trees
#############################


#############################
# Redes Neuronales

# Conversión de variables categóricas en dummies
dmy <- dummyVars(Attrition ~ ., data = data, fullRank = T) 
data.dummies <- data.frame(predict(dmy, newdata = data)) 

# Añadimos columna Attrition con los valores Yes y No
data.dummies <- cbind(data.dummies, Attrition = c(data.df$Attrition))


NN_control <- trainControl(method = "repeatedcv", # validación cruzada con repetición
                           number = 5,            # número de paquetes de muestra
                           repeats = 3,           # repeticiones
                           classProbs = TRUE,     # clasificación
                           summaryFunction = twoClassSummary) # para optimizar métricas 

NN_tuneGrid = expand.grid(size=seq(from = 1, to = 10, by = 1),
                        decay = seq(from = 0.1, to = 0.5, by = 0.1))

NN_model = caret::train(Attrition ~ ., data = train, method = "nnet",
                       trControl = NN_control,
                       preProcess = c("center","scale"),
                       tuneGrid=NN_tuneGrid)

# Matriz de confusion
NN_preds <- predict(NN_model, newdata=xtest, type="raw")
NN_confusion_matrix <- caret::confusionMatrix(as.factor(NN_preds), as.factor(test$Attrition),positive="Yes")

# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = NN_confusion_matrix, caption = "Matriz de confusión NN")

# Redes Neuronales
#############################

######################################
# curvas ROC PARA COMPARAR LOS MODELOS

library(plyr)
library(pROC)

# ROC de KNN
KNN_roc <- llply(unique(knn_model$pred$obs), function(cls) {
  roc(response = knn_model$pred$obs==cls, predictor = knn_model$pred[,as.character(cls)])
})
plot(KNN_roc[[2]],print.auc = TRUE, print.auc.y = 0.55, col = "blue" )

# ROC de árboles
tree_pred_prob <- predict (tree_model, newdata = xtest, type="prob")
tree_roc <- roc (ytest$`test$Attrition`, tree_pred_prob[,1])
plot(tree_roc, print.auc = TRUE, print.auc.y = 0.5, col = "red", add=T)

# ROC de NN
NN_pred_prob <- predict (NN_model, newdata = xtest, type="prob")
tree_roc <- roc (ytest$`test$Attrition`, NN_pred_prob[,1])
plot(tree_roc, print.auc = TRUE, print.auc.y = 0.45, col = "black", add=T)

# curvas ROC PARA COMPARAR LOS MODELOS
######################################

