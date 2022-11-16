# Load the data
# https://cran.r-project.org/web/packages/MASS/MASS.pdf
data("Boston", package = "MASS")

#install.packages("caret")
library(caret)

preProcValues <- preProcess(Boston, method = c("range"))
data.centered.scaled <- predict(preProcValues, Boston)

# Split the data into training and test set
set.seed(2)
training.samples <- caret::createDataPartition(data.centered.scaled$medv, p = 0.8, list = FALSE)
train.data  <- data.centered.scaled[training.samples, ]
test.data <- data.centered.scaled[-training.samples, ]

##########################
# K-NN

KNN_model <- caret::train(
  medv~., data = train.data, method = "knn",
  trControl = caret::trainControl("cv", number = 10),
  tuneLength = 10
)

# Plot KNN_model error RMSE vs different values of k
plot(KNN_model)

# Best tuning parameter k that minimize the RMSE
#KNN_model$bestTune

# Make predictions on the test data
KNN_predictions <- predict(KNN_model,test.data)
#head(KNN_predictions)

# Compute the prediction error RMSE
RMSE(KNN_predictions, test.data$medv)
# K-NN
##########################


##########################
# NN
NN_control <- trainControl(method = "repeatedcv", # validación cruzada con repetición
                           number = 10,            # número de paquetes de muestra
                           repeats = 3)           # para optimizar métricas 

NN_tuneGrid = expand.grid(size=seq(from = 1, to = 10, by = 1),
                       decay = seq(from = 0.1, to = 0.5, by = 0.1))


NN_tuneGrid = expand.grid(layer1=seq(from=1, to=10,by=1),
                          layer2=seq(from=1, to=10,by=1),
                          layer3=seq(from=1, to=10,by=1))

NN_model = caret::train(medv ~ ., data = train.data, method = "neuralnet",
                      trControl = NN_control,
                      tuneGrid=NN_tuneGrid)

# Plot model error RMSE vs different values of k
plot(NN_model)

# Best tuning parameter k that minimize the RMSE
NN_model$bestTune

# Make predictions on the test data
NN_predictions <- predict(NN_model,test.data)
head(NN_predictions)

# Compute the prediction error RMSE
RMSE(NN_predictions, test.data$medv)
# NN
##########################



# plots real vs predicted values
x = 1:dim(test.data)[1]
plot(x, test.data$medv, col = "red", type = "l", lwd=2,
     main = "Boston housing test data prediction with KNN (regression)")

lines(x, KNN_predictions, col = "blue", lwd=2)
lines(x, NN_predictions, col = "green", lwd=2)

legend("topright",  legend = c("original", "predicted-KNN", "predicted-NN"), 
       fill = c("red", "blue","green"), col = 2:3,  adj = c(0, 0.6))
grid() 


