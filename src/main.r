data <- read.csv("src/data/experiments/hyperparameters.csv")
experiment <- data.frame(data)
# Gets the point with the highest accuracy
maximum_accuracy <- experiment[which.max(experiment$accuracy), ]
# Prints it
print(maximum_accuracy)