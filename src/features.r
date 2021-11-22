data <- read.csv("src/data/features/features.csv")
features <- data.frame(data)

maximum_accuracy <- features[which.max(features$accuracy), ]
print("MAXIMUM ACCURACY")
print(maximum_accuracy)

jpeg("src/data/features/features.png")
top_10 <- sorted_features[1:5, ]
barplot(accuracy ~ Features.in.Use, top_10, main = "Accuracy of Used Features",
             xlab = "Features in Use")
dev.off()
jpeg("src/data/features/features1.png")
top_10 <- sorted_features[6:10, ]
barplot(accuracy ~ Features.in.Use, top_10,
    main = "Accuracy of Used Features",
    xlab = "Features in Use"
)
dev.off()
jpeg("src/data/features/features2.png")
top_10 <- sorted_features[11:15, ]
barplot(accuracy ~ Features.in.Use, top_10,
    main = "Accuracy of Used Features",
    xlab = "Features in Use"
)
dev.off()

jpeg("src/data/features/features_single.png")
singles <- features[nchar(features$Features.in.Use) == 3, ]
barplot(accuracy ~ Features.in.Use, singles,
    main = "Accuracy of Used Features",
    xlab = "Features in Use"
)
dev.off()