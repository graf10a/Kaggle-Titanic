# Input data files are available in the "../input/" directory.

# Reading the train and test data.

train <- read.csv('../input/train.csv', na.strings = '')
test <- read.csv('../input/test.csv', na.strings = '')

# Adding the 'Survived' column to the test data 
# to make it compatable with the training data
# for rbind-ing.

test$Survived <- NA

# There are several instance of zero fare in both trainig and test data.
# Also, one value for 'Fare' is missing in the test data.

# Train:

fare_train_ind <- (train$Fare == 0)|is.na(train$Fare)
train[fare_train_ind, ]

# Test:

fare_test_ind <- (test$Fare == 0)|is.na(test$Fare)
test[fare_test_ind, ]

# Replacing the missing and zero values for 'Fare' with median values 
# for corresponding 'Pclass' and 'Embarked'

train$Fare[train$Fare == 0] <- NA
test$Fare[test$Fare == 0] <- NA

# Computing and storing the median fare values for the training data.

library(dplyr)

train <- train %>% 
    group_by(Pclass, Embarked) %>%
    mutate(av_fare = median(Fare, na.rm=TRUE)) %>%
    ungroup() %>%
    mutate(Fare = ifelse(is.na(Fare), av_fare, Fare))

# Computing and storing the median fare values for the training data.

fare_summary <- train %>%
    group_by(Pclass, Embarked) %>%
    summarise(av_fare = median(Fare)) %>%
    na.omit()

# Imputing the fare values in the test set.

for (i in 1:nrow(test)){
    ind <- (fare_summary$Pclass == test$Pclass[i]) & 
        (fare_summary$Embarked == test$Embarked[i])
    test$Fare[i] <- ifelse(is.na(test$Fare[i]), 
                           fare_summary[ind, 'av_fare'], test$Fare[i])
    test$Fare <- unlist(test$Fare)
}

# Removing the average fare column.

train <- as.data.frame(select(train, -av_fare))

# Checking the results.

fare_summary
train[fare_train_ind, ]
test[fare_test_ind, ]

# Storing the size of the training set for future use.

n <- nrow(train)

# Shuffling the indecies of the trainig data set.

set.seed(95454312)
shuffled_ind <- sample(1:n, n)

# Combining the data (we removed the last column 'av_fare' of the 
# train set because it does not appear in the test set).

data <- rbind(train[shuffled_ind, ], test)

# Extracting the title information from the 'Name' column. 
# The title follows the first comma. We first remove all 
# symbols prceeding the title. Then we split the remaning character
# string into separate words and choose the first one.

library(stringr)

data$Prefix <- sapply(data$Name, function(x) 
                      str_split(sub('^.*, ', '' , x), ' ')[[1]][1]
                      )

# Merging rarely occuring titles with more common ones.
# The lists of titles to be replaced:

mister <- c('Capt.', 'Col.', 'Major.', 'Don.', 'Jonkheer.', 
            'Sir.', 'Rev.')
mrs <- c('Lady.', 'Mme.', 'the', 'Dona.')
miss <- c('Mlle.', 'Ms.')

# The function for replacing the titles.

replace_titles <- function(t, rep_list){
   sapply(data$Prefix, function(x) ifelse(x %in% rep_list, t, x))
    }

data$Prefix <- replace_titles('Mr.', mister)
data$Prefix <- replace_titles('Mrs.', mrs)
data$Prefix <- replace_titles('Miss.', miss)

# There is one female 'Dr.' in the data set. 

data[(data$Prefix == 'Dr.')&(data$Sex == 'female'), 'Prefix'] <- 'Mrs.'
data[(data$Prefix == 'Dr.')&(data$Sex == 'male'), 'Prefix'] <- 'Mr.'

# Checking the 'Prefix' column.

table(data$Prefix)

# Extract the first letter from the cabin.

data$CL <- sapply(data$Cabin, function(x) str_split(x, '')[[1]][1])

# Checking the letters:

unique(data$CL)

# Replacing the missing values in 'Cabin' with 'M' (missing).

data$CL[is.na(data$CL)] <- 'M'

# Convert to factors:

data$CL <- as.factor(data$CL)

# Now, let's exatract the last names of the passengers, remove
# the ending coma, and reduce all hyphenated names to the first 
# part before the hyphen.

library(dplyr)

data$Surname <- data$Name %>% 
                sapply(function(x) str_split(x, ' ')[[1]][1]) %>%
                sapply(function(x) 
                    str_split(sub(',$', '' , x), ' ')[[1]][1]) %>%
                sapply(function(x) 
                    str_split(sub('-.*$', '' , x), ' ')[[1]][1]) 


# Creating a variable for the total family size.

data <- mutate(data, FSize = SibSp + Parch + 1)

# Creating a variable for the fare per person. We will use the ticket
# numbers to determine how many people are on the same ticket

data <- data %>% 
    group_by(Ticket) %>% 
    mutate(FarePP = Fare/n()) %>% 
    ungroup()

# Creating a variable for the number of people on the same ticket.

data <- data %>% 
    group_by(Ticket) %>% 
    mutate(Same_ticket = n()) %>% 
    ungroup()

# Converting 'FSize', 'Pclass', 'SibSp', 'Parch', 
# and 'Same_ticket' to ordered factors.

to_ordered_factors <- function(x, desc = FALSE){
    lvls <- sort(pull(unique(data[x])), decreasing = desc)
    data[x] <- as.data.frame(sapply(data[x], as.character))
    lvls <- as.character(lvls)
    return(factor(pull(data[x]), ordered = TRUE, levels = lvls))
}

data['FSize'] <- to_ordered_factors('FSize')
data['Pclass'] <- to_ordered_factors('Pclass', desc = TRUE)
data['SibSp'] <- to_ordered_factors('SibSp')
data['Parch'] <- to_ordered_factors('Parch')
data['Same_ticket'] <- to_ordered_factors('Same_ticket')

# Conerting 'Survived' and 'Prefix' to factors.

data$Survived <- as.factor(data$Survived)
levels(data$Survived) <- c('No', 'Yes')

data$Prefix <- as.factor(data$Prefix)

# There are two instances of missing information in the 'Embarked'
# column of the training data set. 

embarked_missing <- is.na(data$Embarked)
as.data.frame(data[embarked_missing, ])

# The information is missing for two women who paid the same fare 
# and traveled in the same cabin. It is reasonable to assume  
# that they should be assigned the same 'Embarked'
# value. Based on their 'Pclass' and the average fare data it is 
# reasonable to asuume that their 'Embarked' value is 'S'. 

fare_summary

# Replacing the missing information.

data[is.na(data$Embarked), 'Embarked'] = 'S'
data$Embarked <- droplevels(data$Embarked)

# Checking the result.

as.data.frame(data[embarked_missing, ])

# Now we need to impute the missing values in the 'Age' column.
# We will use the KNN algorithm to predict these values. 

# The list of variables we will be using:

cols_age <- c('Age', 'FarePP', 'Parch', 'Prefix')
# cols_age <- c('Age', 'Pclass', 'Parch', 'Prefix')

# Separating the missing age entries in the training set.

age_ind <- is.na(data$Age)
sum(age_ind)
age_train <- data[1:n, cols_age]
age_train <- age_train[!age_ind[1:n], cols_age]
age_test <- data[age_ind, cols_age]
nrow(age_test)

# The KNN method.

library(caret)

cntrl_age <- trainControl(method = "repeatedcv", 
                       number = 10, 
                       repeats = 3)

set.seed(45454)
model_age <- train(Age ~ ., data = age_train,     
                   method = "knn",
                   preProcess = c("center", "scale", "pca"),
                   tuneLength = 15,
                   tuneGrid = expand.grid(
                       k = 1:20
                   ),
                   trControl = cntrl_age)

# The information about the fit (actually, replacing 'FarePP' with
# 'Pclass' might lead to a slightly better fit for k = 2 but it 
# might well be due to overfitting):

model_age
plot(model_age)

# Predicting the missing age values.

age_test$Age <- predict(model_age, age_test)

# Transfering the predicted values to the main data set.

data$Age[age_ind] <- age_test$Age

# Separating the training and test data from 'data'

train <- data[1:n, ]
test <- data[(n + 1):nrow(data), ]

# Creating a list of variables to be included in the model.

cols <- c('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 
          'Parch', 'FSize', 'FarePP', 'Embarked', 'Prefix', 
          'Same_ticket', 'CL')

# Seting up the cross-validated hyper-parameter search.

xgb_grid = expand.grid(
    nrounds = 70, # c(70, 200),
    eta = 0.1, #seq(0.1, 0.9, 0.1),
    max_depth = 3, # seq(1, 10, 1),
    gamma = 1.5, #c(1.7, 1.5, 3.0, 30, 300),
    colsample_bytree = 1,
    min_child_weight = 5, # c(1, 5, 50, 200),
    subsample = 0.5
)

# The grid variables:

xgb_grid

# Packing the training control parameters.

xgb_trcontrol = trainControl(
    method = "repeatedcv",
    repeats = 1,
    number = 5,
    preProcOptions = list(thresh = 0.95), #or list(pcaComp = 7) (0.85 and 0.99 are worse)
    verboseIter = TRUE,
    returnData = FALSE,
    returnResamp = "all",  # save losses across all models
    classProbs = TRUE,     # set to TRUE for AUC to be computed
    summaryFunction = twoClassSummary,
    savePredictions = TRUE,
    allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
# using CV to evaluate

set.seed(54355)
model_xgb = train(
    x = data.matrix(train[cols] %>%           # was as.matrix
                      select(-Survived)),
    y = train$Survived,               # as.factor(train$Survived),
    metric = 'ROC',
    preProcess = c("center", "scale", "pca"),
    trControl = xgb_trcontrol,
    tuneGrid = xgb_grid,
    method = "xgbTree"
)

# The model infromation:

model_xgb

# Checking the accuracy on the training data (just to get a rough idea).

pred_xgb <- predict(model_xgb, 
                    newdata = data.matrix(train[cols] %>%           
                                                    select(-Survived)))
table(pred_xgb, train$Survived)
mean(pred_xgb == train$Survived)

# The variable importance:

plot(varImp(model_xgb, top = 10))

# Making predictions.

test$Survived <- predict(model_xgb, 
                    newdata = data.matrix(test[cols] %>%           
                                              select(-Survived)))

levels(test$Survived) <- c(0, 1)

write.csv(test[c('PassengerId', 'Survived')], 
           'Titanic_submission.csv', 
           row.names = FALSE)