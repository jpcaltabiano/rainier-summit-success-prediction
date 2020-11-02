library(shiny)
library(mltools)
library(data.table)
library(tidyverse)
library(caret)
library(MASS)
library(boot)
library(leaps)
library(glmnet)
library(dplyr)
library(pls)
library(shiny)
library(readr)
library(e1071)
library(ROSE)
library(MLmetrics)
library(pROC)

intro = readChar("intro.txt", file.info("intro.txt")$size)

ui <- fluidPage(
  
  # App title ----
  titlePanel(h1("Can Mt. Rainier's weather predict a climber's success?")),
  
  # Sidebar layout with input and output definitions ----
  
  fluidRow(
    column(12,
      p("Joseph Caltabiano"),
      p("1 Nov 2020"),
      br(),
      p("Mt. Rainier in Washington is one of the most popular mountaineering objectives in the US. It is the tallest mountain in the contiguous US, standing at 14,411ft. Every year around 100 climbers attempt the summit. Typically only 50 of those climbers will be successful, and on average 2 people die every year attempting to climb the mountain."),
      br(),
      p("Mountaineers have robust and generally accurate techniques for predicting mountain weather. They are able to use large amounts of data from mountain weather stations to cross-reference patterns noticied by previous climbers and locals to determine the safest weather windows to climb during. With high confidence in their weather predictions, mountaineers will only attempt a summit bid if they can summit and return to base well witihn the safe weather window."),
      br(),
      p("I am a rock climber and have noticed that while there is a wealth of weather prediction, there is a lack of computational methods for predicting if your expidition will be successful depending on the weather prediction. Climbers generally avoid percipitation and favor colder weather for better ice conditions and reduced hypothermia risk (counterintuitive, but while climbing, the amount you sweat is often more predictive than temperature). But can we figure out if an expidition will be successful with a high degree of confidence based on numbers, instead of instinct and experience?"),
      br(),
      p("I chose to examine data from Mt. Rainier expiditions to explore predicting whether or not an expidtion will be successful based on weather conditions. I found my initial data from Kaggle here: https://www.kaggle.com/codersree/mount-rainier-weather-and-climbing-data. This contains two files, one with data from climbing expiditions, and one with weather data. The data spans from September 2014 through December 2015. The columns names below are adjusted so they are human-readable but the actual data contains different column names."),
      br(),
      tableOutput("climbs.head"),
      br(),
      tableOutput("weather.head"),
      br(),
      h2("Data exploration and preparation"),
      p("The first things apparent about this data was that not all of it was necessary, and some could be condensed. Additionally, there was some data I felt could be useful for this classification task that was not included. To address this, I went to the original source for the weather data and downloaded percipitation and snow depth measurements for the time frame. This data included a date, and 5 readings from different sites on the mountain."),
      tableOutput("snow.head"),
      br(),
      tableOutput("percip.head"),
      br(),
      p("For both snow and percipition data, only data from two sites (Paradise Base and Sunrise Base) were available, the rest were NA. For percipitation, only one site's data was good. The other site that was not NA was all 0 values except for one day, so would not have been a good predictor. Both data sets also had many readings per day at 3 hour intervals, so I took the average for each date."),
      br(),
      p("Looking at the climbing data, I chose to ignore which route was attempted, and also dropped the columns for how many expiditons were attempted that day and what percentage were successful. I condensed the data down to one date per row, and gave it either 0 for \"summit bid unsuccessful\" and 1 for \"summit bid successful\"."),
      p("Looking at the weather data, I found the voltage did not relate to the weather but was an artefact from the weather station's collection tools so I dropped this column. I joined all four datasets on the date to create my final data."),
      br(),
      tableOutput("data.head"),
      br(),
      p("One major issue with this data was that there were only 179 total observations. Prediction confidence and accuracy would be very low only using this amount of data. I chose to utilize a technique called Random Over-Sampling Examples (ROSE) that synthesizes new data simulated according to a smoothed-bootstrapped approach. Given a class, ROSE will generate a new set of predictors based off neighbors in the multi-dimensional feature space. I used ROSE to generate a synthetic data set of 3000 observations which I use as training data, and saved my original 179 observations to use as a validation set to determine how successful my approach would be on real-world data. The training data was evenly balanced with 52% of the observations belonging to the positive class."),
      h2("Modelling"),
      h3("Logistic Regression"),
      p("I chose to use a Support-Vector Machine model for this classificaiton task. I started by creating a baseline to which I could compare performance. To create this baseline, I used logistic regression, one of the simplest methods for classification. Regressing all the predictors on the class, we can see which predictors were statistically significant in this model. I chose to use F1 and AUC as my main measures of error, as well as a confusion matrix to display the different types of error. All error statistics were calculated using the test set of the original data. I plot the ROC curve on sensitivty vs specificity, and get the coordinates for the best threshold. The F1 and confusion matrix were calculated using this threshold."),
    ),
    column(6, 
      verbatimTextOutput("log.fit"),
    ),
    column(6, 
      verbatimTextOutput("log.coords"),
      plotOutput("log.roc"),
      verbatimTextOutput("log.conf"),
      verbatimTextOutput("log.f1"),
    ),
    column(12,
      h3("SVM"),
      p("To create my SVM model, I first had to find the best parameters. SVMs use 'cost' to determine the amount of misclassifications to allow, i.e. how many points can be on the wrong side of the hyperplane, or how 'hard' or 'soft' the margin is. The gamma value is a parameter of the (Guassian) kernel that affects how non-linear relationships are handled. I used the function tune() to test values for cost from 10^-2 to 10^3, and values for gamma in the set { .0001, .001, .01, .1, .5, 1, 2 } using 10-fold cross-validation. I have included this function in the source code as a comment, and I use constant values calculated earlier for this application. The function takes between 5-8 minutes to run, so I will not attempt to run it here. The best cost and gamma values calculated with this function were 10 and 0.1 respectively. I also experimented with the four possible kernels: {radial, linear, polynomial, sigmoid}. I found radial to be the best kernel. Below, the model is configured with its optimal parameters. You can change the values to see how they affect the output. You can also change the threshold used to classify the probability outputs of the SVM. Notice how the best calculated threshold (0.95) produces a slightly lower F1 score than with the default cutoff."),
      p("One way to visualize the output of an SVM in high-dimensional space is to plot two predictors against eachother to see where the decision boundary is located. Try selecting different predictors below. To note - some combinations of predictors will not produce a visible boundary on the plot, so try many different combinations!"),
      br()
    ),
    
    sidebarLayout(
      sidebarPanel(
        selectInput("svm.kernel",
                    label="Select a kernel type",
                    choices=c("radial",
                              "linear",
                              "polynomial",
                              "sigmoid"),
                    selected="radial"
                    ),
        sliderInput("svm.cost",
                    label="Select a cost (10^x)",
                    min=-2,
                    max=3,
                    step=1,
                    value=1
                    ),
        selectInput("svm.gamma",
                    label="Select a gamma value",
                    choices=c(.0001,
                              .001,
                              .01,
                              .1,
                              .5,
                              1,
                              2),
                    selected=.1),
        sliderInput("svm.thresh",
                    label="Select a classification threshold",
                    min=0.00,
                    max=1.00,
                    step=0.01,
                    value=0.95),

        selectInput("svm.bound.x",
                   label="Predictor 1",
                   choices=c("temp",
                             "humidity",
                             "wspeed",
                             "wdir",
                             "solar",
                             "paradise.snow",
                             "sunrise.snow",
                             "paradise.percip"),
                   selected="temp"
        ),
        selectInput("svm.bound.y",
                   label="Predictor 2",
                   choices=c("temp",
                             "humidity",
                             "wspeed",
                             "wdir",
                             "solar",
                             "paradise.snow",
                             "sunrise.snow",
                             "paradise.percip"),
                   selected="humidity"
        ),

      ),
      mainPanel(
        column(6,
               plotOutput("svm.roc"),
               ),
        column(6,
               tableOutput("svm.coords"),
               verbatimTextOutput("svm.conf"),
               verbatimTextOutput("svm.f1"),
               ),
        column(12,
               plotOutput("svm.bound"),
               )
        
      ),
    ),
  ),
  fluidRow(
    column(12,
      h2("Final thoughts and potential issues"),
      br(),
      p("Overall, it seems possible to predict whether or not a summit bid on Mt. Rainier will be successful based on the current weather readings. SVM has proven to be a successful model for this task and produces results with very good error metrics. One thing that I was not expecting during this project was for solar radiation to be a very good predictor for a successful summit bid (note the very small p-value in the logisitc regression model). Initially, I was unsure how this feature would relate to the target, but it predicts better than wind speed, and just as well as temperature. After investigating the data further, I found solar radiation was the reading of the intensity of radiation reaching the Earth's surface from the Sun. Higher levels of radiation are likely the main cause of melting the surface of snow and ice, causing worse climbing conditions and a higher hypothermia risk from higher levels of liquid water on the route."),
      br(),
      p("There are some issues for this classification task that should not be ignored. Primarily, the data does not include a reason for a failed expdition. Most often the reason in the real world is weather. However, Mt. Rainier sees many guided groups of inexperienced climbers, so it is likely some expeditions fail due to a client's lack of fitness or preparation, or some unforseen medical issue. Additionally, the data is already biased towards better weather. Days with weather bad enough that groups choose not to climb would not factor in to our prediction; however, I do not believe this takes away from the success of the prediction. One additional issue I was unsure about was the quality of the training data produced by the ROSE method. Some of the data would not make sense in the real world. For example, wind direction (measured in degrees) ranges from -172 to 464, and some values of percipitation are negative. Data like this would be impossible in reality, but this is most likely not a severe issue, as all the metrics reported were calculated using a validation set of real-world data. If the methods were able to do well on the real data, the results are acceptable."),
      br(),
      h3("Sources:"),
      p("https://nwac.us/data-portal/location/mt-rainier/"),
      p("https://www.kaggle.com/codersree/mount-rainier-weather-and-climbing-data"),
      p("https://www.thenewstribune.com/news/local/article243890242.html#:~:text=In%20the%20area%20where%20Bunker,according%20to%20the%20Park%20Service.")
    ),
  )
)
server <- function(input, output) {
  
  climbs.raw = read.csv("climbing_statistics.csv")
  weather.raw = read.csv("Rainier_Weather.csv")
  snow14 = read.csv("snow_depth_2014.csv")
  snow15 = read.csv("snow_depth_2015.csv")
  percip14 = read.csv("percip_2014.csv")
  percip15 = read.csv("percip_2015.csv")
  
  snow.raw = bind_rows(snow14, snow15)
  percip.raw = bind_rows(percip14, percip15)
  
  colnames(climbs.raw) <- c("date", "route", "attempted", "succeeded", "precentage")
  colnames(weather.raw) <- c("date", "voltage", "temp", "humidity", "wspeed", "wdir", "solar")
  
  colnames(snow.raw) <- c("date", "pwind", "paradise.snow", "sunrise.snow", "supper", "muir")
  colnames(percip.raw) <- c("date", "pwind", "paradise.percip", "sunrise.percip", "supper", "muir")
  
  
  output$climbs.head <- renderTable({head(climbs.raw, n=5)})
  output$weather.head <- renderTable({head(weather.raw, n=5)})
  output$snow.head <- renderTable({head(snow.raw, n=5)})
  output$percip.head <- renderTable({head(percip.raw, n=5)})
  
  set.seed(1)
  
  climbs = climbs.raw[,c("date","succeeded")]
  climbs = aggregate(succeeded~date, data=climbs, sum)
  
  weather = weather.raw[,c("date", "temp", "humidity", "wspeed", "wdir", "solar")]
  
  snow = snow.raw[,c("date", "paradise.snow", "sunrise.snow")]
  snow = aggregate(.~date, data=snow, mean)
  
  percip = percip.raw[,c("date", "paradise.percip")]
  percip = aggregate(.~date, data=percip, mean)
  
  test = merge(merge(merge(climbs, weather, by="date"), snow, by="date"), percip, by="date")
  # data = merge(merge(climbs, weather, by="date"), snow, by="date")
  test$date <- NULL
  test$succeeded[test$succeeded > 0] <- 1
  
  output$data.head <- renderTable({head(test, n=5)})
  
  rose = ROSE(succeeded~., data=test, N=3000, p=0.5)
  data = rose$data
  
  train = data
  nrow(train[which(train$succeeded > 0),])/nrow(train)
  
  log.fit = glm(succeeded ~ ., data=train, family=binomial)
  log.preds = predict(log.fit, test, type="response")
  predictions = NULL
  predictions[log.preds<1-0.602] <- 0
  predictions[log.preds>=1-0.602] <- 1
  log.roc <- roc(test$succeeded ~ as.numeric(log.preds))
  
  output$log.fit <- renderPrint({
    summary(log.fit)
  })
  
  output$log.conf <- renderPrint({
    table(test$succeeded, predictions)
  })
  
  output$log.f1 <- renderPrint({
    logf1 = F1_Score(test$succeeded, predictions, positive="1")
    paste("F1 Score: ", logf1)
  })
  
  output$log.roc <- renderPlot({
    plot(log.roc, print.auc=T)
  })
  
  output$log.coords <- renderPrint({
    coords(log.roc, x="best", best.method="closest.topleft")
  })
  
  svm.fit = reactive({
    svm(succeeded~., data=train, type="C-classification", kernel=input$svm.kernel,
                gamma=input$svm.gamma,
                cost=10^(input$svm.cost),
                probability=TRUE)
  })
  
  # svm.tune = tune(svm, train.x=train, train.y=train$succeeded, kernel="radial", ranges=list(cost=10^(-2:3), gamma=c(.0001,.001,.01,.1,.5,1,2)))
  
  svm.preds = reactive({
    attr(predict(svm.fit(), test, probability=TRUE), "probabilities")
  })
  
  svm.roc.obj = reactive({
    roc(test$succeeded ~ as.numeric(svm.preds()[,1]))
  })
  
  best.thresh = reactive({
    coords(svm.roc.obj, x="best", best.method="closest.topleft")
  })
  
  output$svm.coords <- renderTable({
    table(format(best.thresh(), digits=4))
  })
  
  output$svm.roc <- renderPlot({
    plot(svm.roc.obj(), print.auc=T)
  })
  
  output$svm.conf <- renderPrint({
    svm.preds.temp <- svm.preds()
    predictions <- NULL
    predictions[svm.preds.temp[,1]>input$svm.thresh] <- 1
    predictions[svm.preds.temp[,1]<=input$svm.thresh] <- 0
    table(test$succeeded, predictions)
  })
  
  output$svm.f1 <- renderPrint({
    svm.preds.temp <- svm.preds()
    preds <- NULL
    preds[svm.preds.temp[,1]>input$svm.thresh] <- 1
    preds[svm.preds.temp[,1]<=input$svm.thresh] <- 0
    svmf1 = F1_Score(test$succeeded, preds, positive="1")
    paste("F1 Score: ", svmf1)
  })
  
  output$svm.bound <- renderPlot({
    fmla <- paste(input$svm.bound.x, input$svm.bound.y, sep="~")
    plot(svm.fit(), test, eval(parse(text=fmla)), col = c("#64dfdf", "#5e60ce"))
  })
}

shinyApp(ui = ui, server = server)