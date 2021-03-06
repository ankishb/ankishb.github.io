https://github.com/JifuZhao/120-DS-Interview-Questions

## Data Analysis (27 questions)

1. (Given a Dataset) Analyze this dataset and tell me what you can learn from it.

2. What is R2? What are some other metrics that could be better than R2 and why?

3. What is the curse of dimensionality?

4. Is more data always better?

5. What are advantages of plotting your data before performing analysis?
- it can give some insight of data that is hard to interpret
- visulaize text data as rare words and most occured words
- clustering
- nice plot of distrinution (can find some hatch some missing value region, or even abnormalities in distribution)

## 6. How can you make sure that you don’t analyze something that ends up meaningless?
- this is generally not possible that without any information or analysis about feature, we skip it by assuming it is meaningful
- with some prior knowlegde about feature, we can through away feature
- for exp: ID of table

## 7. What is the role of trial and error in data analysis? What is the the role of making a hypothesis before diving in?
    1. to evaluate the changes in business model, even dataset collection, we need some statistical test. Stat test can give us more certainty about our claims. This we need to create hypothesis
    2. Explain hypothesis testing, Null and alternative hypothesis and p-value to test it
    3. Exp: collecting dataset from city X. 
        - NUll: dataset represent whole country
        - Alt: dataset reflect only one region
        - test and reject NULL hypothesis

## 8. How can you determine which features are the most important in your model?
1. information gain(gini in tree based model)
    1. As feature are splitted at each node
    2. we calculate the sum info-gain of feature at each node, divide by number of nodes
    3. to compare features, we can use normalized version of this
2. Variance thresholding
    - check the variance in each feature
    - feature with 0 or very less varaince, doesn't really make an impact on model prediction
3. p-value
4. permutation importance (shuffling data of each feature, and check the changes in testing accuracy)
    Pros:
    1. applicable to any model
    2. reasonably efficient
    3. reliable technique
    4. no need to retrain the model at each modification of the dataset

    Cons:
    1. more computationally expensive than the default feature_importances
    2. permutation importance overestimates the importance of correlated predictors — Strobl et al (2008)

5. Partial plot(also used for interpretability)
    - apply after train the model
    - change value of one feature in a observation
    - for example: in soccre goal prediction, if we keep increasing the speed of player, what would model predict
6. Shap
7. Lime
    - add random noise in data
    - interpret the reason
    - an explanation is obtained by locally approximating the selected model with an interpretable one (our fitted model)
    - only linear models are used to approximate local behavior

## 9.How do you deal with some of your predictors being missing?
There are 


10. You have several variables that are positively correlated with your response, and you think combining all of the variables could give you a good prediction of your response. However, you see that in the multiple linear regression, one of the weights on the predictors is negative. What could be the issue?

11. Let’s say you’re given an unfeasible amount of predictors in a predictive modeling task. What are some ways to make the prediction more feasible?

12. Now you have a feasible amount of predictors, but you’re fairly sure that you don’t need all of them. How would you perform feature selection on the dataset?
  
13. Your linear regression didn’t run and communicates that there are an infinite number of best estimates for the regression coefficients. What could be wrong?

14. You run your regression on different subsets of your data, and find that in each subset, the beta value for a certain variable varies wildly. What could be the issue here?

15. What is the main idea behind ensemble learning? If I had many different models that predicted the same response variable, what might I want to do to incorporate all of the models? Would you expect this to perform better than an individual model or worse?



16. Given that you have wifi data in your office, how would you determine which rooms and areas are underutilized and over-utilized?

17. How could you use GPS data from a car to determine the quality of a driver?

18. Given accelerometer, altitude, and fuel usage data from a car, how would you determine the optimum acceleration pattern to drive over hills?

19. Given position data of NBA players in a season’s games, how would you evaluate a basketball player’s defensive ability?

20. How would you quantify the influence of a Twitter user?

21. Given location data of golf balls in games, how would construct a model that can advise golfers where to aim?

22. You have 100 mathletes and 100 math problems. Each mathlete gets to choose 10 problems to solve. Given data on who got what problem correct, how would you rank the problems in terms of difficulty?

23. You have 5000 people that rank 10 sushis in terms of saltiness. How would you aggregate this data to estimate the true saltiness rank in each sushi?

24. Given data on congressional bills and which congressional representatives co-sponsored the bills, how would you determine which other representatives are most similar to yours in voting behavior? How would you evaluate who is the most liberal? Most republican? Most bipartisan?

25. How would you come up with an algorithm to detect plagiarism in online content?

26. You have data on all purchases of customers at a grocery store. Describe to me how you would program an algorithm that would cluster the customers into groups. How would you determine the appropriate number of clusters to include?

27. Let’s say you’re building the recommended music engine at Spotify to recommend people music based on past listening history. How would you approach this problem?


## A/B testing metrics Microsoft Bing team
Queries/Month = Queries/session * sesssion/users users/month

## Click trough Rate
- CTR is the number of clicks that your ad receives divided by the number of times your ad is shown: clicks ÷ impressions = CTR. For example, if you had 5 clicks and 100 impressions, then your CTR would be 5%.
- You can use CTR to gauge which ads and keywords are successful for you and which need to be improved. The more your keywords and ads relate to each other and to your business, the more likely a user is to click on your ad after searching on your keyword phrase.
- time based feature
- type of apps
- if ad is more related to material of website, there are more likely chances of clicking on that
```python
df_click = train[train['click'] == 1]
df_hour = train[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()
df_hour = df_hour.rename(columns={'click': 'impressions'})
df_hour['clicks'] = df_click[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()['click']
df_hour['CTR'] = df_hour['clicks']/df_hour['impressions']*100
```



## Statistical Inference (15 questions)

### 1. In an A/B test, how can you check if assignment to the various buckets was truly random?
Visually compare the distribution (box and whiskers, histogram etc.) of each variable in group A and group B. The more similar they are in appearance the more likely it is that assignment was random.

Statistically compare the distributions of each variable in group A and B using a goodness of fit test. There are a bunch to choose from. Two common ones ares Kolmogorov-Smirnov Test for continuous features and Chi-Squared Test for categorical features.

Build a simple model to try and predict which group an observation will be assigned to. If assignment is truly random then you should not be able to easily predict which group the observation was assigned to.


###2. What might be the benefits of running an A/A test, where you have two buckets who are exposed to the exact same product?

3. What would be the hazards of letting users sneak a peek at the other bucket in an A/B test?

4. What would be some issues if blogs decide to cover one of your experimental groups?

5. How would you conduct an A/B test on an opt-in feature? 

6. How would you run an A/B test for many variants, say 20 or more?
  
7. How would you run an A/B test if the observations are extremely right-skewed?
  
8. I have two different experiments that both change the sign-up button to my website. I want to test them at the same time. What kinds of things should I keep in mind?

9. What is a p-value? What is the difference between type-1 and type-2 error?

10. You are AirBnB and you want to test the hypothesis that a greater number of photographs increases the chances that a buyer selects the listing. How would you test this hypothesis?

11. How would you design an experiment to determine the impact of latency on user engagement?

###12. What is maximum likelihood estimation? Could there be any case where it doesn’t exist?
- For given data sample, we want to fit a model, while using linear regression, we fit a linear curve and then measure its coefficient.(Assumption: data is normally distributed). **But if it is not**, then we have to model it with some other method
- We model with some disgtibution, which is defined by some `stat-properties`. For normal dist it is mean and var
- To maximize that likelihood(probability of success), we estimate these parameter. It is called `MLE`

- It doesn't work for mixture density model(GMM), probabilistic PCA, where we use EM algo. Here we use expected log likelihood

### 13. What’s the difference between a MAP, MOM, MLE estimator? In which cases would you want to use each?

### 14. What is a confidence interval and how do you interpret it?
Confidence interval define the 

### 15. What is unbiasedness as a property of an estimator? Is this always a desirable property when performing inference? What about in data analysis or predictive modeling?
Unbiasedness means that the expected value of your estimator should be equal to the true value of the variable estimated. Though not always necessary to qualify an estimator as good, it is a great quality to have because it says that if you do an estimate again and again on different samples from the same population, their average must equal the actual value, which is something you'd ordinarily accept.
However, unbiasedness is not the only thing that matters. As you'd see, you only have a single sample and thus the expected value doesn't make too much sense. What matters for you is how likely it is for you to get a value that is quite close to the true value and for that we consider another quality called efficiency which measures the variance of your estimator. Assuming a normal distribution or using the Chebychev's inequality you can then know how likely, it is to get close to the true value.
As you must have guessed, an unbiased estimator with a huge variance would be useless as would be an efficient estimator with significant bias. Often, we need to go somewhere in the middle and we often try to minimize a quantity known as the Mean Squared Error.
Another quality that is most often used is consistency, which is an asymptotic property. This is often considered a necessary condition for an estimator to satisfy to be of any use. In vague terms, an estimator will be consistent if its expected value tends to the true value and the variance to zero as the sample size grows without bound.


### Micro-average vs Macro-average

Original Post - http://rushdishams.blogspot.in/2011/08/micro-and-macro-average-of-precision.html

In Micro-average method, you sum up the individual true positives, false positives, and false negatives of the system for different sets and the apply them to get the statistics.

Tricky, but I found this very interesting. There are two methods by which you can get such average statistic of information retrieval and classification.
1. Micro-average Method

In Micro-average method, you sum up the individual true positives, false positives, and false negatives of the system for different sets and the apply them to get the statistics. For example, for a set of data, the system's

True positive (TP1)  = 12
False positive (FP1) = 9
False negative (FN1) = 3

Then precision (P1) and recall (R1) will be 57.14%=TP1TP1+FP1
and 80%=TP1TP1+FN1

and for a different set of data, the system's

True positive (TP2)  = 50
False positive (FP2) = 23
False negative (FN2) = 9

Then precision (P2) and recall (R2) will be 68.49 and 84.75

Now, the average precision and recall of the system using the Micro-average method is

Micro-average of precision=TP1+TP2TP1+TP2+FP1+FP2=12+5012+50+9+23=65.96

Micro-average of recall=TP1+TP2TP1+TP2+FN1+FN2=12+5012+50+3+9=83.78

The Micro-average F-Score will be simply the harmonic mean of these two figures.
2. Macro-average Method

The method is straight forward. Just take the average of the precision and recall of the system on different sets. For example, the macro-average precision and recall of the system for the given example is

Macro-average precision=P1+P22=57.14+68.492=62.82
Macro-average recall=R1+R22=80+84.752=82.25

The Macro-average F-Score will be simply the harmonic mean of these two figures.

Suitability Macro-average method can be used when you want to know how the system performs overall across the sets of data. You should not come up with any specific decision with this average.

On the other hand, micro-average can be a useful measure when your dataset varies in size.





## Predictive Modeling (19 questions)
1. (Given a Dataset) Analyze this dataset and give me a model that can predict this response variable.

2. What could be some issues if the distribution of the test data is significantly different than the distribution of the training data?
    - If there is a time component this could imply that there was a change point.  This can happen because of external events, or even yearly seasonality.  If you are training a time series model on data that has yearly seasonality but your training window doesn't have enough data to determine the yearly pattern, then a test set which happens later in the year will look like it has a distribution different fr the training.

3. What are some ways I can make my model more robust to outliers?

4. What are some differences you would expect in a model that minimizes squared error, versus a model that minimizes absolute error? In which cases would each error metric be appropriate?

5. What error metric would you use to evaluate how good a binary classifier is? What if the classes are imbalanced? What if there are more than 2 groups?

6. What are various ways to predict a binary response variable? Can you compare two of them and tell me when one would be more appropriate? What’s the difference between these? (SVM, Logistic Regression, Naive Bayes, Decision Tree, etc.)


7. What is regularization and where might it be helpful? What is an example of using regularization in a model?

8. Why might it be preferable to include fewer predictors over many?

9. Given training data on tweets and their retweets, how would you predict the number of retweets of a given tweet after 7 days after only observing 2 days worth of data?

10. How could you collect and analyze data to use social media to predict the weather?

11. How would you construct a feed to show relevant content for a site that involves user interactions with items?

12. How would you design the people you may know feature on LinkedIn or Facebook?
This should be a collection of various algorithms to come up with this list and each algorithm can be given a weight.

    - Consider graph data structure to come up with a list of "friends" and friends of friends". Give a weight to this, say W1.
    Use user data like "Interest group" and come up with list of users which can be given to this user. Give a weight to this say W2.
    Use user demo graphics like school, college, universties, Work places user went to and give a weight to this say W3.
    For #3, each of these categories can have weight of their own.
    Finally sort the list from highest weight to lowest weight and present to the user.

    - "People you may know" feature is a personalized recommendation system. I would think Collaborative filtering and Content-based filtering techniques and probably propose I hybrid system of those two. I would also consider the advantages of a graph database.


13. How would you predict who someone may want to send a Snapchat or Gmail to?

14. How would you suggest to a franchise where to open a new store?

15. In a search engine, given partial data on what the user has typed, how would you predict the user’s eventual search query?

16. Given a database of all previous alumni donations to your university, how would you predict which recent alumni are most likely to donate?

17. You’re Uber and you want to design a heatmap to recommend to drivers where to wait for a passenger. How would you approach this?

18. How would you build a model to predict a March Madness bracket?

19. You want to run a regression to predict the probability of a flight delay, but there are flights with delays of up to 12 hours that are really messing up your model. How can you address this?





## Probability (19 questions)

1. Bobo the amoeba has a 25%, 25%, and 50% chance of producing 0, 1, or 2 o spring, respectively. Each of Bobo’s descendants also have the same probabilities. What is the probability that Bobo’s lineage dies out?

2. In any 15-minute interval, there is a 20% probability that you will see at least one shooting star. What is the probability that you see at least one shooting star in the period of an hour?

3. How can you generate a random number between 1 - 7 with only a die?

4. How can you get a fair coin toss if someone hands you a coin that is weighted to come up heads more often than tails?

5. You have an 50-50 mixture of two normal distributions with the same standard deviation. How far apart do the means need to be in order for this distribution to be bimodal?

6. Given draws from a normal distribution with known parameters, how can you simulate draws from a uniform distribution?

7. A certain couple tells you that they have two children, at least one of which is a girl. What is the probability that they have two girls?

8. You have a group of couples that decide to have children until they have their first girl, after which they stop having children. What is the expected gender ratio of the children that are born? What is the expected number of children each couple will have?

9. How many ways can you split 12 people into 3 teams of 4?

10. Your hash function assigns each object to a number between 1:10, each with equal probability. With 10 objects, what is the probability of a hash collision? What is the expected number of hash collisions? What is the expected number of hashes that are unused.

11. You call 2 UberX’s and 3 Lyfts. If the time that each takes to reach you is IID, what is the probability that all the Lyfts arrive first? What is the probability that all the UberX’s arrive first?

12. I write a program should print out all the numbers from 1 to 300, but prints out Fizz instead if the number is divisible by 3, Buzz instead if the number is divisible by 5, and FizzBuzz if the number is divisible by 3 and 5. What is the total number of numbers that is either Fizzed, Buzzed, or FizzBuzzed?

13. On a dating site, users can select 5 out of 24 adjectives to describe themselves. A match is declared between two users if they match on at least 4 adjectives. If Alice and Bob randomly pick adjectives, what is the probability that they form a match?

14. A lazy high school senior types up application and envelopes to n different colleges, but puts the applications randomly into the envelopes. What is the expected number of applications that went to the right college?

15. Let’s say you have a very tall father. On average, what would you expect the height of his son to be? Taller, equal, or shorter? What if you had a very short father?

16. What’s the expected number of coin flips until you get two heads in a row? What’s the expected number of coin flips until you get two tails in a row?

17. Let’s say we play a game where I keep flipping a coin until I get heads. If the first time I get heads is on the nth coin, then I pay you 2n-1 dollars. How much would you pay me to play this game?

18. You have two coins, one of which is fair and comes up heads with a probability 1/2, and the other which is biased and comes up heads with probability 3/4. You randomly pick coin and flip it twice, and get heads both times. What is the probability that you picked the fair coin?

19. You have a 0.1% chance of picking up a coin with both heads, and a 99.9% chance that you pick up a fair coin. You flip your coin and it comes up heads 10 times. What’s the chance that you picked up the fair coin, given the information that you observed?

20. What is a P-Value ?








## Product Metrics (15 questions)

1. What would be good metrics of success for an advertising-driven consumer product? (Buzzfeed, YouTube, Google Search, etc.) A service-driven consumer product? (Uber, Flickr, Venmo, etc.)

2. What would be good metrics of success for a productivity tool? (Evernote, Asana, Google Docs, etc.) A MOOC? (edX, Coursera, Udacity, etc.)

3. What would be good metrics of success for an e-commerce product? (Etsy, Groupon, Birchbox, etc.) A subscription product? (Net ix, Birchbox, Hulu, etc.) Premium subscriptions? (OKCupid, LinkedIn, Spotify, etc.) 

4. What would be good metrics of success for a consumer product that relies heavily on engagement and interaction? (Snapchat, Pinterest, Facebook, etc.) A messaging product? (GroupMe, Hangouts, Snapchat, etc.)

5. What would be good metrics of success for a product that offered in-app purchases? (Zynga, Angry Birds, other gaming apps)

6. A certain metric is violating your expectations by going down or up more than you expect. How would you try to identify the cause of the change?

7. Growth for total number of tweets sent has been slow this month. What data would you look at to determine the cause of the problem?

8. You’re a restaurant and are approached by Groupon to run a deal. What data would you ask from them in order to determine whether or not to do the deal?

9. You are tasked with improving the efficiency of a subway system. Where would you start?

10. Say you are working on Facebook News Feed. What would be some metrics that you think are important? How would you make the news each person gets more relevant?

11. How would you measure the impact that sponsored stories on Facebook News Feed have on user engagement? How would you determine the optimum balance between sponsored stories and organic content on a user’s News Feed?

12. You are on the data science team at Uber and you are asked to start thinking about surge pricing. What would be the objectives of such a product and how would you start looking into this?

13. Say that you are Netflix. How would you determine what original series you should invest in and create?

14. What kind of services would find churn (metric that tracks how many customers leave the service) helpful? How would you calculate churn?

15. Let’s say that you’re are scheduling content for a content provider on television. How would you determine the best times to schedule content?





## Communication (5 questions)

1. Explain to me a technical concept related to the role that you’re interviewing for.

2. Introduce me to something you’re passionate about.

3. How would you explain an A/B test to an engineer with no statistics background? A linear regression?

4. How would you explain a confidence interval to an engineer with no statistics background? What does 95% confidence mean?

5. How would you explain to a group of senior executives why data is important?





## Does kmean converge to global; solution?
No, it converge to local solution. Kmean solves a NP hard problem. Even for 30 clusters, it can have billions of possibiities to explore.
- Second, it keep one parameter space fixed, while optimizing another parameteric space
- closed form solution can be found, it it can be exponential nature.




## Interview Questions on Machine Learning (Analytics Vidhya)



Q1. You are given a train data set having 1000 columns and 1 million rows. The data set is based on a classification problem. Your manager has asked you to reduce the dimension of this data so that model computation time can be reduced. Your machine has memory constraints. What would you do? (You are free to make practical assumptions.)

Q2. Is rotation necessary in PCA? If yes, Why? What will happen if you don’t rotate the components?

Q3. You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?

Q4. You are given a data set on cancer detection. You’ve build a classification model and achieved an accuracy of 96%. Why shouldn’t you be happy with your model performance? What can you do about it?

Q5. Why is naive Bayes so ‘naive’ ?

Q6. Explain prior probability, likelihood and marginal likelihood in context of naiveBayes algorithm?

Q7. You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. Can this happen? Why?

Q8. You are assigned a new project which involves helping a food delivery company save more money. The problem is, company’s delivery team aren’t able to deliver food on time. As a result, their customers get unhappy. And, to keep them happy, they end up delivering food for free. Which machine learning algorithm can save them?

Q9. You came to know that your model is suffering from low bias and high variance. Which algorithm should you use to tackle it? Why?

Q10. You are given a data set. The data set contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run PCA. Would you remove correlated variables first? Why?

Q11. After spending several hours, you are now anxious to build a high accuracy model. As a result, you build 5 GBM models, thinking a boosting algorithm would do the magic. Unfortunately, neither of models could perform better than benchmark score. Finally, you decided to combine those models. Though, ensembled models are known to return high accuracy, but you are unfortunate. Where did you miss?

Q12. How is kNN different from kmeans clustering?


Q13. How is True Positive Rate and Recall related? Write the equation.

Q14. You have built a multiple regression model. Your model R² isn’t as good as you wanted. For improvement, your remove the intercept term, your model R² becomes 0.8 from 0.3. Is it possible? How?

Q15. After analyzing the model, your manager has informed that your regression model is suffering from multicollinearity. How would you check if he’s true? Without losing any information, can you still build a better model?

Q16. When is Ridge regression favorable over Lasso regression?

Q17. Rise in global average temperature led to decrease in number of pirates around the world. Does that mean that decrease in number of pirates caused the climate change?

Q18. While working on a data set, how do you select important variables? Explain your methods.

Q19. What is the difference between covariance and correlation?

Q20. Is it possible capture the correlation between continuous and categorical variable? If yes, how?

Q21. Both being tree based algorithm, how is random forest different from Gradient boosting algorithm (GBM)?

Q22. Running a binary classification tree algorithm is the easy part. Do you know how does a tree splitting takes place i.e. how does the tree decide which variable to split at the root node and succeeding nodes?

Q23. You’ve built a random forest model with 10000 trees. You got delighted after getting training error as 0.00. But, the validation error is 34.23. What is going on? Haven’t you trained your model perfectly?

Q24. You’ve got a data set to work having p (no. of variable) > n (no. of observation). Why is OLS as bad option to work with? Which techniques would be best to use? Why?

Q25. What is convex hull ? (Hint: Think SVM)

Q26. We know that one hot encoding increasing the dimensionality of a data set. But, label encoding doesn’t. How ?

Q27. What cross validation technique would you use on time series data set? Is it k-fold or LOOCV?

Q28. You are given a data set consisting of variables having more than 30% missing values? Let’s say, out of 50 variables, 8 variables have missing values higher than 30%. How will you deal with them?

29. ‘People who bought this, also bought…’ recommendations seen on amazon is a result of which algorithm?

Q30. What do you understand by Type I vs Type II error ?

Q31. You are working on a classification problem. For validation purposes, you’ve randomly sampled the training data set into train and validation. You are confident that your model will work incredibly well on unseen data since your validation accuracy is high. However, you get shocked after getting poor test accuracy. What went wrong?

Q32. You have been asked to evaluate a regression model based on R², adjusted R² and tolerance. What will be your criteria?

Q33. In k-means or kNN, we use euclidean distance to calculate the distance between nearest neighbors. Why not manhattan distance ?



Q34. Explain machine learning to me like a 5 year old.

Q35. I know that a linear regression model is generally evaluated using Adjusted R² or F value. How would you evaluate a logistic regression model?

Q36. Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?

Q37. Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?

Q38. When does regularization becomes necessary in Machine Learning?

Q39. What do you understand by Bias Variance trade off?

Q40. OLS is to linear regression. Maximum likelihood is to logistic regression. Explain the statement.
