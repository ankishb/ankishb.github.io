https://github.com/JifuZhao/120-DS-Interview-Questions

## Data Analysis (27 questions)

1. (Given a Dataset) Analyze this dataset and tell me what you can learn from it.

2. What is R2? What are some other metrics that could be better than R2 and why?

3. What is the curse of dimensionality?

4. Is more data always better?

5. What are advantages of plotting your data before performing analysis?

6. How can you make sure that you don’t analyze something that ends up meaningless?

7. What is the role of trial and error in data analysis? What is the the role of making a hypothesis before diving in?

8. How can you determine which features are the most important in your model?

9. How do you deal with some of your predictors being missing?
  
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





## Statistical Inference (15 questions)

1. In an A/B test, how can you check if assignment to the various buckets was truly random?
  
2. What might be the benefits of running an A/A test, where you have two buckets who are exposed to the exact same product?

3. What would be the hazards of letting users sneak a peek at the other bucket in an A/B test?

4. What would be some issues if blogs decide to cover one of your experimental groups?

5. How would you conduct an A/B test on an opt-in feature? 

6. How would you run an A/B test for many variants, say 20 or more?
  
7. How would you run an A/B test if the observations are extremely right-skewed?
  
8. I have two different experiments that both change the sign-up button to my website. I want to test them at the same time. What kinds of things should I keep in mind?

9. What is a p-value? What is the difference between type-1 and type-2 error?

10. You are AirBnB and you want to test the hypothesis that a greater number of photographs increases the chances that a buyer selects the listing. How would you test this hypothesis?

11. How would you design an experiment to determine the impact of latency on user engagement?

12. What is maximum likelihood estimation? Could there be any case where it doesn’t exist?

13. What’s the difference between a MAP, MOM, MLE estimator? In which cases would you want to use each?

14. What is a confidence interval and how do you interpret it?

15. What is unbiasedness as a property of an estimator? Is this always a desirable property when performing inference? What about in data analysis or predictive modeling?









## Predictive Modeling (19 questions)

1. (Given a Dataset) Analyze this dataset and give me a model that can predict this response variable.

2. What could be some issues if the distribution of the test data is significantly different than the distribution of the training data?

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
