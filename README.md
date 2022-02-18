# Recipe-Recommend
A python project built for recommending the best recipe for you!
It can also predicting someone's rating of a certain recipe.

About Data features & forms
1.1 Item/User Popularity and its forms
There are many ways to construct the indicator of popularity. For instance, someone can sort all recipes (users) by their times of occurrence and use the ranking as the indicator. People can also draw a line between popular recipes and unpopular ones, and use a Boolean value as the indicator. Directly using the value of occurrence times is another way to deal with the data.
When it comes to the competition, I find the ranking and the value-based system does not as good as the Boolean-based system. I am assured that the ranking system is penalizing entries unreasonably, so it should not be adopted. The real-value metrics, however, surely have some advantages over the true/false system. Yet its performance after optimization is still not better than the true/false system, which can hardly be optimized.
          


1.2 Similarity
The similarity is used to describe the likeness between things and users. In this scenario, it is used to estimate the likeness between recipes people have done and a target recipe. I have tried two ways to measure the similarity, which are the Jaccard similarity and the Cosine similarity. Based on what we have discussed above, here we can use the numerical representation or the Boolean representation. Both of them make some sense as the value is distributed on When tested on the validation set, Jaccard similarity does slightly better than its counterpart. Still, the model only with the similarity cannot outperform the popular model, which means I should at least consider using them with a combination or give up similarity features.
  

1.3 Date
The date is also a vital feature of describing events. They appear in every record and can be extracted easily. At the very beginning, I use a dictionary that maps (u, i) tuple to the date. But soon I find this method does not work on the test set as it does not provide the date information. The only possible way to handle this is to make a guess but the result appears in a poor shape. Therefore, I decided to take another approach of utilizing time information. I make a dictionary to describe user-dates/recipe-dates relationship. For each entry, a list of dates shows the time that a user has made something or a recipe has been cooked. By comparing the median date of the recipe to a threshold, the system can decide whether a recipe is recently cooked. This feature can help the system to decide as people tend to try recently cooked recipes.
 

1.4 Number of features
Although many features seem promising during the competition, toppling features might not be a good idea as too much of them might drag slow the evaluation process and make the model overfitting. Too few features, however, can make the model very coarse and the result to be less accurate.
For me, I inspect the value of the coefficient for each feature in the model after the fitting. Since all of them use 0,1 as their value, it would be easy to tell vital features from trivial ones. Once, the model is optimized and the coefficient is settled. I would check them and leave out the unimportant features, only keeping necessary ones. The whole process is an adding up the process: starting with only 1 feature(in_return1), I gradually add features to the system and delete the trivial ones, as the following picture shows:
 
 
The feature that indicates whether a user made a recipe recently “u_month_near(u)” only has a coefficient of 0.02547, which shows it has barely any impact on the result. The recipe version of the feature has a way larger coefficient, which indicates its vitality, so it should be kept.
Meanwhile, the Jaccard similarity appears to play a major role in determining whether a recipe would be made. As the following picture shows, its coefficient is quite large, indicating it should be kept.
