## 1 Introduction
The task involved classifying the binned rating
of movies on IMDB (between 1 and 5) the key
difficulty in this task stems from the fact that
the data is of very small proportion of the
greater dataset, and as a result there is a
disparity in feature content between train and
test. This research aims to provide general
analysis for future use, rather than provide a
model with practical utility in classifying
IMDB ratings.
- Feature engineering - as discussed in
feature engineering section.
- Outliers - removed with z score = 3
- Scaling - using maximum absolute
value of a feature to scale.
- Measures of accuracy were done with
5-fold cross validation
## 2. Preprocessing (methodology)
### 2.1 Feature Engineering
Feature engineering proved extremely difficult
and often reduced the accuracy of data.
Binning was attempted for many features due
to the broad range of data scales in the dataset,
however, it universally reduced performance
of models future studies should analyse
binning techniques, as in theory this dataset is
a good candidate for binning.
</br></br>
**Figure 1- histogram of the duration column,
showing clear distributions Duration had a MI
score of 0.07, whilst the binned duration had a MI
score of 0.04.**
</br></br>
The "gross" column underwent the most
significant transformation in the process of
feature engineering. It was noticed that the
train data spanned many years, going into the
1930s, and as such, the gross revenue of a
movie would be hardly representative of its
significance in predicting the binned IMDB
score. Therefore, using the American CPI
(consumer price index) library, which has data
for calculating inflation going back to 1913,
was used to adjust the revenue for inflation.
The adjusted gross has a MI score of 0.06, a
marked improvement over the 0.04 of gross,
Every genre in the genres feature were split
into individual features. This was done after
analysis concluded that a movies presence in
genre had significant influence on its
classification. For example, the "drama" genre
had a mutual information score of 0.03, which
was significant.
### 2.2 Outlier removal
Removing outliers was done using z-score
removal. Changing the Z score value from 3 -
8 made noticeable difference in final output of
the data. This could indicate that the presence
of outliers in the training set were relatively
important in the test set, or rather, that the
outliers were extremely difficult to generalise
to with the data provided. It was particularly
noticed that beyond z = 3, for example
decision tree had a mean squared error (MSE)
of 0.33 at z = 3, but at z = 8, this climbed to
0.37
</br></br>
Figure 2- histogram of the num_voted_users, the
most significant column in relation to IMDB score
according to MI, showing the presence of outliers
particularly with x = 4.
</br></br>
### 2.3 Imputation
Imputation was initially done with KNN with
K = 5. The KNN performance with 5-fold
cross validation was low on the engineered
training data, at 0.63 accuracy, and suggests
other methods might be better for model
performance. Instead, average was used. This
resulted in slightly better performance,
indicating the unreliability of KNN, however,
average is not adequate either, as the data
distribution was impacted by scale and
outliers.
</br></br>
**Figure 3- scatterplot of the two most significant
features. Clearly showing a relationship that isn’t
very linear, and therefore making imputation by
KNN a poor choice.**
</br></br>
### 2.4 feature selection
When using feature selection with mutual
information, the average performance of all
models decreased by a small but noticeable
amount. This has a key implication for the
dataset. the dataset has little noise, and all
features have some relevance to each other,
even though the mutual information scores to
the label are quite low.
### 2.5 Scaling
MaxAbsScaler and StandardScaler were both
tested. MaxAbsScaler was chosen to be used
in the end because it has less sensitivity to
outliers since everything is scaled by its
maximum value. Relying on the mean in
StandardScaler, which can be heavily altered
by outliers, was not appropriate in this dataset,
which has many outliers evident in the z-score
removal of outliers.
</br></br>
**Figure 4- same distribution as figure 3, but scaled
and outliers removed to improve the ability to
discern a relationship.**
</br></br>
### 2.6 Weighting
Weighting was attempted using L1
regularization penalty and logistic regression.
L1 regularization was chosen to reduce the
overfitting that was expected to be
problematic. However, the use of this method
likely penalized too heavily, as the accuracy
was reduced, indicating information loss even
though logistic regression would have likely
provided good coefficients due to its
reasonable performance as a classifier on this
dataset. Weighting was thus removed from the
preprocessing pipeline.
### 2.7 Principal component analysis
PCA was undertaken with various values and
was expected to be successful due to the high
correlation of data. The performance
decreased substantially however with PCA,
indicating the variance of the data was likely
lost in PCA. In the end, there were only 41
features, so it was likely too few for PCA to
function properly.
### 2.8 Handling class imbalance
It was observed that there was substantial class
imbalance, with the dataset being
predominantly skewed toward middle rated
movies with there only being 24 instances of
the 0 label in the dataset but 1792 of the 2
label. It is unknown whether this is
representative of the general population of
IMDB listings, however, to strengthen the
model it was deemed necessary to make
synthetic data that could accurately predict
these lower rated movies. SMOTE from the
imbLearn library was used for this. The result
was there being 1799 instances of each class.
This caused substantial overfitting, and hence
was disabled.
## 3 Models (results)
Hyperparameter tuning of each model was
done iteratively using the GridSearchCV
function of SKLearn.
### 3.1 - Logistic Regression
The pattern observed in logistic regression was
common among all models used. The models
can reliably predict the majority class, but face
immense struggles with those that aren’t
labelled 2. There is significant class imbalance
that is difficult to face. It did not perform too
poorly however, having a 5-fold CV accuracy
of 0.73. The F1 score for the 0 label was 0.08,
suggesting extremely poor performance in
predicting the minority class. Reasonable
performance despite this suggests that the data
is linearly separable by a decision boundary to
some extent, and due to the minimal data
engineering and preprocessing in general, that
the features are highly correlated. Given this, it
is likely that models which also utilise
decision boundaries will be useful in a
combination classifier to optimize accuracy
from this weak dataset.
</br></br>
**Figure 5- Confusion matrix for linear regression,
indicating great success at predicting label = 2, but
extreme lack of performance on 1, 3, and 0, which
were often predicted as 2 due to class imbalance.**
</br></br>
### 3.2 - Decision Tree
Decision tree performed noticeably worse than
logistic regression. This is suspected to be
because its methodology causes information
loss in this imbalanced dataset. The splits it
makes which are functionally decision
boundaries will be biased and lead to poor
representation of the minority class. Its
accuracy was 0.68 on 5-fold CV. This theory
was proven by the f1-score on the label 0
being 0.00, indicating extremely poor
performance on the minority class.
</br></br>
**Figure 6- Confusion matrix for decision tree
exhibiting a similar pattern to linear regression, but
with higher variance.**
</br></br>
### 3.3 - Random Forest
Random Forest performed the same as
Logistic regression with an accuracy of 0.73.
this is not informative, as the number of
classifiers used was 100, indicating that the
performance likely came from sheer number of
decision trees, as the f1-scores outside of the
one for the label 2 were extremely poor. This
informs that tree classifiers are not suitable for
the classification in this dataset, as if this
distribution applies to the greater population,
they will continue to fail. The f1-score for 0
was once again 0.00, indicating it never
predicts correctly, 0.2 for 1, 0.6 for 3, and 0.6
for 4. This highlights a key flaw of the tree
algorithms: a tendency to overfit.
### 3.4 - Support Vector
Support vector classification had extremely
similar performance to the logistic regression
with a 5 fold CV accuracy of 0.74. this
indicates that the dataset is likely linearly
separable, which logistic regression and
support vector classifiers both perform well
under. It was relatively easy to draw decision
boundaries with both, whether that was
through a hyperplane or a single split. This
model furthers the hypothesis that the data is
well clustered and hence linearly separable.
The lower performance of decision trees may
imply there is still some complexity in the
relationship that is misunderstood and may be
investigated in further research.
</br></br>
**Figure 7- Confusion matrix for support vector
classifier.**
</br></br>
### 3.5 - KNN
KNN had a performance of 0.63 accuracy,
which was higher than expected given its
simplicity and lack of training. KNN was
performed with K = 5, and even though there
was high class imbalance, it suggests that data
can be linearly grouped, hence why other prior
tested models performed better.
</br></br>
**Figure 8 - Confusion matrix for KNN, reinforcing
the observed pattern of class imbalance and so
similar to other matrices that a linear relationship
is possible**
</br></br>
### 3.6 - Naïve Bayes
Naïve Bayes had by far the worst
performance, at 0.19 accuracy . This is due to
the key failure that's an assumption of Naïve
Bayes, being that the features are unrelated.
The low MI score of individual features but
high accuracy of most models with cross
validation implies that the features are related,
and it is in fact their interconnection that
allows tree-based models and regression to
perform substantially better.
### 3.7 - Bagging and Voting
Bagging and voting were two combination
models used in conjunction with each other.
The increase in computation time to get an
accuracy of 0.73 in the CV couldn’t possibly
justify the use of this model in the future and is
suggested to be used once the base models
used to solve this problem well are more
understood. Its performance is incredibly hard
to discern due to the models involved, and
hence will not be analysed here.
## 4 - Final submission (discussion)
The final model was automatically chosen by
the program based on the top performance of
all models, this was always stacking with
random forest, logistic regression, and SVC.
More models could have been used in the
stacking, but the increase in performance was
not justifiable for the computational time.
With cross validation, this model resulted in
0.74 accuracy, but in the final test, it resulted
in 0.712 Each of the models analysed
informed that the data is reasonably linearly
correlated with the label of IMDB ratings
bagged, as observed in the KNN and Logistic
regression, and the features are highly related
as observed in random forest, decision trees,
and the poor performance of Naïve Bayes.
Future research can investigate the distribution
of the data further alongside this information
to pick the perfect model, as well as
understand the models tested in this report and
their suspected benefits and flaws to inform
better decisions. The results are informative
and may assist with research into picking a
specific model and preprocessing pipeline,
where this research focussed on general
analysis at the cost of reduced accuracy.
</br></br>
**Figure 9 – all 5-CV accuracies**
</br></br>
## 5 - Conclusions
Ultimately, with the small dataset the model
chosen ended up being of marginal relevance.
Improvement in performance came down to
preprocessing. The small dataset combined
with vastly varying scales meant models
struggled to generalise to the test case, and the
presence of outliers alongside class imbalance
significantly reduced the performance of all
models. These issues were addressed in this
report to some success, but any significant
future improvement will come from the ability
to generate synthetic data to mitigate the
high-class imbalance and using background
research such as this one to optimize the
preprocessing pipeline and use of models
