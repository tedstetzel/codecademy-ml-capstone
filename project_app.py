import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

'''
- at least two graphs containing exploration of the dataset
- a statement of your question (or questions!) and how you arrived there 
- the explanation of at least two new columns you created and how you did it
- the comparison between two classification approaches, including a qualitative discussion of simplicity, time to run the model, and accuracy, precision, and/or recall
- the comparison between two regression approaches, including a qualitative discussion of simplicity, time to run the model, and accuracy, precision, and/or recall
- an overall conclusion, with a preliminary answer to your initial question(s), next steps, and what other data you would like to have in order to better answer your question(s)
'''



#Create your df here:
df = pd.read_csv("profiles.csv")
df_income = pd.read_csv("profiles.csv")



#two graphs containing exploration of the dataset
#print(df.diet.value_counts())


'''
| body_type      | bt_code     |
|----------------|-------------|
| thin           | 0           |
| skinny         | 0           |
| athletic       | 1           |
| fit            | 1           |
| jacked         | 1           |
| average        | 2           |
| curvy          | 3           |
| a little extra | 3           |
| full figured   | 3           |
| overweight     | 3           |
| used up        | 4           |
| rather not say | 5           |
'''

body_type_group_mapping = {"thin": 0, "skinny": 0, "athletic": 1, "fit": 1, "average": 2, "jacked": 1, "curvy": 3, "a little extra": 3, "full figured": 3, "overweight": 3, "used up": 4, "rather not say": 5}
#df.dropna(subset=['body_type'], how='any', inplace = True)

#df["bt_code"] = df.body_type.map(body_type_mapping)
#df["bt_group_code"] = df.body_type.map(body_type_group_mapping)

#print(df.bt_code.value_counts())

'''
| diet                | diet_code   |
|---------------------|-------------|
| anything            | 0           |
| mostly anything     | 0           |
| strictly anything   | 0           |
| halal               | 1           |
| mostly halal        | 1           |
| strictly halal      | 1           |
| kosher              | 2           |
| mostly kosher       | 2           |
| strictly kosher     | 2           |
| vegan               | 3           |
| mostly vegan        | 3           |
| strictly vegan      | 3           |
| vegetarian          | 4           |
| mostly vegetarian   | 4           |
| strictly vegetarian | 4           |
| other               | 5           |
| mostly other        | 5           |
| strictly other      | 5           |
'''


#df["diet_code"] = df.diet.map(diet_mapping)

diet_group_mapping = {"anything": 0, "mostly anything": 0, "strictly anything": 0, "halal": 1, "mostly halal": 1, "strictly halal": 1, "kosher": 2, "mostly kosher": 2, "strictly kosher": 2, "vegan": 3, "mostly vegan": 3, "strictly vegan": 3,
"vegetarian": 4, "mostly vegetarian": 4, "strictly vegetarian": 4, "other": 5, "mostly other": 5, "strictly other": 5}

diet_groups = ["anything", "halal","kosher", "vegan","vegetarian", "other" ]

#df["diet_group_code"] = df.diet.map(diet_group_mapping)

#Classification: Can we predict Body type based on Sex, Age, and Diet?

df['diet_group_code'] = df.diet.map(diet_group_mapping)
df['diet_group_code'] = df['diet_group_code'].replace(np.nan, 0, regex=True)

datapoints = df['income']
labels = df['diet_group_code']




#print(df.diet_group_code.value_counts())
'''
bt_explore = df.groupby('body_type').size()
bt_explore.plot(kind='bar')
plt.title("Frequecy of Body Type")
plt.xlabel("Body Type")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
'''

#Create Exploitory chart for Diet 
'''
diet_explore = df.groupby('diet').size()
diet_explore.plot(kind='bar')
plt.title("Frequecy of Diet")
plt.xlabel("Diet")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
'''

training_data, validation_data, training_labels, validation_labels = train_test_split(datapoints, labels, test_size=0.2, random_state = 99)

'''
accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  score = classifier.score(validation_data,validation_labels)
  accuracies.append(score)

k_list = []
for i in range(1,101):
  k_list.append(i)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Classifier Accuracy")
plt.show()
'''
#Regression: Do age and income increase?

regr = LinearRegression()
df_income = df_income[df_income.income  > 0]
df_income = df_income[df_income.age  < 55]

X = df_income['age'].replace(np.nan, 0, regex=True)
X = X.values.reshape(-1,1)
y = df_income['income']

regr.fit(X, y)
print('slope of the line:')

print( regr.coef_)
print('The intercept of the line:')
print(regr.intercept_)

y_predict = regr.predict(X)


plt.plot(X, y_predict)
plt.title("Age vs Income - 55 or Younger")
plt.xlabel("Age")
plt.ylabel("Income")
plt.tight_layout()
plt.show()



