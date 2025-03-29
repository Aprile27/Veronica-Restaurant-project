# %% [markdown]
# # **WEEK 1: DATA EXPLORATION**
# - Explore dataset dimensions. Check
# for missing values. Perform data
# type conversions as needed.
# - Analyze "Aggregate rating"
# distribution. Address any class
# imbalances.
# - Calculate statistics for numerical
# columns. Explore categorical
# variables. Identify top 5 cuisines
# and cities.

# %%
#import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#load data
df = pd.read_csv('https://raw.githubusercontent.com/Oyeniran20/axia_class_cohort_7/refs/heads/main/Dataset%20.csv')

# %%
df.head()

# %% [markdown]
# - Explore dataset dimensions

# %%
df.shape

# %%
df.info()

# %% [markdown]
# - check for missing values
# 

# %%
#checking for missing values
df.isna().sum().sort_values(ascending=False)

# %%
df.Cuisines

# %%
#finding the most common cuisine
most_common_cuisine = df['Cuisines'].mode()[0]

#fill missing values with the most common cuisine
df['Cuisines'].fillna(most_common_cuisine, inplace=True)

# %%
print(df['Cuisines'].isnull().sum())

# %%
print('Most common cuisine:', most_common_cuisine)

# %% [markdown]
# # **Observations**:
# - this data contains 9551 entry with 21 features
# - there are 9 missing rows at the cuisine columns
# - the most common cuisine is 'North Indian'
# - all features are with the right data types
# 

# %%
#checking aggregate rating distributon
plt.figure(figsize=(8, 5))
sns.histplot(df['Aggregate rating'], bins=20, kde=True, color='pink')
plt.title('Aggregate Ratings Distribution')
plt.xlabel('Aggregate Ratings')
plt.ylabel('Count')
plt.show()

# %%
rating_counts = df['Aggregate rating'].value_counts()

# %%
print('Rating distribution:\n', rating_counts)

# %% [markdown]
# ### Observations
# - most resturants have ratings between 2.5 and 4.0.
# - very few resturants have extreme ratings
# - the peak is around 3.2, which means most of the resturants recieve average ratings.

# %%
#class imbalancing
df['Rating Category'] = pd.cut(df['Aggregate rating'], bins=[0, 2.5, 3.5, 5],
                               labels=['Low', 'Medium', 'High'], include_lowest=True)

# %%
class_counts = df['Rating Category'].value_counts()
print('\nClass distribution before balancing:\n', class_counts)

# %%
X = df.drop(columns=["Aggregate rating", "Rating Category"])
y = df["Rating Category"]

# %%
df.Votes

# %%


# %%
# Statistics
columns = ["Aggregate rating", "Average Cost for two", "Price range", "Votes"]
numerical_stat = df[columns].describe().T
numerical_stat

# %%


# %% [markdown]
# ## **Top Cuisines and Cities**

# %%
# Top 5 most common cuisines
top_cuisines = df["Cuisines"].str.split(",").explode().str.strip().value_counts()
top_cuisines.head(5)

# %%
# plot
top_cuisines = df["Cuisines"].value_counts().head(5)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette="coolwarm")
plt.title("Top 5 Most Common Cuisines", fontsize=14)
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine Type")
plt.show()


# %% [markdown]
# **Top 5 Most Common Cuisines**
# 
# - North Indian is the most popular cuisine, followed by Chinese and Fast Food.
# - Mughlai and Italian food are also represented.
# 

# %%
top_cities = df["City"].value_counts()
top_cities.head(5)

# %%
#plot
top_cities = df["City"].value_counts().head(5)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_cities.values, y=top_cities.index, palette="inferno")
plt.title("Top 5 Cities with Most Restaurants", fontsize=14)
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.show()


# %% [markdown]
# # **Observations**
# - New Delhi dominates with the most resturants.
# - Gurgaon and Noida takes second place
# - smaller cities have fewer resturants.

# %% [markdown]
# # **Week 2: Data visualization**
# - Create histograms,
# bar plots, and box
# plots of ratings.
# Compare average
# ratings across
# cuisines and cities
# - Map restaurant
# locations using
# coordinates.
# Analyze distribution
# across cities.
# Correlate location
# with ratings
# - Identify outliers and
# their effects.
# Determine
# relationship
# between votes and
# ratings.
# 

# %% [markdown]
# ### **NOTE**: Ratings visualization are displayed above.

# %% [markdown]
# ## **Comparing average ratings accross Cuisines and City**
# 

# %%
#Average ratings for each cuisine
avg_rate_cuisines = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)

avg_rate_cuisines.head(10)

# %% [markdown]
# **Plotting the Average ratings accross cuisines**

# %%
avg_rate_cuisines = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(y=avg_rate_cuisines.index, x=avg_rate_cuisines.values, palette='coolwarm')
plt.title('Top 10 Cuisines with Highest Average Rate')
plt.xlabel('Average Rate')
plt.ylabel('Cuisines')
plt.show()

# %%
#Average ratings for each city
avg_rate_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
avg_rate_city.head(10)

# %% [markdown]
# **Plotting the average ratings across cities**

# %%
avg_rate_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(y=avg_rate_city.index, x=avg_rate_city.values, palette='magma')
plt.title('Top 10 Highest Average Rates')
plt.xlabel('Average Rates')
plt.ylabel('City')
plt.show()

# %% [markdown]
# **Observations**
# - Italian, Deli tend to have the higest average ratings in cuisines
# - philipine based cities like Quezon city, Makati city rank high.
# 

# %% [markdown]
# **MAPPING RESTURANT LOCATIONS WITH COORDINATES**

# %%
import folium
from folium.plugins import MarkerCluster

# Create a map centered around the average location
map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=5)

# Add restaurant locations as markers
marker_cluster = MarkerCluster().add_to(restaurant_map)

for index, row in df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f'{row["Restaurant Name"]} - Rating: {row["Aggregate rating"]}',
        icon=folium.Icon(color="purple" if row["Aggregate rating"] >= 4 else "orange")
    ).add_to(marker_cluster)


restaurant_map

# %% [markdown]
# ### **Identifying outliers and their effects**

# %%
#using boxplot for aggregate ratings
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Aggregate rating'])
plt.title('Boxplot for the Resturant Ratings')
plt.xlabel('Aggregate Rating')
plt.show()

# %%
# Histogram for Votes
plt.figure(figsize=(12, 6))
sns.histplot(df['Votes'], bins=30, kde=True)
plt.title('Distribution of Votes')
plt.xlabel('Number of Votes')
plt.ylabel('Count')

#for clearer visibility
plt.xlim(0, 5000)
plt.show()


# %%


# %% [markdown]
# **effects of outliers**
# - high voted resturant may have stronger customer engagements
# -some resturants have few number of votes making their ratings less reliable
# - removing outliers could make the vote analysis to be more stabled.
# -

# %% [markdown]
# ***Relationship between votes and ratings***

# %%
# Plot of Votes vs. Ratings
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Votes'], y=df['Aggregate rating'], alpha=0.6,)
plt.title('Votes vs. Ratings')
plt.xlabel('Number of Votes')
plt.ylabel('Aggregate Rating')

#for better visibility
plt.xlim(0, 5000)
plt.show()


# %% [markdown]
#  **Observations**
#  - The best visualization to use is a scatter plot because helps to directly show the vote ratings relationship better than the others.
# 
# - Resturants with more votes usually have higher ratings
# - Some high rated resturants have fewer votes(new buisness)

# %%
# correlation
correlation = df['Votes'].corr(df['Aggregate rating'])
correlation


# %% [markdown]
# **SUMMARY**:
# - most resturants have ratings between 2.5 to 4.0
# - higher priced restaurants tend to have better ratings
# - more votes generally leads to higher ratings.
# - the heatmap shows a lot of reataurant density across diffrent cities.

# %%


# %% [markdown]
# # **Week 3: Customer Preferences**
# - Cuisine Analysis: Identify highest-rated cuisines
# - Price Range: Compare ratings across price points
# - Service Features: Analyze table booking and delivery
# 
# Analyze relationships between cuisines and ratings. Identify popular cuisines by votes. Determine which price ranges
# receive highest ratings. Compare restaurants with and without table booking.
# 
# **Additional Insights**
# 
# - Table Booking Impact: Determine if table booking
# availability affects ratings
# across different cities.
# Compare average ratings
# with and without this
# feature.
# 
# - Online Delivery Analysis: Calculate percentage of
# restaurants offering
# delivery. Analyze availability
# across different price
# ranges.
# 
# - Customer Preferences: Identify specific cuisines that consistently receive higher
# ratings. Determine city-specific preferences
# 
# 

# %%
#cuisine analysis
# top 5 cuisines with highest average ratings
top_cuisine_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
top_cuisine_ratings.head(5)

# %%
#plotting average ratings of top cuisines
top_cuisine_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(5)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_cuisine_ratings.index, y=top_cuisine_ratings.values)
plt.xlabel('Cuisines')
plt.ylabel('Average Rating')
plt.title('Top cuisines by average ratings')
plt.show()

# %%
#top 10 most voted cuisines
top_cuisine_votes = df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
top_cuisine_votes.head(10)

# %%
#plotting for most voted cuisines
top_cuisine_votes = df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_cuisine_votes.index, y=top_cuisine_votes.values, palette='magma')
plt.xlabel('Cuisines')
plt.ylabel('Top votes')
plt.title('Top most voted cuisines')
plt.show()

# %% [markdown]
# ## **OBSERVATION**
# - this shows that North indian, Mughlai and Chinese cuisines are the most popular by votes.
# - italian ,deli & hawiian seafood cuisines have the highest ratings.

# %% [markdown]
# **Price Range**

# %%
price_range_ratings = df.groupby("Price range")["Aggregate rating"].mean()
price_range_ratings.head()

# %%
#Bar plot of price range and ratings
price_range_ratings = df.groupby("Price range")["Aggregate rating"].mean().head(5)
plt.figure(figsize=(10, 5))
sns.barplot(x=price_range_ratings.index, y=price_range_ratings.values, palette='coolwarm')
plt.xlabel('Price Range')
plt.ylabel('Aggregate Rating')
plt.title('Price Range on Ratings')
plt.show()

# %% [markdown]
# **COMMENT**: Customers tend to rate expensive restaurants higher, possibly due to better quality service and food.

# %% [markdown]
# ### **Analyzing Table Booking and Online Delivery**

# %%
table_booking_ratings = df.groupby('Has Table booking')['Aggregate rating'].mean()
table_booking_ratings.head()

# %%
#plotting
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Has Table booking'], y=df['Aggregate rating'])
plt.xlabel('AVAILABLE TABLE BOOKINGS')
plt.ylabel('Aggregate Rating')
plt.title('Table Booking of Ratings')
plt.show()

# %%
online_delivery_ratings = df.groupby('Has Online delivery')['Aggregate rating'].mean()
online_delivery_ratings.head()

# %%
#plotting
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Has Online delivery'], y=df['Aggregate rating'], palette='Set3')
plt.xlabel('Online Delivery Available')
plt.ylabel('Aggregate Rating')
plt.title('Online Delivery on Ratings')
plt.show()

# %% [markdown]
# **OBSERVATIONS**
# - Without table booking = 2.56
# - With table booking = 3.44
# 
# Therefore customers prefer restaurants with table bookings.(peharps due to better services)
# - Without online delivery = 2.46
# - With online delivery = 3.25
# 
# Therefore restaurants that offers online delivery tend to recieve higher ratings (peharps because of convenience).

# %% [markdown]
# ## **City Specific Preferences**

# %%
# Find the most common cuisine in each city
city_cuisine = df.groupby("City")["Cuisines"].agg(lambda x: x.mode())
city_cuisine.head()


# %%
# Compute average rating per city
city_avg_rating = df.groupby("City")["Aggregate rating"].mean().sort_values(ascending=False)
city_avg_rating.head()

# %%
# Merging both results
city_cuisine_rating = pd.DataFrame({'Most Popular Cuisine': city_cuisine, 'Average Rating': city_avg_rating})
city_cuisine_rating.head(10)

# %%


# %% [markdown]
# # **Week 4: Predictive Modeling**
# - Feature Engineering:
# Extract additional features from existing columns. Create new features by encoding categorical variables.
# - Model Building:
# Build regression models to predict restaurant ratings. Split data into training and testing sets.
# - Model Evaluation:
# Evaluate using RMSE, MAE, and R-squared. Compare different algorithms like linear regression and random forest.

# %% [markdown]
# **Feature engineering (encoding categorical variables)**

# %%
#importing necessary libraies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# %%
features = ['Votes', 'Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery']
target = 'Aggregate rating'

df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
X_train.shape

# %%
X_test.shape

# %% [markdown]
# Training set: 7,640
# Test set: 1,911
# 

# %% [markdown]
# **Linear Regression model**
# 

# %%
model = LinearRegression()
model.fit(X_train, y_train)

#predictions on the test set
y_pred = model.predict(X_test)

#evaluation metrics for linear regression
rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# %%
rmse

# %%
mae

# %%
r2

# %% [markdown]
# - Root Mean Squared Error = 1.29
# - Mean Absolute Error = 1.08
# - R-Squared Score = 0.26
# 
# **OBSERVATIONS**
# - On average the model's predications deviate bay 1.29 rating
# - The average absolute error in predictions ia 1.08 rating points.
# - The model explains 26% of the variance in the restaurant ratings, showing a weak predictive power.
# - The model is not very accurate so suggesting that restaurant ratings depends on more complex factors that aren't captured in the dataset.
# - Should most likely compare with more advanced models like Random forest.
# 

# %% [markdown]
# **Using Random Forest**
# 

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#predicting the test set
y_pred_rf = rf_model.predict(X_test)

#evaluation metrics for random forest model
rmse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# %%
rmse_rf

# %%
mae_rf

# %%
r2_rf

# %% [markdown]
# **OBSERVATIONS**
# - Root mean squared error = 0.37
# - Mean absolute error = 0.24
# - R-Squared score = 0.94
# 
# 
#   The model explains that 94% of the variance in restaurant ratings. meaning it is highly accurate in it's prediction.
# 
#   - Random Forest is a better option than Linear Regression.
#   - it also suggest that the restaurant ratings are highly influenced by complex non linear relationships which random forest captured very well.
#   **CONCLUSION**
#   Restaurant ratings are influenced by multiple factors, including service availability, pricing, and location-based preferences. Businesses can enhance ratings by focusing on convenience, quality, and data-driven decision-making. By implementing these strategies, restaurants can improve their reputation and customer satisfaction.

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



