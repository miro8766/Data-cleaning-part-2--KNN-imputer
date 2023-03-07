```python
import pandas as pd
```


```python
# load dataset
df =pd.read_csv(r'C:\Users\afard\Downloads\Airbnb_Open_Data.csv', na_values='?')
```


```python
df.head()
```


```python
df.info()
```


```python
#let's get rid of some of the useless non-numeric columns
df = df.drop(['host_identity_verified', 'host name', 'neighbourhood', 'country', 'country code', 'house_rules','license'], axis=1)

```


```python
df.head()
```


```python
# we have to fix first the last_review and then price and service fees. these are all object
df['last review'] = pd.to_datetime(df['last review'])
```


```python
print(df['last review'].dtypes)
```


```python
# we need to first remove the dollar
df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
```


```python
df['price'] = pd.to_numeric(df['price'])
```


```python
df['service fee'] = df['service fee'].str.replace('$', '').str.replace(',', '')
df['service fee'] = pd.to_numeric(df['service fee'])
```


```python
df.head()
```


```python
df.info()
```


```python
df = df.drop(['NAME'], axis=1)
```


```python
# we have some variables that need to be changed from object to numeric such as binary variables
df['neighbourhood group'].value_counts()
```


```python
df.loc[df['neighbourhood group']=='manhatan', 'neighbourhood group'] = 'Manhattan'
df.loc[df['neighbourhood group']=='brookln', 'neighbourhood group'] = 'Brooklyn'
```


```python
df['neighbourhood group'].value_counts()
```


```python
# first we are going to create a categorical number for each neighborhood, create a new var and delet the old one
df['Neighborhood'] = df['neighbourhood group'].astype('category').cat.codes
# it is better to jot down the name of each group because once you delete the original value you won't have it anymore
df = df.drop(['neighbourhood group'], axis=1)
```


```python
df.head()
```


```python
df['instant_bookable'].value_counts()
```


```python
df['cancellation_policy'].value_counts()
```


```python
df['room type'].value_counts()
```


```python
df['cpolicy'] = df['cancellation_policy'].astype('category').cat.codes
df['rtype'] = df['room type'].astype('category').cat.codes
```


```python
df = df.drop(['cancellation_policy','room type'], axis=1)
```


```python
# lastly i am changing the instant_bookable into 0 and 1

df['instantb'] = df['instant_bookable'].astype(bool).astype(int)
df = df.drop(['instant_bookable'], axis=1)
```


```python
df.info()
#finally everything is numeric
```


```python
# summarize the number of rows with missing values for each column
for i in range(dataframe.shape[1]):
 # count number of rows with missing values
 n_miss = dataframe[[i]].isnull().sum()
 perc = n_miss / dataframe.shape[0] * 100
 print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
```

## Nearest Neighbor Imputation with KNNImputer

"KNNImputer is a data transform that is first configured based on the method used to estimate the missing values.

The default distance measure is a Euclidean distance measure that is NaN aware, e.g. will not include NaN values when calculating the distance between members of the training dataset. This is set via the “metric” argument.

The number of neighbors is set to five by default and can be configured by the “n_neighbors” argument.

Finally, the distance measure can be weighed proportional to the distance between instances (rows), although this is set to a uniform weighting by default, controlled via the “weights” argument."

**SOURCE: https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/**


```python
import numpy as np
from sklearn.impute import KNNImputer
```


```python
# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
```


```python
df = df.drop(df[df['last review'] > pd.Timestamp('2022-12-31')].index)
```


```python
df.info()
```


```python
# Separate the datetime column from the numerical and categorical columns
datetime_col = df['last review']
df_numerical = df.select_dtypes(include=['float64','int64', 'int8', 'int32'])
df_categorical = df.select_dtypes(include=['object'])
```


```python
# Use the KNNImputer to impute missing values in the numerical columns
df_imputed = pd.DataFrame(imputer.fit_transform(df_numerical), columns=df_numerical.columns)
```


```python
# Combine the imputed numerical columns with the datetime and categorical columns
df_imputed = pd.concat([datetime_col, df_imputed, df_categorical], axis=1)
```


```python
df_imputed.shape
```


```python
df.shape
```


```python
missing_df_count = df.isnull().sum()
```


```python
# to get an overview of what 5 of our dataset is missing, we can use the following

total_obs = np.product(df.shape)
total_missing = missing_df_count.sum()


# percent of data that is missing
(total_missing/total_obs) * 100
```


```python

```
