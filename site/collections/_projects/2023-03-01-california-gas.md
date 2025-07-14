---
date: 2023-03-01 05:20:35 +0300
title: 'Modeling California Gas Prices Using Linear Regression'
subtitle: Regression Modeling + Plotly + Pandas + Scikit-learn + Python 
image: '/images/gas.png'
---
# Introduction

## Prologue

Gas prices in the United States have consistently been an area of interest due to heavy reliance on automotive transportation$^{[1]}$. Despite observable increase in EV usage in the US, California especially, gas expenses remain an issue for those who commute by internal combustion vehicles. The goal of this study was to make a small contribution to optimizing currently used modes of transportation. In doing so, I could explore processes of building a regression model.

In general, I was curious to see if price at a gas station was predictable given a set of parameters at a *fixed* time. This would serve as a precursor to predicting the price of a gas station given parameters at an **********indeterminate********** time. 

## Methodology

To establish a baseline for predicting gas prices intrinsically given fixed parameters, I aimed to gather data containing several parameters irrespective of time. 

To optimize feature selection, I examined properties, distributions, and associations with target variable of each potential feature. Features were dropped if they met any of the following criteria:

- Noisy:
    - Large number of possible categorical values that could not be grouped easily
- Feature $x$ has insignificant association with target variable $y$:
    - $f(x)=\hat y=ax+b$ where slope $a$ is significantly low
    - $f(x)=\hat y=ax+b$ where fit $R^2$ is significantly low
- Only had one possible value
- Aren’t practically involved
    - There is no physical attribution to target variable

This study covers development and assessment of a linear regression model. Other models should be considered beyond a linear regression, as natural occurrences are generally non-linear.

# Data Acquisition

I found available scraped data from the community gas pricing app, *******Gas Buddy*******, on *Kaggle$^{[2]}$*. This user-provided data includes individual gas station ratings, location, and current price (as of date scraped). The data was scraped from California on the date April 17, 2021. The time window of user-reported data for each gas station varied depending on the ******Gas Buddy****** search filter used by the scraper. 

## Properties

The CSV dataset contained 23 columns, including floating point, integer, and object (in this case, string) values. 

```python
Data columns (total 23 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   services_included  53737 non-null  object 
 1   price_time_stamp   53737 non-null  object 
 2   currency           53737 non-null  object 
 **3   postal_code        53699 non-null  object** 
 4   loc_name           53737 non-null  object 
 5   city               53737 non-null  object 
 **6   review_count       52907 non-null  float64**
 7   state              53737 non-null  object 
 8   zip_code_searched  53737 non-null  int64  
 9   latitude           53737 non-null  float64
 10  product_name       53737 non-null  object 
 11  payment_type       53737 non-null  object 
 12  DATE_SCRAPED       53737 non-null  object 
 13  RUN_START_DATE     53737 non-null  object 
 14  source_url         53737 non-null  object 
 **15  phone              50483 non-null  object** 
 16  loc_number         53737 non-null  int64  
 17  price_current      53737 non-null  float64
 18  country            53737 non-null  object 
 19  longitude          53737 non-null  float64
 20  address_1          53737 non-null  object 
 21  address_2          53727 non-null  object 
 **22  overall_rating     52907 non-null  float64**
```

Features highlighted in red have null values. While there weren’t many null values (Non-Null Count > 50000), these were handled first during feature selection. Features highlighted in yellow are features related to the scrape, and not the source data. Those features were dropped as they were not directly attributed to gas prices, and the method of data acquisition will vary in the model’s application.

Additionally, I considered the number of possible values in each column:

```python
services_included     2932
price_time_stamp     11960
currency                 1
postal_code           8508
loc_name              1292
city                  1010
review_count           469
state                    2
zip_code_searched      547
latitude              9459
product_name             6
payment_type             2
DATE_SCRAPED          1472
RUN_START_DATE           1
source_url            9471
phone                 8605
loc_number            9471
price_current          283
country                  1
longitude             9460
address_1             9446
address_2             5826
overall_rating          40 
```

Below is a table containing more detailed information about each column.

| Feature | Data Type | Kind | Description | Example(s)                                          |
| --- | --- | --- | --- |-----------------------------------------------------|
| services_included | str | Categorical | List of services provided at station | 'Regular, Midgrade, Premium, C-Store, Pay At Pu...' |
| price_time_stamp | str | Ordinal | Datetime | 2022-04-17 14:49:58                                 |
| currency | str | Categorical | Current price currency abbreviation | 'USD'                                               |
| postal_code | str | Categorical | Station zip code | '94015', '90001-2731'                               |
| loc_name | str | Categorical | Station name | '7Eleven'                                           |
| city | str | Categorical | Station city | 'Los Angeles'                                       |
| review_count | float64 | Numeric | Number of listing reviews | 432                                                 |
| state | str | Categorical | Station state abbreviation | CA                                                  |
| zip_code_searched | int64 | Categorical | Zip code searched in scrape | 94015                                               |
| latitude | float64 | Numeric | Station latitude | 33.974931                                           |
| product_name | str | Categorical | Product name for listed price | 'Regular'                                           |
| payment_type | str | Categorical | Method for listed price | 'Credit'                                            |
| DATE_SCRAPED | str | Ordinal | Datetime of scrape | 2022-04-18T05:01:23+8:00                            |
| RUN_START_DATE | str | Ordinal | Datetime scraping for the record started | 2022-04-18T05:01:23+8:00                            |
| source_url | str | Descriptive | URL record scraped from | ‘https://www.gasbuddy.com/Station/10451'            |
| phone | str | Descriptive | Station phone number | '323-582-6158'                                      |
| loc_number | int64 | Categorical | Number corresponding to loc_name | 10451                                               |
| price_current | float64 | Numeric | Price reported at station for given product_name | 5.49                                                |
| country | str | Categorical | Station country abberviation | 'US'                                                |
| longitude | float64 | Numeric | Station longitude | -118.237916                                         |
| address_1 | str | Descriptive | Station address 1 | '1935 E Florence Ave'                               |
| address_2 | str | Descriptive | Station address 2 | 'Wilson Ave'                                        |
| overall_rating | float64 | Numeric | Overall user rating of station, out of five | 3.8                                                 |

# Exploratory Analysis

To de-noise and process data for training, it was necessary to drop features. A possible method for determining what features to drop and what preprocessing methods to use is through exploratory analysis. Specifically, I looked for associations between datapoints and any features with data that was too granular. 

## Distributions

First, I examined some feature distributions to get an idea of whether the variable has an observable distribution or if it’s noisy.

### services_included

Distribution after label encoding and standardizing:

![Code 2.1](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled.png)

### city

Raw distribution, after label encoding:

![Code 2.8](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%201.png)

### overall_rating

Raw distribution:

![Code 2.3](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%202.png)

### loc_number

Raw distribution, after label encoding:

![Code 2.5](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%203.png)


### payment_type

Raw boxplot distribution relative to price: 

![Code 2.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%204.png)

### review_count

Raw distribution:

![Code 2.4](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%205.png)

### loc_name

Raw distribution, after label encoding:

![Code 2.6](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%206.png)


### product_name

Raw boxplot distribution relative to price: 

![Code 2.9](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%207.png)


### price_current

Distribution of non-zero gas prices, colored by type of gas:

![Code 2.10](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%208.png)

## Target-Feature Relationships

The next part I looked at in the features was relationships between features themselves and the target variable (gas price in our case). Hence, “target-feature relationships”. 

### price_current vs review_count

Association between price and number of reviews. The trendline below ($R^2=0.0228$) was calculated in `plotly` using OLS (ordinary least squares), with current price $p$ and reviews $r$:

$$
p=(-0.6645\times10^{-3})r+5.9429
$$

![Code 2.11](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%209.png)

The trend was notable. For exploratory purposes and level of noise in the data, $R^2=0.0228$ was considered significant. Therefore, there was an association between gas price and number of reviews. It should be noted there is a cluster of outlier visible.

### price_current vs overall_rating

Association between price and overall rating. The trendline below ($R^2=0.0011)$ was calculated in `plotly` using OLS (ordinary least squares), with current price $p$ and rating $a$:

$$
p=-0.0254a+5.9882
$$

![Code 2.12](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2010.png)

Contrary to presupposition, $R^2=0.0011$ did not suggest a significant relationship between gas price and rating. 

However, the smaller cluster $(r,p) \in ((3,5), (3,5))$ is worth nothing in another analysis— why would there be a cluster of positively reviewed gas stations, with lower prices than others, separate from the other “main” cluster?

### review_count vs overall_rating

Association between number of reviews and overall rating. The trendline below ($R^2=0.0121)$ was calculated in `plotly` using OLS (ordinary least squares), with number of reviews $r$ and rating $a$:

$$
r=19.207a+6.014
$$

![Code 2.13](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2011.png)

Since there was a difference in association between price and rating/review variables, it was worth looking for an association between review and rating. While there was some association between the two, it appears there may have been a polynomial relationship between the two. This may have explained the difference between the features’ association significance with gas price.

### price_current vs location

Association between price and latitude of the gas station. 

The trendlines below were calculated in `plotly` using OLS (ordinary least squares), with current price $p$ and latitude $c_a$ ($R^2=0.0128)$, longitude $c_b$ ($R^2=0.0047)$, respectively:

$$
p=-0.022c_a+6.693
$$

$$
p=0.014c_b+7.661
$$

![Code 2.14](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2012.png)

![Code 2.15](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2013.png)

There were small linear associations between price and location. Notably, the latitudinal location.  

However, there were similar outlier clusters as there was in overall rating. This helped answer the previous question of why there was a similar cluster of outliers in the relationship. I considered if there may have been gas stations that were located in a distinct area, where the areas could have vastly different tax policies, suppliers, demand, etc.

To see if this was the case, I plotted the coordinates of each station:

![Code 2.16](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2014.png)

### E85 Prices

It was difficult to see where outliers were from the previous map. Adding filter $p<5$ and coloring points by `product_name` resulted in this:

![Code 2.17](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2015.png)

First, it should be noted that E85 color was at the top layer in the map. Therefore, it was not possible to view underlying locations, such as those containing Regular gas. 

The outlier was clearly visible. E85 prices in the Los Angeles area observably represented the outlier cluster I saw in multiple relationships. Looking back at the distribution of `price_current` created by Code 10 and the `product_name` distribution created by Code 9, there was an outlying distribution for E85 prices, distinct from the rest of products.

![Code 2.10](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%208.png)

![Code 2.9](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%207.png)

During feature selection, E85 prices weren’t necessarily be excluded or normalized to prevent offset. In the case of predicting E85 prices, product was notably relevant. I suspected it may have been more difficult for the model to predict an E85 price given other parameters, especially if E85’s significant price difference depends solely on the fact it was E85 (generally, E85 is much cheaper). 

### price_current vs loc_name

For this part, `loc_name` was encoded using `LabelEncoder()` from scikit-learn (see Code 7). There were so many unique values of `loc_name` that coloring based on `loc_name` would not reveal anything when plotted other than confetti.

Since `loc_name` was encoded and wasn’t ordinal, I didn’t use a trendline to look for trends. Instead, I immediately noticed there are a few values that 哈d a large spread of price. This suggested much more price variance in certain locations (in this context, companies).

![Code 2.18](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2016.png)

`loc_name` was observerably relevant. Instead of using `loc_number` which was assumed to be *Gas Buddy’s* method of encoding `loc_name` across all locations, it was ideal to use a standard scaled encoded version of `loc_name`.

# Feature Selection

## Drops

As discussed earlier and in good practice, the feature selection was done by first *********dropping********* features rather than adding them. The following features were ********dropped******** based on feature metadata and exploratory analysis:

| Feature | Reason |
| --- | --- |
| price_time_stamp | Range too small to be generalizable |
| currency | Only one possible value: 'USD' |
| postal_code | Too many unique values (too noisy), city has around 850% less unique values and would be a better choice |
| state | Only one possible value: 'CA' |
| zip_code_searched | Not relevant, data is from scraping |
| DATE_SCRAPED | Not relevant, data is from scraping |
| RUN_START_DATE | Not relevant, data is from scraping |
| source_url | Not relevant, data is from scraping |
| phone | Too many unique values after encoding (too noisy) |
| loc_number | Encoded version of loc_name but not standardized |
| country | Only one possible value: 'US' |
| address_1 | Too many unique values after encoding (too noisy) |
| address_2 | Too many unique values after encoding (too noisy) |
| overall_rating | No significant association with target variable |

## Features

Remaining features were processed for training:

| Feature | Encoded | Scaled | Normalized |
| --- | --- | --- | --- |
| services_included | Yes | Yes | No |
| loc_name | Yes | No | No |
| city | Yes | No | No |
| review_count | No | No | No |
| latitude | No | No | No |
| product_name | Yes | No | No |
| payment_type | Yes | No | No |
| longitude | No | No | No |

Processed training dataset metadata:

```python
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   services_included  37375 non-null  float64
 1   loc_name           37375 non-null  int64  
 2   city               37375 non-null  int64  
 3   review_count       37375 non-null  float64
 4   latitude           37375 non-null  float64
 5   product_name       37375 non-null  int64  
 6   payment_type       37375 non-null  int64  
 7   price_current      37375 non-null  float64
 8   longitude          37375 non-null  float64
dtypes: float64(5), int64(4)
```

# Model Training - Approach I

In this section, I detail the training process and any feature adjustments that were made. 

## Performance Statistics

The mathematical functions for performance metrics (mean square error, mean error, variance, and adjusted mean square error) were respectively defined as:

$$
MSE=\frac{1}{n}\sum_{i=0}^{n}(y_i-\hat y_i )^2
$$

$$
ME=\sqrt{MSE}
$$

$$
MPE=\frac{ME}{2\lambda\sigma_{y_i}}
$$

$$
var = \sigma^2_{y_i-\hat y}=\sigma^2_E
$$

$$
MSE_{adj}=\frac{ME}{2\lambda\sigma_E}
$$

with length of target vector $n$, index $i$, actual target $y_i$, estimated target $\hat y_i$, standard deviation of error  $\sigma_E$ and sigma y_i, and standard deviation constant $\lambda$.

The standard deviation constant determines the number of standard deviations from the mean $MSE$ should be evaluated against. $MSE_{adj}$ is a measure of error ***********in context*********** with the target variable. It considers the average error in a given range $\pm\lambda\sigma_E$. That is, $\hat y_i$  tends to be off by $ME=\sqrt{MSE}$, and therefore percent error in range $\pm \lambda\sigma_E$ is defined by 

$$
MSE_{adj}=\frac{ME}{2\lambda\sigma_E}
$$

In practice, the model predicts $y_i$ being off by $\pm ME$ on average. This prediction would be $MSE_{adj}$ percent of the target variable values within $\pm \lambda\sigma_E$.

## Round One

### Summary

For the first round, the data was split 50% training and 50% validation datasets. 

The training resulted in the following coefficients and performance:

```python
WEIGHTS---------------------
services_included    -0.00431103797913138
loc_name             2.5146702856355995e-05
city                 9.093322109624498e-06
review_count         -0.0007604982817237283
latitude             -0.0743771952029833
product_name         -0.11080503056641938
payment_type         0.11359291876034668
longitude            -0.054464306101689355

PERFORMANCE----------------
Mean Square Error    0.14230670918075836
Mean Error           0.3772356149421186
Mean Percent Error   0.44929925024792267
Error Variance       0.14230695308823443
Adjusted MSE         0.499999571511484
```

Mathematically, assuming $\lambda = 1$, the metrics were:

$$
MSE=\frac{1}{n}\sum_{i=0}^{n}(y_i-\hat y_i )^2 =0.1423
$$

$$
ME=\sqrt{MSE}=0.3772
$$

$$
MPE=\frac{ME}{2\lambda\sigma_y}=0.4493
$$

$$
\sigma_E^2 = 0.1423
$$

$$
MSE_{adj}=\frac{ME}{2\lambda\sigma_E}= 0.4999
$$

### Performance

The performance of this model was not remarkable. According to $ME$ and $MPE$, the model predicted $y_i$ incorrectly by approximately 45% of $y_i$ ($0.38) in range $\pm\sigma_y$. This means the predicted price within one standard deviation of the mean was almost one standard deviation off on average.

The distribution of the difference between the actual and estimated target variable $y_i-\hat y_i$ is shown below. Notice the small group in $(-3, -2)$. What caused this noticeable error in prediction?

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2017.png)

$MSE_{adj}$ and MPE across values of $\lambda \in [1, 4]$ (that is, up to four standard deviations $4\sigma$ from the mean error), is shown below. Notice the mean error **********decreased********** and therefore performance *********increased********* as the range grows. This was contrary to the original statement made about a 45% mean error in the first standard deviation an inaccuracy.

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2018.png)

Notice values for $MPE$ and $MSE_{adj}$ converge as the number of standard deviations $\lambda$ increased in the range of possible $y_i \in (\pm\lambda\sigma)$. Values decreasd as $\lambda$ increased. This suggests the model predicted more accurately as it moved further from the mean. However, this also means the model may have been overfitting.

Therefore, the performance of the model was problematic because there was higher error where the model is generalized and lower error when predicting outliers. This suggests possibility of overfitting. This was potentially related to E85 price outliers.

### Coefficients

The coefficients had a crucial role in determining feature importance and improving performance. If the coefficient of a feature was significantly small compared to the other features, then its overall importance to the model was insignificant. Therefore, to remove noise, that feature was dropped. 

In this case, I dropped:

```python
loc_name             2.5146702856355995e-05
city                 9.093322109624498e-06
```

## Round Two

The data was split 50% training and 50% validation datasets. 

Training resulted in the following coefficients and performance:

```python
WEIGHTS---------------------
services_included    -0.004530089520414092
review_count         -0.0007616799643414677
latitude             -0.07411481781125721
product_name         -0.11083770106503381
payment_type         0.11394799105571007
longitude            -0.05457979902439968

PERFORMANCE----------------
Mean Square Error    0.1423210683033398
Mean Error           0.3772546464966864
Mean Percent Error   0.4493219174162908
Error Variance       0.14232106952576778
Adj MSE Performance  0.4999999978526933
```

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2019.png)

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2020.png)

In this case, the performance was almost identical to round one. This was expected, due to the insignificant coefficients on the dropped features from round one.

After dropping further values, including `loc_name`, `review_count`, `services_included`, there was no significant gain in performance. In fact, the performance worsened:

$$
MSE_{adj}\in(0.499, 0.51)
$$

## Round Three

The model appeared to be generalizing well but not performing well. This indicated there may have been an inadequate number of features or incorrect feature selection entirely. I first tried including ****all**** features, scraper data excluded. Then, I encoded features using `LabelEncoder` if the feature was a string, yielding the following result:

```python
WEIGHTS---------------------
services_included    -0.006080094325712118
price_time_stamp     -7.652457292853966e-06
postal_code          -2.4682812213000834e-05
loc_name             1.0593781082173985e-05
city                 7.299525766148018e-06
review_count         -0.0007095705300390016
latitude             -0.045786318498422074
product_name         -0.11066987393028102
payment_type         0.1129598937297415
phone                1.5364280558249126e-05
loc_number           -4.784908412533288e-07
longitude            -0.05164099187263918
address_1            -8.649758433449654e-08
address_2            -4.420264224776627e-06
overall_rating       -0.019539501622791316

PERFORMANCE----------------
Mean Square Error    0.14047041691015122
Mean Error           0.37479383254017296
Mean Percent Error   0.44639101210972065
Error Variance       0.1404709672192695
Adj MSE Performance  0.4999990205989391
```

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2021.png)

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2022.png)

Below is a comparison of previous training round performance and this one. The best performing statistics are highlighted in green, which were unanimously in Round 3:

```python
ROUND 1
Mean Square Error    0.14230670918075836
Mean Error           0.3772356149421186
Mean Percent Error   0.44929925024792267
Error Variance       0.14230695308823443
Adjusted MSE         0.499999571511484

ROUND 2
Mean Square Error    0.1423210683033398
Mean Error           0.3772546464966864
Mean Percent Error   0.4493219174162908
Error Variance       0.14232106952576778
Adj MSE Performance  0.4999999978526933

ROUND 3
Mean Square Error    0.14047041691015122
Mean Error           0.37479383254017296
Mean Percent Error   0.44639101210972065
Error Variance       0.1404709672192695
Adj MSE Performance  0.4999990205989391
```

However, it is important to note that all of these differences were minute. Changes were $\pm 0.003$ at most. 

# Model Training - Approach II

As a second approach, I decided to retain the same performance metrics but use a different method for feature selection. This approach was based on one-hot encoding features. Revisiting feature metadata, most features were of high dimensionality:

```python
services_included     2932
price_time_stamp     11960
postal_code           8508
loc_name              1292
city                  1010
review_count           469
latitude              9459
product_name             6
payment_type             2
phone                 8605
loc_number            9471
longitude             9460
address_1             9446
address_2             5826
overall_rating          40 
```

I started processing by re-engineering features to have less possible values:

| Feature          | Derived Feature(s)         | Reduction ratio | Description                                                               |
|------------------|----------------------------|-----------------|---------------------------------------------------------------------------|
| price_time_stamp | price_month, price_weekday | 0.999           | price_month: month (integer)<br/>price_weekday: day of the week (integer) |
| postal_code      | zip2                       | 0.988           | Digits 2 and 3 of postal code (integer)                                   |

Then, I dropped the `phone` feature, as values did not follow a consistent syntax and parsing was more complex than expected, therefore not time efficient. 

After dropping values that were not numeric or had too many features, the remaining features were:

```python
review_count       468
latitude          7673
product_name         6
payment_type         2
price_current      281
longitude         7671
overall_rating      39
price_weekday        5
price_month          1
zip2                97
```

## Round Four

Data was split into training and validation sets at a 0.5 ratio. Any string features were one-hot encoded.

```python
WEIGHTS---------------------
review_count         -0.000691065651205967
latitude             -0.059624871858608486
longitude            -0.044003733284796315
overall_rating       -0.03772778683760426
product_name_Diesel  20173515.667534664
product_name_E85     20173515.613046616
product_name_Midgrade 20173515.665604018
product_name_Premium 20173515.65938174
product_name_Regular 20173515.669795394
product_name_UNL88   20173515.618638746
payment_type_Cash    9460663.077685451
payment_type_Credit  9460663.090123942
payment_type_nan     2379783.527764337
price_weekday_2      -6841421.024984235
price_weekday_3      -6841421.036985903
price_weekday_4      -6841421.116363608
price_weekday_5      -6841421.0725848265
price_weekday_6      -6841421.092495415
price_weekday_nan    -560327.3956550999
price_month_4        177450.30075325648
price_month_nan      -995137.8220946082
zip2_00              1491785.2896409
zip2_01              1491785.2832014658
zip2_02              1491785.296428862
zip2_03              1491785.271096426
...
zip2_97              1491785.4190337956
zip2_98              1491785.2780059823
zip2_99              1491785.318603877
zip2_n               1491785.2915149732
zip2_nan             0.0

PERFORMANCE----------------
Mean Square Error    0.16613992835019825
Mean Error           0.4076026598909755
Mean Percent Error   0.4903045100555646
Error Variance       0.16612460377932686
Adj MSE Performance  0.5000230613303597
```

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2023.png)

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2024.png)

```python
ROUND 3
Mean Square Error    0.14047041691015122
Mean Error           0.37479383254017296
Mean Percent Error   0.44639101210972065
Error Variance       0.1404709672192695
Adj MSE Performance  0.4999990205989391
```

After this training round, performance was still slightly worse than that of Round 3. However, $MPE$ and $MSE_{adj}$ followed each other very closely. That is, the difference between the two as $\lambda$ increased was approximately 0.01. Compared to other rounds where the gap was larger, these trends suggested that the performance was more closely associated with the percent error than in previous rounds. 

Looking at the coefficients in this round, I noticed they were much larger than that of previous rounds. This was due to the nature of one-hot encoding where features only had binary values of 0 or 1. The model needed to compensate for this by applying weights in correlation with the number of one-hot features derived from a feature. 

Therefore, to prepare for the next round of training, I used only the second digit of `postal_code` to generalize more **[Code 26]. Therefore, there was approximately 10% of one-hot features derived from `postal_code` than existed previously.

## Round Five

Data was split into training and validation sets at a 0.5 ratio. Any string features were one-hot encoded.

```python
WEIGHTS---------------------
review_count         -0.0006811546668406076
latitude             -0.06281044883265349
longitude            -0.04304913337469244
overall_rating       -0.04014679251403738
product_name_Diesel  0.019994130906086396
product_name_E85     -0.03065226194202552
product_name_Midgrade 0.019612415682188238
product_name_Premium 0.013004303855560393
product_name_Regular 0.024532528612474784
product_name_UNL88   -0.04649111711428439
payment_type_Cash    -0.005946844412835399
payment_type_Credit  0.005946844412835438
payment_type_nan     9.540979117872439e-17
price_weekday_2      0.02383624765638306
price_weekday_3      0.03154011054874849
price_weekday_4      -0.03879190674409882
price_weekday_5      0.0017927688086668283
price_weekday_6      -0.018377220269699516
price_weekday_nan    1.734723475976807e-18
price_month_4        2.0816681711721685e-17
price_month_nan      1.2414114874959026e-17
zip2_0               0.0024084866763303633
zip2_1               -0.005692229525371817
zip2_2               -0.01910903673875078
zip2_3               -0.03610737245113943
zip2_4               -0.02026117411922948
zip2_5               0.0009144515032055327
zip2_6               0.002290758639481109
zip2_7               0.028382142318244204
zip2_8               0.004383286605656246
zip2_9               0.02203029005837197
zip2_n               0.020760397033202253
zip2_nan             0.0

PERFORMANCE----------------
Mean Square Error    0.16618296785931672
Mean Error           0.40765545238511985
Mean Percent Error   0.4903680140522828
Error Variance       0.16617724469515113
Adj MSE Performance  0.5000086099557438
```

Notice the weights were far less extreme than in the previous round. As expected, reducing the number of features dramatically reduced the extreme weight values. 

Now that there were relatively stable weights, I dropped features based off weights:

```python
price_month_4        2.0816681711721685e-17
review_count         -0.0006811546668406076
```

Dropping `price_month` and `review_count` features resulted in:

```python
WEIGHTS---------------------
latitude             -0.05483102974939453
longitude            -0.03929415576564802
overall_rating       -0.05210422471316397
product_name_Diesel  0.022292454390816552
product_name_E85     -0.02838790745609404
product_name_Midgrade 0.02203713587995084
product_name_Premium 0.018147720530396975
product_name_Regular 0.028979769474557653
product_name_UNL88   -0.06306917281962854
payment_type_Cash    -0.005524391106947284
payment_type_Credit  0.005524391106947055
payment_type_nan     3.2959746043559335e-17
price_weekday_2      0.028852329568320206
price_weekday_3      0.03399916386734846
price_weekday_4      -0.041592605783037216
price_weekday_5      -0.0010280424420649528
price_weekday_6      -0.020230845210566735
price_weekday_nan    -2.7755575615628914e-17
zip2_0               0.0024469808293921527
zip2_1               -0.004309589041327545
zip2_2               -0.01879932895279793
zip2_3               -0.03884421947517518
zip2_4               -0.02911194500004397
zip2_5               0.0017616726045597152
zip2_6               -0.0037304668155399615
zip2_7               0.02277627009154581
zip2_8               0.0015641900227241865
zip2_9               0.011484847479073659
zip2_n               0.05476158825758912
zip2_nan             0.0

PERFORMANCE----------------
Mean Square Error    0.17047609567180935
Mean Error           0.41288750970671095
Mean Percent Error   0.4966616464400879
Error Variance       0.1704725712996516
Adj MSE Performance  0.5000051685058705
```

![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2025.png)
![Code 3.2](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2026.png)

Compared to the previous training round, this resulted in lower performance. Additionally, notice the gap between $MPE$ and $MSE_{adj}$ has decreased. Recall that approximated gap was previously 0.01, and in this round it was 0.05. 

Dropping `payment_type` and `zip2` had similar results:

```python
PERFORMANCE----------------
Mean Square Error    0.17055956566465952
Mean Error           0.41298857812857187
Mean Percent Error   0.4967832214638522
Error Variance       0.17055333344097498
Adj MSE Performance  0.5000091352168653
```

## Evaluation

Here is a comparison of results from the previous rounds:

```python
ROUND 1----------------
Mean Square Error    0.14230670918075836
Mean Error           0.3772356149421186
Mean Percent Error   0.44929925024792267
Error Variance       0.14230695308823443
Adjusted MSE         0.499999571511484

ROUND 2----------------
Mean Square Error    0.1423210683033398
Mean Error           0.3772546464966864
Mean Percent Error   0.4493219174162908
Error Variance       0.14232106952576778
Adj MSE Performance  0.4999999978526933

ROUND 3----------------
Mean Square Error    0.14047041691015122
Mean Error           0.37479383254017296
Mean Percent Error   0.44639101210972065
Error Variance       0.1404709672192695
Adj MSE Performance  0.4999990205989391

ROUND 4----------------
Mean Square Error    0.16613992835019825
Mean Error           0.4076026598909755
Mean Percent Error   0.4903045100555646
Error Variance       0.16612460377932686
Adj MSE Performance  0.5000230613303597

ROUND 5.1----------------
Mean Square Error    0.16618296785931672
Mean Error           0.40765545238511985
Mean Percent Error   0.4903680140522828
Error Variance       0.16617724469515113
Adj MSE Performance  0.5000086099557438

ROUND 5.2----------------
Mean Square Error    0.17047609567180935
Mean Error           0.41288750970671095
Mean Percent Error   0.4966616464400879
Error Variance       0.1704725712996516
Adj MSE Performance  0.5000051685058705

ROUND 5.3----------------
Mean Square Error    0.17055956566465952
Mean Error           0.41298857812857187
Mean Percent Error   0.4967832214638522
Error Variance       0.17055333344097498
Adj MSE Performance  0.5000091352168653
```

Unsurprisingly, the model in Round 3 performed the best (values highlighted in green). The worst performing model was from Round 5 (values highlighted in red). 

For prediction tests, I used something close to the Round 3 model (excluding `address_1, address_2` features) since that scored the highest. The training resulted in:

```python
WEIGHTS---------------------
postal_code          -2.6973089476740057e-05
loc_name             7.622216565954395e-06
city                 9.301117670492063e-06
review_count         -0.0007587656938362128
latitude             -0.04383266193454007
product_name         -0.11204501975386945
payment_type         0.11596150784419308
phone                1.3424202596577471e-05
loc_number           -4.62540213410143e-07
longitude            -0.05025225668878873
overall_rating       -0.01819079701964795

PERFORMANCE----------------
Mean Square Error    0.14895945706472152
Mean Error           0.38595266168886766
Mean Percent Error   0.4526450696391125
Error Variance       0.14897869703715916
Adj MSE Performance  0.49996771250848143
```

where the features were of the following types:

```python
#   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   postal_code     37375 non-null  int64  
 1   loc_name        37375 non-null  int64  
 2   city            37375 non-null  int64  
 3   review_count    37375 non-null  float64
 4   latitude        37375 non-null  float64
 5   product_name    37375 non-null  int64  
 6   payment_type    37375 non-null  int64  
 7   phone           37375 non-null  int64  
 8   loc_number      37375 non-null  int64  
 9   price_current   37375 non-null  float64
 10  longitude       37375 non-null  float64
 11  overall_rating  37375 non-null  float64
```

# Predictions

To develop predictions, I re-created a Gasbuddy webscraper (see Code 4.1) to retrieve real-time (station info updated in the last 24 hours) features used in training data. To avoid making too many requests, I limited the search to 25 zip codes in CA, and randomly selected between 90000 and 96000. 

Before processing the scraped features for the model, the features looked like:

```python
#   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   postal_code     264 non-null    object 
 1   loc_name        264 non-null    object 
 2   city            264 non-null    object 
 3   review_count    264 non-null    int64  
 4   latitude        264 non-null    object 
 5   product_name    264 non-null    object 
 6   payment_type    264 non-null    object 
 7   phone           264 non-null    object 
 8   loc_number      264 non-null    object 
 9   longitude       264 non-null    object 
 10  overall_rating  264 non-null    float64
```

To process the scraped features, I reordered and formatted the features to be the same as those used to train the data. After the processing was done, the features looked like:

```python
#   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   postal_code     570 non-null    int64  
 1   loc_name        570 non-null    int64  
 2   city            570 non-null    int64  
 3   review_count    570 non-null    int64  
 4   latitude        570 non-null    float64
 5   loc_number      570 non-null    int64  
 6   phone           570 non-null    int64  
 7   longitude       570 non-null    float64
 8   payment_type    570 non-null    int64  
 9   overall_rating  570 non-null    float64
 10  product_name    570 non-null    int64
```

Once processed, I made predictions using the regression model and analyzed the error in those predictions.

Error was defined by:

$$
E_r=y_r-\hat y_r
$$

The real-time test data was scraped on March 26 2023 11:30AM (Brisbane GMT+10), therefore having the range of March 24 2023 6:30PM (San Francisco GMT-7) to March 25 2023 6:30PM (San Francisco GMT-7). 

The distribution of $E_r$ looked like:

![Code 6.1](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2027.png)

This meant the model’s predictions were almost always above the actual price, generally by USD $1.00, with exception of the few outliers around USD -$3.10. In context with crude oil prices:

![Chart from Statista [3]](https://github.com/aspenflow/gas/raw/main/data/Modeling%20California%20Retail%20Gas%20Prices%20Using%20Linear%209d86d3ad466548c0a3f40caf0d47abed/Untitled%2028.png)

In April, the closing price of crude oil per barrel was approximately USD $60. As of the scraped data retrieved in March 2023, the closing price was USD $75. 

Interestingly, predictions made using this model were ********greater******** than the actual prices by approximately USD $1.00. This was likely because price of gas depends on more features than provided in this study. Note this is approximately twice the value of $ME$, meaning predictions on real-time data given fixed training data are approximately USD $1.00 off on average.

# Conclusions

Training a linear regression to predict gas prices given implicit data on the gas stations only has limited output performance. Another model may be more suitable for predicting gas prices when the model is only trained using data from *********Gas Buddy*********. Data from external sources on the gas stations, raw material sourcing, and tax rates could be useful for a linear regression to predict the prices more accurately. Alternatively, both addition of external features and use of a different model could have improved performance. Unsurprisingly like any other case, increasing sample size would yield more precise results, and 

Overall, a linear regression based on features provided by **********Gas Buddy********** from a single date and state are not enough to make accurate predictions on future gas prices. In practice, however, if a model is overcompensating for something causing a prediction that is approximately 20% above the actual cost of gas, the model could be usable for personal finance applications or budgeting in general. 

The research helped my understanding of building models, properties of linear regression, and feature engineering— knowledge that can be applied later to more complex models.

# Limitations

While this study went in depth on training a linear regression model, part of my goal was to explore more aspects of training regression models. Therefore, several factors were left out. Limitations of this study included:

- The linear regression model did not have the data it needed to make precise predictions but had a relatively normal distribution of error, suggesting other external factors existed that were associated with price of gas.
- It is possible that another model may have been more ideal for this situation, since feature relationships in the data were unlikely linear.
- The model relied on the data from *********Gas Buddy********* which was sourced from user input. Such data is subjective and sensitive to many factors. Other factors in price include geopolitical events, economic events, etc., which were not accounted for in this model.
- Training data was not balanced. Several categorical features had high variability and there were large differences of data.
- Real-time prediction data was not large enough. Significantly more data would have been required to increase the probability of offsetting the error in a more positive direction.

# References

## Sources

1. Richter, Felix (2022). “Cars Still Dominate the American Commute”. ******************************Statista: The Statistics Portal******************************. Retrieved 25 March 2023 from [https://www.statista.com/chart/18208/means-of-transportation-used-by-us-commuters/](https://www.statista.com/chart/18208/means-of-transportation-used-by-us-commuters/)
2. Barkingdata (2022). “US Gas Station Pricing Data - GasBuddy Pricing”. ******Kaggle******. Retrieved 25 March 2023 from [https://www.kaggle.com/datasets/polartech/us-gas-station-pricing-data-gasbuddy-pricing](https://www.kaggle.com/datasets/polartech/us-gas-station-pricing-data-gasbuddy-pricing)
3. Worldwide; OPEC; BNN Bloomberg; Intercontinental Exchange; March 2, 2020 to February 27, 2023. “Closing price of Brent, OPEC basket, and WTI crude oil at the beginning of each week from March 2, 2020 to February 27, 2023”. ******************************Statista: The Statistics Portal******************************. Retrieved 25 March 2023 from [https://www.statista.com/statistics/326017/weekly-crude-oil-prices/](https://www.statista.com/statistics/326017/weekly-crude-oil-prices/).

Full code is available on my Github: [https://github.com/hainesdata/gas](https://github.com/hainesdata/gas)