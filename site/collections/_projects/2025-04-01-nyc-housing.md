---
date: 2025-04-01 05:20:35 +0300
title: 'Sleeper Neighborhoods in NYC: Data Workflow & Anomaly Detection'
subtitle: Python + Docker + Airflow + Postgres + Pydeck
image: '/images/housing.jpg'
---
Rent in NYC is expensive. I built a system to find neighborhoods with unexpectedly low rent for their respective noise levels, crime rates, and historical rent prices. I built a pipeline that aggregated complaint data from NYPD, construction permit data from NYC, historic rent data from Zillow & StreetEasy, and geospatial transportation noise data from USDOT into each neighborhood. Then, I built an interactive visualization tool that highlights the location of each sleeper neighborhood with its rent, safety, etc.

Disclaimer: all values were modified from their originals for this deployment, so these results may not be accurate to
the original data.

## Interactive Map
The interactive map below uses a cylinder whose height represents a score computed from noise levels, crime rates, and historic
rent prices. Lower scores generally represent a higher quality neighborhood, based on the factors included in the neighborhood. 
Cylindrical columns highlighted in green indicate a "sleeper neighborhood": those with current rent costs lower than costs expected
by statistical modelling. 

<iframe src="https://aspenflow.github.io/sleeper-neighborhoods/" width="100%" height="500px"></iframe>

## Data Processing
### Sources

| Description                              | Source                              |
|------------------------------------------|-------------------------------------|
| Neighborhood coordinates & identifiers   | OpenStreetMap                       | 
| Median asking rent for studio apartments | StreetEasy                          | 
| Complaint incident reports               | NYPD / NYCOpenData                  | 
| Housing construction jobs                | NYC Planning                        | 
| Road, rail, and aviation noise data      | Bureau of Transportation Statistics | 

### Dataset Descriptions
#### Neighborhood Coordinates & Identifiers
Data is retrieved from OpenStreetMap's Nominatim API. This is used to build a key, where each neighborhood name listed by 
StreetEasy references its respective coordinates.

#### Median Asking Rent
Data retrieved from StreetEasy's Data Dashboard. This is a wide dataset containing records of neighborhood-wise rent 
prices for each month since 2010. 

Sample:

| areaName           | Borough   | areaType     | 2010-01 | 2010-02 | 2010-03 | 2010-04 |
|--------------------|-----------|--------------|---------|---------|---------|---------|
| Astoria            | Queens    | neighborhood | 1600    | 1650    | 1620    | 1600    |
| Auburndale         | Queens    | neighborhood |         |         |         |         |
| Bath Beach         | Brooklyn  | neighborhood |         |         |         |         |
| Battery Park City  | Manhattan | neighborhood | 3495    | 3346    | 3268    | 3295    |

#### NYPD Complaint Incidents
Incident-level data retrieved from NYPD City-Wide Crime Statistics endpoint. Each row contains 36 values about an individual complaint,
including coordinates referencing the incident's location. 

#### Housing Construction Jobs
Data retrieved from NYC Planning's DCP Housing Database. The data contains all records since 2010 for building construction and demolition permits, 
including the building's street address, coordinates, number of floors, zoning district, completion year, and permit years.

#### Road, Rail, and Aviation Noise
This geospatial raster data comes from the US Bureau of Transportation Statistics. The data contains measurements of 
aviation, road, and rail noise pollution across the US, as of 2020. 

<div style="text-align: center;">
  <img src="/images/usdot-noise.png" alt="USDOT Noise" width="50%">
</div>

### Preprocessing & Transformation
Pre-processing and transformation was orchestrated in Airflow.

<div style="text-align: center;">
  <img src="/images/housing-pipeline.png" alt="Housing Pipeline" width="50%">
</div>

All aggregate data was assigned to its respective neighborhood based on coordinates. A record is attributed to the neighborhood whose
coordinates are the shortest distance from the record's. Since data transformations were being performed in Postgres, PostGIS
was necessary for computing the nearest neighbor for each coordinate pair. Additionally, spatial indices were added where
necessary for downstream aggregate computation.

To optimize memory usage, TIFF raster tiles were filtered to only the NY tile and cropped to boundaries including NYC. 
The raster needed to be repaired. In addition, it needed a spatial index assigned to noise level, so aggregate noise levels
for a given neighborhood could be computed downstream. 

Rent data was transposed to long format, and missing values were handled during aggregation. 

Neighborhood-wise aggregate data was computed as follows: 
* **Crime:** number of crimes per neighborhood 
* **Noise:** average noise per neighborhood
* **Recent rent:** median across the most recent $n$ available records for each neighborhood, where $n <= 15$. 
* **Overall rent:** median across all available records for each neighborhood
* **Floors:** median number of floors across all construction projects for each neighborhood

Following the completion of the output data format construction, standardization and scoring were performed. 
All residual tables from joins and aggregations were removed at the end of the pipeline.

### Standardization
After the final data structure was formed, additional standardized columns were added to enable more precise weighting and 
clearer analysis downstream. Standardization was computed using:

$$
\text{standard}(x) = \frac{x - \text{median}(x)}{x_{0.75}-x_{0.25}}
$$

where `x` is the numeric value, `x_0.75` and `x_0.25` are the 75th and 25th percentiles of `x`, respectively.

### Scoring
A score was defined to enable simpler neighborhood comparison. Score is defined as:

$$
\text{score} = w_1\cdot\text{crime} + w_2\cdot\text{noise}  + w_3\cdot\frac{(\text{rent}_\text{overall} - \text{rent}_\text{recent})}{\text{rent}_\text{overall}} + w_4\cdot\text{age} + w_5\cdot \text{floors}
$$

For this deployment, all weights were left as 1, but can be altered in the pipeline as needed. That is, 

$$
w_1=\dots=w_5=1
$$

Furthermore, scores were normalized between 0 and 1:

$$
\text{norm}(\text{score})=\frac{\text{score} - \text{min}(\text{score})}{\text{max}(\text{score})- \text{min}(\text{score})}
$$

### Distributions
Below shows kernel densities of each standardized variable. Visualizing the distribution of each standardized variable not only enables more intuitive score weighting, but also 
provides explanatory insight into the overall characteristics of neighborhoods in NYC. 

<div style="text-align: center;">
  <img src="/images/housing-dists.png" alt="KDEs" width="75%">
</div>

Differences in variables become apparent when looking at the distribution tails, widths, kurtosis, and skew. For example, it is 
clear on visual inspection that the height of building projects in NYC might vary more across neighborhoods than noise 
level and crime. In this case, it may be appropriate to assign more weight to floors, but this should only be done after 
testing quantitatively for differences between distributions.

## Anomaly Detection
To discover which neighborhoods are sleepers (unexpectedly low rent), a regression-based anomaly detection 
model was constructed and assessed for validity.

### Regression Model
#### Construction
A regression model was constructed to predict the recent median rent given the number of crimes, average noise levels, and average 
building age in a given neighborhood:

$$
\text{rent}_\text{pred} = \beta_0+\beta_1\cdot \text{crimes}+\beta_2\cdot\text{noise} + \beta_3\cdot\text{age}
$$

The model was fitted using `statsmodels.api`, and residuals were computed, producing the model with coefficients 

| Variable    | Coef      | Std Err   | t       | P>\|t\| |
|-------------|-----------|-----------|---------|---------|
| const       | 3811.4810 | 2343.985  | 1.626   | 0.108   |
| num_crimes  | -0.0083   | 0.027     | -0.311  | 0.757   |
| avg_noise   | -52.0183  | 41.119    | -1.265  | 0.209   |
| avg_age     | 184.3320  | 101.876   | 1.809   | 0.074   |

#### Assessment
To ensure the regression model met theoretical assumptions and consequent valid conclusions, it was assessed using a series of tests. 
Using a model residuals test, it was confirmed no non-linearity or heterscedasticity was present, either of which would
violate linear regression assumptions. 

<div style="text-align: center;">
  <img src="/images/housing-resid-fit-plot.png" alt="KDEs" width="75%">
</div>

Additionally, there were no collinearity issues found between predictors, as seen with variance inflation factors:

<div style="text-align: center;">
  <img src="/images/housing-pred-cormat.png" alt="KDEs" width="75%">
</div>

### Anomaly Classification
A rent price is considered an anomaly if the residual rent cost $\epsilon$ is less than a threshold $\lambda$ defined by 
1.25 standard deviations $\sigma$ below the residual mean $\bar{\epsilon}$:

$$
\lambda = \bar{\epsilon} - 1.25 \cdot \sigma_{\epsilon} \quad \epsilon \lt \lambda
$$

Then, each anomaly is highlighted in green on the plot. 

## Conclusion
It is already well known that rent prices are dependent on several factors, but those are proprietary. In a housing market
like New York City's, one of the most expensive cities to live in within the US, saving on rent without compromising 
quality of life is important. The goal of this project was solely for me to find neighborhoods that were "hidden gems".
With that goal in mind, I turned it into an opportunity to utilize and further develop my abilities in statistical 
analysis and data engineering, and this project did just that.