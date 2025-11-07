# Estimation of obesity levels based on eating habits and physical condition 
## Cohort: 7, Team: ML10

## Members

- [Eka Dwipayana]()
- [Olalekan Kadri]()
- [Rameez Rosul]()
- [Shefali Lathwal]()
- [Suni Bek]()
- [Vinushiya Shanmugathasan]()

# Contents
<!-- We will add a table of contents here -->

# Overview
We have access to a dataset of 2111 individuals that records their obesity Level along with 17 attributes related with eating habits, physical condition, and demographics.

## Business Objective
World Health Organization (WHO) is interested in understanding key features associated with obsesity levels in three different locations - Mexico, Peru and Colombia. Using the provied dataset, we want to understand the main factors that contribute to obesity levels. Once the top factors are identified, WHO can make a decision about focusing educational initiatives and allocating financial resources towards the identified factors in order to achieve the most improvement in health outcomes.

Therefore, the key deliverable for us is the list of top 3 factors that are most strongly correlated with obesity levels.

## Dataset details

The data consist of 2111 individuals with 17 attributes recorded for each individual. The list of attributes is as follows:
| Variable Name | Type | Description | Category |
 | --- | --- | --- | --- |
 | Gender | Categorical | - | Demographic |
 | Age | Continuous | - | Demographic |
 | Height | Continuous | - | Other |
 | Weight | Continuous | - | Other |
 | family_history_with_overweight | Categorical | Has a family member suffered or suffers from overweight? | Family History |
 | FAVC | Categorical | Do you eat high caloric food frequently? | Eating Habits |
 | FCVC | Integer | Do you usually eat vegetables in your meals? | Eating Habits |
 | NCP | Continuous | How many main meals do you have daily? | Eating Habits |
 | CAEC | Categorical | Do you eat any food between meals? | Eating Habits |
 | SMOKE | Categorical | Do you smoke? | Lifestyle |
 | CH2O | Continuous | How much water do you drink daily? | Eating Habits |
 | SCC | Categorical | Do you monitor the calories you eat daily? | Eating Habits |
 | FAF | Continuous | How often do you have physical activity? | Lifestyle |
 | TUE | Integer | How much time do you use technological devices such as cell phone, videogames, television, computer and others? | Lifestyle |
 | CALC | Categorical | How often do you drink alcohol? | Lifestyle |
 | MTRANS | Categorical | Which transportation do you usually use?| Lifestyle |
 | NObeyesdad | Categorical | Obesity level | Target |
 
## Potential risks and uncertainty
- Some important factors such as genetic pre-disposition to obesity, presence of diabetes, etc. are missing from the data and will therefore be ignored.
- While the models will tell us the most important features that predict obesity, the relationships are not necessarily causal, which means that improving that factor may not reduce obesity levels.
- BMI (weight/height^2) has been used to define the target categories, and both weight and height have also been included in the data as predictors. From our initial analysis, we see that weight is the most important predictor for each category as expected. Therefore, it's unclear if weight and height should be included as predictors.
- Any insights we draw from these data will be geographically limited to Latin America and not transferable to other geographies.
- Only 23% data are directly collected from people, the rest have been synthetically generated. Synthetic data may not be representative of the real population.
- We have 7 categories in our target variable. Some of these may be hard to distinguish.
- Some of the variables in the data are self-reported and may contain biases inherent in self-reported data. For example, a variable like FAVC (Do you eat high caloric food frequently?) is difficult to self-assess and report accurately.

# Methodology
The overall plan to tackle the project is as follows:

1. **Exploratory Data Analysis (EDA)**: understand each feature, explore relationships between features, understand class imbalances, analyze any missing data, and to generate ideas for feature engineering and data-pre-processing.
2. **Unsupervised analysis**: perform unsupervised clustering of data and analyze the key features contributing to each cluster.
3. **Develop and evaluate predictive ML models**:
    - Create a baseline logistic regression model
    - Evaluate the performance of the model using metrics such as precision, recall, and confusion matrices.
    - Finetune baseline model and try other algorithms such as Random Forest Classifier and XGBoost classifier
    - Compare performance of the more complex models with the baseline model.
4. **Choose the best model**: Compare performance of different models to choose the best one, calculate feature importance of input features
5. Make recommendations and suggestions for the executive team at WHO.

## Git structure

```
├── data
├──── raw
├── experiments
├── models
├── images
├── README.md
└── .gitignore
```
* **Data:** Contains the raw data. 
* **Experiments:** A folder containing ipython notebook for data exploration and experiments.
* **Models:** A folder containing the final trained model
* **Images:** Contain all images used in the README.md file
* **README:** This file!
* **.gitignore:** Files to exclude from this folder (e.g., large data files).

## Technical stack
- scikit-learn library to fit and evaluate supervised and unsupervised ML models.
- pandas and numpy: to load and explore the dataset and basic data manipulations
- matplotlib: to create visualizations

## Task assignment
- Eka Dwipayana: Starter code for exploratory data analysis and models, XGBoost experiments and model (including SHAP analysis)
- Olalekan Kadri: Baseline logistic regression and experiments (including SHAP analysis)
- Rameez Rosul: Unsupervised data analysis 
- Shefali Lathwal: Exploratory data analysis
- Suni Bek: Exploratory data analysis
- Vinushiya Shanmugathasan: Experiment with Random Forest model (including SHAP analysis)

## Exploratory data analysis

## Model Development and Evaluation

### Model 1

#### Model 2

### Model 3

### Final model

# Conclusions and Future Directions

# Team Videos

# References
- [Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) has been sourced from UC Irvine Machine Learning Repository
- [Publication](https://doi.org/10.1016/j.dib.2019.104344) linked to the dataset