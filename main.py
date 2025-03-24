#%% md
# # Introduction: A Life-Saving Question
# 
# Imagine a 55-year-old man named David. He leads a relatively healthy life but occasionally smokes and has a family history of heart disease. One day, he starts feeling unusual chest pains but dismisses them as stress. Could a data-driven approach have warned him earlier that he was at risk?
# 
# **This leads us to the central question of our project:**
# What if we could spot the warning signs early, using only lifestyle and health data?
# 
# # About Us and Our Plan for This Project
# 
# As computer science students with a deep interest in data science, we,  Hila Giladi (312557606) and Kfir Shuster (315695122), have chosen to focus on heart disease prediction and analysis. We are committed to leveraging data science to better understand risk factors and potentially help improve early detection and prevention strategies.
# 
# Through careful analysis of various health metrics and lifestyle factors, we aim to contribute to the broader understanding of heart disease risk factors and their complex interactions.
# 
# # What’s the Risk We’re Facing?
# To understand how cases like David’s might be predicted earlier, we first need to explore the broader problem heart disease presents.
# 
# Our analysis of the data reveals a complex picture of the challenges in understanding heart disease factors. While approximately 8.6% of the population in our dataset suffers from heart disease, our analysis uncovers an intricate web of relationships between various risk factors, ranging from demographic factors such as age and gender, through lifestyle habits like smoking and physical activity, to existing medical conditions such as diabetes and stroke. This complexity emphasizes the need for a comprehensive and multidimensional approach to understanding heart disease risk.
# 
# # Why Solving It Matters
# Heart disease remains one of the leading causes of mortality worldwide, making it a critical public health concern. Understanding and predicting heart disease risk isn't just about extending life expectancy - it's about improving quality of life and reducing the enormous burden on healthcare systems and families.
# 
# # How We're Going to Do It
# Heart disease is a complex health issue influenced by numerous factors, and we aim to address this complexity through a comprehensive analysis approach. This project will develop a predictive model for heart disease risk using a dataset that captures various physiological, behavioral, and demographic factors, including BMI, physical health metrics, lifestyle factors, and various health conditions.
# 
# By leveraging machine learning techniques to predict heart disease risk, we aim to:
# - Identify key factors that influence heart disease risk and their relative importance
# - Understand the complex relationships between lifestyle choices, existing health conditions, and heart disease
# - Develop models that could assist in early risk identification and intervention
# 
# Through this analysis, we hope to contribute to the broader understanding of heart disease risk factors and potentially develop tools that could help individuals and healthcare providers in monitoring and improving heart health outcomes. Our approach combines statistical analysis, machine learning, and data visualization to uncover patterns and relationships that might not be immediately apparent, potentially leading to more effective prevention strategies and early intervention opportunities.
# 
# The dataset used in this project is based on the CDC’s 2020 Behavioral Risk Factor Surveillance System (BRFSS), a large-scale health survey conducted annually across the United States.
# 
# Source: [Heart Disease Prediction Dataset on Kaggle](https://www.kaggle.com/code/andls555/heart-disease-prediction/notebook)
# 
# 
# To build our model, we analyzed 319,795 records covering a range of health, lifestyle, and demographic factors. This dataset allows us to explore the intricate relationships between lifestyle choices, medical history, and heart disease risk.
# 
# 
# # Exploring the Dataset
# 
# We utilized a comprehensive dataset containing 319,795 records with 18 different variables, covering a wide range of metrics:
# - Physiological measures: BMI, physical health
# - Lifestyle habits: smoking, alcohol consumption, physical activity, sleep hours
# - Health conditions: diabetes, stroke, asthma, kidney disease
# - Demographic characteristics: age, sex, race
# 
# This dataset was selected for its extensive scope and rich variety of variables, allowing us to examine the complex relationships between various factors and heart disease. The data provides a solid foundation for in-depth analysis and the development of predictive models that may help improve our understanding of heart disease risk factors and enhance early detection capabilities.
# 
# # Data Analysis
# 
# Now that we understand the challenge and the data we’re working with, it's time to start looking for answers.
# 
# We begin by exploring the dataset itself — transforming it into a structured format and getting familiar with the variables. This will help us uncover patterns, correlations, and surprising insights that may help predict heart disease risk.
# 
# Let’s load the data and take our first look.
# 
#%%
import pandas as pd

df = pd.read_csv('heart_2020_cleaned.csv')
display(df)
#%% md
# With the data loaded and prepared, we can now begin exploring it more deeply.
# 
# This stage is where the story starts to unfold — we’ll look at how heart disease appears in the population, and how different lifestyle choices, health conditions, and demographics might be linked to it.
# 
# Our goal is to identify which variables truly help distinguish between individuals with and without heart disease — and which ones may not be as informative as they seem.
# 
#%% md
# Before diving into the data and modeling, it's helpful to understand how this project is structured.
# We've organized the work into clearly defined sections, each building on the previous one — from framing the problem to interpreting the results.
# 
# Here’s an overview of the main components of our analysis:
# 
# # Project Structure Overview
# 
# This project is organized into the following main sections:
# 
# 1. **Data Preparation** – Cleaning, encoding, restructuring, and balancing the dataset
# 2. **Exploratory Data Analysis (EDA)** – Identifying trends, patterns, and correlations
# 3. **Predictive Modeling** – Training and evaluating machine learning models
# 4. **Model Comparison and Interpretation** – Comparing performance and using SHAP to explain results
# 5. **Final Reflections and Future Directions** – Key takeaways and ideas for continued work
# 
# 
#%% md
# # 1. Data Preparation
# 
# Before we could explore patterns in the data, we needed to clean and structure it for analysis. This included converting binary variables, regrouping age categories, and filtering out unrealistic values to ensure our dataset is ready for meaningful analysis.
# 
# <details>
# <summary>🔧 Preprocessing Details (click to expand)</summary>
# 
# 1. **Binary conversion (Yes/No → 1/0)**
#    - Applied to variables such as `HeartDisease`, `Smoking`, `AlcoholDrinking`, `Stroke`, etc.
#    - Special handling for `KidneyDisease` to ensure correct integer type.
# 
# 2. **Age re-grouping into 10-year categories**
#    - Original narrow categories (e.g., `"18–24"`, `"25–29"`) were merged into broader decade ranges.
#    - Final categories: `"18–29"`, `"30–39"`, ..., `"80+"`.
# 
# 3. **Sleep duration filtering**
#    - Removed unrealistic values by keeping only entries with 1 to 16 hours of sleep per day.
#    - This eliminated extreme outliers likely caused by data entry errors or unusual reporting.
# 
# </details>
# 
# 
#%%
pd.set_option('future.no_silent_downcasting', True)
binary_columns = [
    'HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
    'DiffWalking', 'PhysicalActivity', 'Asthma',
    'KidneyDisease', 'SkinCancer'
]
# Map Yes/No to 1/0
for column in binary_columns:
    if column == 'KidneyDisease':
        df['KidneyDisease'] = df['KidneyDisease'].replace({'Yes': 1, 'No': 0}).astype(int)

    else:
        df[column] = df[column].map({'Yes': 1, 'No': 0})


def get_age_group_10_years(age_category):
    """
    Takes the original age category and returns a new 10-year category

    Parameters:
    age_category (str): Original age category (e.g. "18-24", "25-29" etc.)

    Returns:
    str: New age category ("18-29", "30-39" etc.)
    """
    if age_category in ['18-24', '25-29']:
        return '18-29'
    elif age_category in ['30-34', '35-39']:
        return '30-39'
    elif age_category in ['40-44', '45-49']:
        return '40-49'
    elif age_category in ['50-54', '55-59']:
        return '50-59'
    elif age_category in ['60-64', '65-69']:
        return '60-69'
    elif age_category in ['70-74', '75-79']:
        return '70-79'
    elif age_category == '80 or older':
        return '80+'
    else:
        return 'Unknown'


# Add new age group column
df['AgeCategory'] = df['AgeCategory'].apply(get_age_group_10_years)

display(df)
#%%
df = df[(df['SleepTime'] >= 1) & (df['SleepTime'] <= 16)]
#%% md
# # Summary of Data Preparation
# 
# With the noise removed and key variables organized, our data is finally ready to speak.
# This clean foundation gives us a reliable base to begin uncovering the hidden patterns behind heart disease.
#%% md
# # 2. Exploratory Data Analysis (EDA)
# 
# Now that our data is clean and well-structured, we’re ready to start uncovering patterns.
# 
# In this stage, we begin asking the questions that brought us here in the first place:
# Who is most at risk of heart disease — and why?
# 
# We’ll explore how heart disease is distributed in the population and how it relates to factors such as age, sex, smoking, physical activity, and medical history. Through visualizations and statistical tests, we aim to find which variables truly matter, and which might be less informative than expected.
# 
# To guide our exploration, we focused on the following key questions:
# - Which factors have the strongest connection to heart disease?
# - Are commonly cited risk factors like BMI and sleep duration actually meaningful predictors?
# - How do combinations of conditions (e.g., diabetes and stroke) affect risk levels?
# 
# These questions shaped the structure of our analysis and helped us focus on the variables with the most impact.
# 
# <details>
# <summary>🗂️ EDA Structure Overview (click to expand)</summary>
# 
# To guide our analysis, we organized the EDA into the following sections:
# 
# 1. **Overall Distribution of Heart Disease** – Class balance of heart disease cases.
# 2. **Heart Disease and Demographics** – Patterns across race, sex, and age.
# 3. **Statistical Analysis of Health Variables** – Correlations, multicollinearity, and redundancy checks.
# 4. **Heart Disease and Lifestyle Habits** – Smoking, alcohol, physical activity, general health perception.
# 5. **Heart Disease and Pre-Existing Health Conditions** – Diabetes, stroke, kidney disease, asthma.
# 6. **Heart Disease and Physiological Measurements** – BMI, sleep time, and related analyses.
# 7. **Heart Disease and Interacting Health Factors** – Combined health effects and subgroups.
# 8. **PCA (Principal Component Analysis)** – Dimensionality reduction of key health variables.
# 
# </details>
# 
# 
# 
#%% md
# ## Overall Distribution of Heart Disease
# 
# Before diving into specific risk factors, it's important to understand the overall distribution of heart disease in our dataset.
# How common is it? Are we dealing with a balanced population — or is one group significantly more represented than the other?
# 
# 
# ### Distribution of heart disease cases in our population
# We begin by looking at the big picture: how is heart disease distributed across the dataset?
# This simple histogram helps us understand the overall class balance — a key factor in choosing the right modeling strategy later on.
# 
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Create the figure
plt.figure(figsize=(13, 6))

# Create histogram/count plot with updated syntax
sns.countplot(data=df, x='HeartDisease', hue='HeartDisease', palette='Set2', legend=False)

# Calculate percentages
total = len(df['HeartDisease'])
percentages = df['HeartDisease'].value_counts(normalize=True) * 100

# Add percentage labels on top of each bar
for i, percentage in enumerate(percentages):
    plt.text(i, df['HeartDisease'].value_counts()[i],
             f'{percentage:.1f}%',
             horizontalalignment='center',
             verticalalignment='bottom')

# Customize the plot
plt.title('Distribution of Heart Disease Cases', pad=20)
plt.xlabel('Heart Disease')
plt.ylabel('Count')

# Show plot
plt.show()
#%% md
# The graph reveals a significant class imbalance: 91.4% of individuals in the dataset do not have heart disease, while only 8.6% do.
# This imbalance is important to keep in mind, as it will impact both how we analyze the data and how we train predictive models later in the project.
# 
#%% md
# ## Heart Disease and Demographics
# In this section, we examine how heart disease prevalence varies across key demographic characteristics — including **race**, **sex**, and **age**.
# Understanding these basic population-level patterns helps provide context for the deeper analyses that follow.
#%% md
# ### Distribution of Heart Disease by Race
# Next, we examine the distribution of heart disease across racial groups.
# This helps us check whether certain racial demographics show higher prevalence — or if race has limited predictive value in our dataset.
#%%
sns.countplot(data=df, x='Race', hue='HeartDisease', palette='YlOrBr')
plt.xlabel('Race')
plt.ylabel('Frequency')
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.show()
#%% md
# 
# The graph shows that White individuals make up the majority of the dataset, while other racial groups are represented in smaller numbers.
# However, the proportion of heart disease within each group is relatively similar, suggesting that race may not be a strong predictor in this context.
# 
# To avoid introducing potential bias and to simplify our models, we chose to remove the `Race` column from our dataset.
# This decision is based on the relatively uniform distribution observed across racial groups, and our focus on features with stronger variation in relation to heart disease.
#%%
# Remove Race column from dataset as it doesn't provide significant predictive value
df = df.drop('Race', axis=1)
print(f"Race column removed. Dataset now has {df.shape[1]} columns.")
display(df)
#%% md
# ### Distribution of Heart Disease by Sex
# We now turn to sex as a potential risk factor.
# Are there notable differences between males and females in terms of heart disease prevalence — or is the pattern relatively balanced?
#%%
sns.countplot(data=df, x='Sex', hue='HeartDisease', palette='YlOrBr')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()
#%% md
# The dataset contains slightly more female than male participants.
# Both sexes show a similar distribution of heart disease cases, with no major disparity between them.
# 
# This suggests that while sex may play a role in heart disease risk, it likely interacts with other factors (like age or health conditions) rather than acting as a strong standalone predictor.
# We'll revisit this variable later when analyzing correlations and model feature importance.
# 
#%% md
# ### Distribution of Heart Disease by Age Category
# Age is one of the most well-known risk factors for heart disease — but how clearly does that show up in our dataset?
# 
# Let’s look at how heart disease is distributed across different age groups to see if the risk indeed increases with age, and whether certain decades stand out more than others.
# 
#%%
# create ordered list of age categories
age_order = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

# create the countplot with ordered categories
sns.countplot(data=df, x='AgeCategory', hue='HeartDisease',
              palette='YlOrBr', order=age_order)
plt.xlabel('Age Category')
plt.ylabel('Frequency')
plt.title('Heart Disease Distribution by Age Category')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()
#%% md
# As expected, the prevalence of heart disease rises with age.
# The 60–69 age group shows the highest frequency, and the upward trend continues into the 70s and 80+ groups.
# 
# This confirms the strong connection between age and heart disease and supports the importance of including age as a key variable in our predictive models.
# 
#%% md
# ## Statistical Analysis of Health Variables
# Next, we shift our focus to statistical relationships between variables.
# Using correlation analysis and multicollinearity checks, we explore how features relate to each other and to heart disease — and assess whether any of them provide redundant or overlapping information.
# 
#%% md
# ### Correlation Matrix of Health Variables
# Before diving into combinations of variables, we want to understand how each individual feature relates to heart disease.
# 
# To do this, we create a correlation matrix that shows the strength and direction of the relationship between heart disease and other variables in the dataset.
# This helps us spot the features that are most likely to be useful in prediction — and those that might be less relevant.
#%% md
# First we convert the non-numeric columns to numeric columns for the correlation matrix.
# 
#%%
df_copy = df.copy()

# Sex mapping
df_copy['Sex'] = df_copy['Sex'].replace({'Male': 1, 'Female': 0})

# Age mapping
age_map = {
    '18-29': 1,
    '30-39': 2,
    '40-49': 3,
    '50-59': 4,
    '60-69': 5,
    '70-79': 6,
    '80+': 7
}
df_copy['AgeCategory'] = df_copy['AgeCategory'].replace(age_map)

# Diabetic mapping
diabetic_mapping = {
    'No': 0,
    'No, borderline diabetes': 1,
    'Yes (during pregnancy)': 2,
    'Yes': 3
}
df_copy['Diabetic'] = df_copy['Diabetic'].replace(diabetic_mapping)

# General Health mapping
genhealth_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

df_copy['GenHealth'] = df_copy['GenHealth'].replace(genhealth_mapping)
df_copy['Sex'] = df_copy['Sex'].replace({'Male': 1, 'Female': 0}).astype(int)
display(df_copy)
#%%
# Create correlation matrix
correlation_matrix = df_copy.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,  # show correlation values
            cmap='coolwarm',  # color scheme
            center=0,  # center the colormap at 0
            fmt='.2f',  # show 2 decimal places
            square=True,  # make the plot square-shaped
            vmin=-1, vmax=1)  # set the range of values

plt.title('Correlation Matrix of Variables')
plt.tight_layout()
plt.show()
#%% md
# The correlation matrix confirms several expected patterns:
# 
# - **Age**, **stroke**, **difficulty walking**, and **diabetes** all show positive correlations with heart disease.
# - On the other hand, **physical activity** and **general health** are negatively correlated, meaning healthier and more active individuals are less likely to have heart disease.
# 
# Interestingly, some variables that are often considered risk factors — like **BMI** and **sleep time** — show surprisingly weak correlations with heart disease.
# This suggests that they might not be strong predictors on their own, and we’ll take this into account when selecting features for modeling.
# 
# We also notice a moderately strong negative correlation between **physical health** and **general health**, which raises the question:
# Do these two variables provide overlapping information, or do they each capture something unique?
# To answer that, we turn to multicollinearity analysis using the Variance Inflation Factor (VIF).
# 
#%% md
# ### Investigating Multicollinearity Using VIF Analysis
# While the correlation matrix shows how each variable relates to heart disease, it doesn’t tell us whether two variables are providing redundant information.
# 
# **Multicollinearity** — when two or more features contain overlapping information — can distort model interpretation and reduce performance.
# 
# In particular, we noticed a moderate negative correlation between **general health** and **physical health**.
# To check whether one of them could be dropped without losing valuable information, we calculate the Variance Inflation Factor (VIF), which helps us detect multicollinearity between variables.
# 
# 
#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np

# First, let's create a new DataFrame with just the columns we need
# This helps us avoid modifying the original DataFrame
df_vif = pd.DataFrame()

# Convert GenHealth to numeric values
genhealth_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}
df_vif['GenHealth_numeric'] = df['GenHealth'].replace(genhealth_mapping).astype(float)

# Ensure PhysicalHealth is numeric and handle any non-numeric values
df_vif['PhysicalHealth'] = pd.to_numeric(df['PhysicalHealth'], errors='coerce')

# Drop any rows with NaN values to ensure clean data for VIF calculation
df_vif = df_vif.dropna()

# Add a constant column for VIF calculation
X = add_constant(df_vif)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print the VIF data
print("VIF Data:")
print(vif_data)

# If you want to calculate VIF after dropping a feature
X_dropped = add_constant(df_vif[['PhysicalHealth']])
vif_data_dropped = pd.DataFrame()
vif_data_dropped["Feature"] = X_dropped.columns
vif_data_dropped["VIF"] = [variance_inflation_factor(X_dropped.values, i) for i in range(X_dropped.shape[1])]

print("\nVIF Data after dropping GenHealth_numeric:")
print(vif_data_dropped)
#%% md
# The results show that both **general health** and **physical health** have low VIF values (~1.3), which indicates very low multicollinearity.
# 
# This means that although they are somewhat related, each provides distinct information — and both are worth keeping in the analysis.
# Keeping both allows our models to capture both subjective and objective perspectives on a person’s health status.
# 
#%% md
# ### Do Physical Activity and Difficulty Walking Overlap?
#%% md
# Next, we want to understand whether **physical activity** and **difficulty walking** are capturing the same behavior — or providing different signals.
# 
# At first glance, both variables seem to relate to mobility.
# But are they statistically distinct? To test this, we compare their distributions using a T-test.
# 
#%%
from scipy.stats import ttest_ind

t_stat, p_value_ttest = ttest_ind(df['PhysicalActivity'], df['DiffWalking'])

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value of T-test: {p_value_ttest:.3f}")
#%% md
# The T-test reveals a statistically significant difference between the two variables (p < 0.001), meaning they are not interchangeable.
# 
# Although both relate to physical function, they likely represent **different aspects**:
# - Physical activity captures voluntary behavior
# - Difficulty walking reflects physical limitation
# 
# We’ll keep both features in our analysis, since each may provide unique predictive value.
# 
#%% md
# ## Heart Disease Distribution by Lifestyle Habits
# Lifestyle choices play a major role in cardiovascular health.
# Behaviors such as smoking, alcohol consumption, and physical activity can influence both the development and prevention of heart disease.
# 
# In this section, we explore how combinations of these lifestyle habits relate to heart disease prevalence in our dataset.
# We’ll examine whether certain behaviors are more strongly associated with risk, and how subjective health perception interacts with activity levels.
# 
#%% md
# ### Heart Disease by Smoking and Alcohol Habits
# We begin our analysis of lifestyle habits by examining the combination of smoking and alcohol consumption.
# While both behaviors are commonly linked to heart disease, their combined effect — and how it plays out in real-world data — is worth a closer look.
# 
# To explore this, we categorized individuals into four groups based on whether they smoke and/or drink alcohol, and compared the prevalence of heart disease in each group.
#%%
# read the data again to avoid changes that affect the next graph
df_smoke_drink = pd.read_csv('heart_2020_cleaned.csv')

# create a new column combining smoking and alcohol status
df_smoke_drink['Habits'] = (df_smoke_drink['Smoking'].map({'Yes': 'Smoker', 'No': 'Non-Smoker'}) + ', ' +
                            df_smoke_drink['AlcoholDrinking'].map({'Yes': 'Drinker', 'No': 'Non-Drinker'}))

# calculate heart disease percentage for each combination
heart_disease_stats = df_smoke_drink.groupby('Habits')['HeartDisease'].apply(
    lambda x: (x == 'Yes').mean() * 100).reset_index()

# sort values for better visualization
heart_disease_stats = heart_disease_stats.sort_values('HeartDisease')

# create a bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Habits',
                 y='HeartDisease',
                 data=heart_disease_stats,
                 color='skyblue')

# add percentage labels on top of each bar
for i, v in enumerate(heart_disease_stats['HeartDisease']):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.title('Percentage of Heart Disease by Smoking and Alcohol Habits')
plt.xlabel('Lifestyle Habits')

plt.ylabel('Percentage of People with Heart Disease (%)')

plt.tight_layout()
plt.show()
#%% md
# The results show a clear difference between the groups:
# - The highest heart disease rate (12.8%) is found among individuals who smoke but do not drink.
# - The lowest rate (3.0%) appears in those who drink but do not smoke.
# 
# This suggests that smoking is a stronger risk factor than moderate alcohol consumption in this dataset.
# It also highlights that lifestyle factors don’t act in isolation — their combinations can reveal more than each variable on its own.
# 
#%% md
# ### Heart Disease by Physical Activity and General Health
# Next, we explore the relationship between physical activity and general health — two variables that are often linked to heart disease risk.
# 
# People who stay physically active tend to report better overall health, but does this translate into lower heart disease prevalence?
# We created a combined feature that looks at all combinations of activity level and self-reported general health, and measured how heart disease is distributed across those groups.
# 
# 
#%%
# read the data again to avoid changes that affect the next graph
df_physic_gen = pd.read_csv('heart_2020_cleaned.csv')

# create a new column combining physical activity and general health status
df_physic_gen['Health_Status'] = (df_physic_gen['PhysicalActivity'].map({'Yes': 'Active', 'No': 'Inactive'}) + ', ' +
                                  df_physic_gen['GenHealth'])

# calculate heart disease percentage for each combination
heart_disease_stats = df_physic_gen.groupby('Health_Status')['HeartDisease'].apply(
    lambda x: (x == 'Yes').mean() * 100).reset_index()

# sort values for better visualization
heart_disease_stats = heart_disease_stats.sort_values('HeartDisease')

# create a bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Health_Status',
                 y='HeartDisease',
                 data=heart_disease_stats,
                 color='skyblue')

# add percentage labels on top of each bar
for i, v in enumerate(heart_disease_stats['HeartDisease']):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.title('Percentage of Heart Disease by Physical Activity and General Health Status')
plt.xlabel('Health Status')
plt.ylabel('Percentage of People with Heart Disease (%)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
#%% md
# The results are striking:
# - Individuals who are physically active and report excellent health have the lowest heart disease rate (2.1%).
# - In contrast, inactive individuals with poor health show a dramatically higher rate (35.5%).
# 
# This strong gradient emphasizes how both lifestyle behavior and self-perception of health interact in predicting heart disease risk.
# It also suggests that subjective health assessments — often overlooked — may contain valuable predictive information.
# 
#%% md
# ### Distribution of Heart Disease by Physical and Mental Health Days
# As part of our exploration of lifestyle and well-being, we also consider how individuals experience their day-to-day health.
# 
# The dataset includes two variables — **PhysicalHealth** and **MentalHealth** — which indicate the number of days in the past month a person felt physically or mentally unwell.
# We explore whether a higher number of "unhealthy days" corresponds with greater heart disease risk.
# 
#%%
# read the data again to avoid changes that affect the next graph
df_physic_mental = pd.read_csv('heart_2020_cleaned.csv')


# create categories for Physical and Mental Health
def categorize_health(value):
    if value == 0:
        return 'Perfect (0 days)'
    elif value <= 5:
        return '1-5 days'
    elif value <= 15:
        return '6-15 days'
    else:
        return 'Over 15 days'


# create new columns with categorized health values
df_physic_mental['PhysicalHealth_Cat'] = df_physic_mental['PhysicalHealth'].apply(categorize_health)
df_physic_mental['MentalHealth_Cat'] = df_physic_mental['MentalHealth'].apply(categorize_health)

# create a new column combining both health categories
df_physic_mental['Health_Status'] = 'Physical: ' + df_physic_mental['PhysicalHealth_Cat'] + ', Mental: ' + \
                                    df_physic_mental['MentalHealth_Cat']

# calculate heart disease percentage for each combination
heart_disease_stats = df_physic_mental.groupby('Health_Status')['HeartDisease'].apply(
    lambda x: (x == 'Yes').mean() * 100).reset_index()

# sort values for better visualization
heart_disease_stats = heart_disease_stats.sort_values('HeartDisease')

# create a bar plot
plt.figure(figsize=(15, 8))
ax = sns.barplot(x='Health_Status',
                 y='HeartDisease',
                 data=heart_disease_stats,
                 color='skyblue')

# add percentage labels on top of each bar
for i, v in enumerate(heart_disease_stats['HeartDisease']):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.title('Percentage of Heart Disease by Physical and Mental Health Status')
plt.xlabel('Health Status')
plt.ylabel('Percentage of People with Heart Disease (%)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
#%% md
# The graph shows that heart disease becomes more common as the number of physically unhealthy days increases.
# People who reported many days of poor physical health in the past month were significantly more likely to have heart disease.
# 
# In contrast, the connection between mental health and heart disease is weaker, with only a slight increase in prevalence among those with more mentally unhealthy days.
# 
# This suggests that frequent physical discomfort may reflect underlying issues related to heart disease,
# while mental health — at least in this dataset — appears to have a smaller direct impact.
# 
#%% md
# ## Heart Disease and Pre-Existing Health Conditions
# Chronic health conditions are known to be major contributors to heart disease risk.
# 
# In this section, we examine how specific pre-existing medical issues — such as stroke, diabetes, kidney disease, and others — relate to the presence of heart disease in our dataset.
# 
# By looking at combinations and interactions between these conditions, we aim to uncover which ones have the most substantial impact and how they may compound each other.
# 
#%% md
# ### Heart Disease by Diabetes and Stroke Status
# We begin with two of the most well-established cardiovascular risk factors: **stroke** and **diabetes**.
# 
# To explore their combined effect, we created a feature that represents all four possible combinations of stroke and diabetes status, then calculated the prevalence of heart disease in each group.
#%%
# read the data again to avoid changes that affect the next graph
df_diabetic_stroke = pd.read_csv('heart_2020_cleaned.csv')

# create a combination of health conditions
df_diabetic_stroke['Health_Conditions'] = (
        df_diabetic_stroke['Diabetic'].map({'Yes': 'Diabetic', 'No': 'Non-Diabetic'}) + ', ' +
        df_diabetic_stroke['Stroke'].map({'Yes': 'Stroke', 'No': 'No Stroke'}))

# calculate heart disease percentage for each combination
heart_disease_stats = df_diabetic_stroke.groupby('Health_Conditions')['HeartDisease'].apply(
    lambda x: (x == 'Yes').mean() * 100).reset_index()

# sort values for better visualization
heart_disease_stats = heart_disease_stats.sort_values('HeartDisease')

# create a bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Health_Conditions',
                 y='HeartDisease',
                 data=heart_disease_stats,
                 color='skyblue')

# add percentage labels on top of each bar
for i, v in enumerate(heart_disease_stats['HeartDisease']):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.title('Percentage of Heart Disease by Diabetes and Stroke Status')
plt.xlabel('Health Conditions')
plt.ylabel('Percentage of People with Heart Disease (%)')

plt.tight_layout()
plt.show()
#%% md
# The results are clear and concerning:
# - Individuals with both stroke and diabetes have a heart disease prevalence of 48.1% — nearly half of that group.
# - In contrast, those with neither condition show a rate of just 5.8%.
# 
# This suggests a **strong compounding effect** between these two chronic conditions, underlining the importance of early monitoring and intervention for people living with both.
# 
#%% md
# ## Heart Disease and Physiological Measurements
# Physiological metrics such as body mass index (BMI), sleep duration, and kidney function provide measurable indicators of a person's physical state.
# These factors are commonly considered in medical assessments, but how well do they actually correlate with heart disease?
# 
# In this section, we examine the relationship between these physiological measurements and heart disease status — using visualizations, statistical tests, and pairwise comparisons to explore their potential predictive power.
# 
#%% md
# ### BMI Distribution by Heart Disease Status
# 
# We start by looking at **BMI**, a common health indicator often associated with cardiovascular risk.
# 
# We compare the distribution of BMI values between individuals with and without heart disease to identify potential differences.
# 
#%%
# create the density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='BMI', hue='HeartDisease', fill=True, common_norm=False)

plt.title('BMI Distribution by Heart Disease Status')
plt.xlabel('BMI')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
#%% md
# The distributions show that BMI tends to be slightly higher on average among individuals with heart disease.
# However, the difference is not dramatic, and the overall spread of values is fairly similar.
# 
# This suggests that while BMI may have some predictive value, it may not be a strong standalone indicator in this dataset.
# 
#%% md
# #### ANOVA Test: BMI Differences by Heart Disease Status
# To determine whether the difference in BMI between the two groups is statistically significant, we conducted a one-way ANOVA test.
# 
# This allows us to test if the mean BMI differs meaningfully between individuals with and without heart disease.
# 
# 
#%%
from scipy.stats import f_oneway

df_balanced = pd.read_csv('heart_2020_cleaned.csv')
df_balanced = df_balanced.dropna(subset=["BMI"])

df_filtered = df_balanced.dropna(subset=["BMI"])  # Remove NaNs from BMI
df_filtered["HeartDisease"] = df_filtered["HeartDisease"].map({"No": 0, "Yes": 1})
bmi_no_heart_disease = df_filtered[df_filtered["HeartDisease"] == 0]["BMI"]
bmi_with_heart_disease = df_filtered[df_filtered["HeartDisease"] == 1]["BMI"]


# Perform ANOVA test
stat, p_value = f_oneway(bmi_no_heart_disease, bmi_with_heart_disease)

# Print results
print(f"ANOVA Statistic: {stat:.4f}, p-value: {p_value:.4f}")
#%% md
# The ANOVA test reveals a highly statistically significant difference in BMI between individuals with and without heart disease (F = 860.5, p < 0.001).
# This confirms that the average BMI of people with heart disease is different from those without, and that this difference is unlikely to be due to chance.
# 
#%% md
# ### Kidney Disease Distribution by Heart Disease Status
# 
# Next, we explore **kidney disease**, another chronic condition often linked to cardiovascular health.
# 
# We compare the proportion of individuals with and without kidney disease across the heart disease groups to see if there’s a notable association.
# 
#%%
sns.kdeplot(data=df, x='KidneyDisease', hue='HeartDisease', fill=True, common_norm=False)

plt.title('KidneyDisease Distribution by Heart Disease Status')
plt.xlabel('KidneyDisease')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
#%% md
# The data shows that individuals with kidney disease are more likely to have heart disease compared to those without kidney issues.
# This supports existing medical knowledge that kidney function and heart health are closely connected, and that impaired kidney function can be a significant risk factor for cardiovascular disease.
#%% md
# ### Sleep Time Distribution by Heart Disease Status
# 
# Sleep is often considered a crucial component of overall health — but how strongly is it associated with heart disease?
# 
# In this section, we examine the distribution of sleep duration (in hours) among individuals with and without heart disease to see if sleep time differs between the two groups.
# 
#%%
sns.kdeplot(data=df, x='SleepTime', hue='HeartDisease', fill=True, common_norm=False)

plt.title('Sleep Time Distribution by Heart Disease Status')
plt.xlabel('Hours of Sleep')
plt.ylabel('Density')

# Add vertical line for recommended sleep (8 hours)
plt.axvline(x=8, color='red', linestyle='--', alpha=0.5, label='Recommended Sleep (8h)')
plt.legend(title='Heart Disease')

plt.tight_layout()
plt.show()
#%% md
# The distributions of sleep time between the two groups appear very similar.
# There is no obvious difference in the number of sleep hours reported by individuals with and without heart disease.
# 
# This suggests that **sleep duration alone** may not be a strong indicator of heart disease risk — at least within the range of values retained in our cleaned dataset.
# 
#%% md
# ### BMI vs Sleep Time by Heart Disease Status
# 
# To further explore the potential relationship between BMI and sleep time, we plotted the two variables together, segmented by heart disease status.
# 
# This allows us to observe whether certain combinations of BMI and sleep duration are more common among individuals with or without heart disease.
# 
# 
#%%
# separate plots by Heart Disease status for BMI vs Sleep Time
plt.figure(figsize=(15, 6))
for i, condition in enumerate(df['HeartDisease'].unique()):
    plt.subplot(1, 2, i + 1)
    subset = df[df['HeartDisease'] == condition]
    sns.kdeplot(
        data=subset, x="BMI", y="SleepTime",
        fill=True, thresh=0, levels=20, cmap="rocket"
    )
    plt.title(f'BMI vs Sleep Time\nHeart Disease: {condition}')

plt.tight_layout()
plt.show()
#%% md
# The plot reveals no clear separation between the two groups.
# Individuals with and without heart disease are spread similarly across the BMI–sleep space.
# 
# This supports earlier findings that neither BMI nor sleep time — on their own or combined — show strong differentiating patterns in this dataset.
#%% md
# #### Cohen’s d: Sleep Time Differences by Heart Disease Status
# While the visual distributions of sleep time appeared similar, statistical significance does not always mean practical significance.
# 
# To evaluate the actual magnitude of the difference, we used **Cohen’s d**, a standard measure of effect size, comparing sleep duration between the two heart disease groups.
# 
# 
#%%
df["SleepTime"].dropna(inplace=True)
# Group Sleep Time by Heart Disease Status
group1 = df[df["HeartDisease"] == 0]["SleepTime"]
group2 = df[df["HeartDisease"] == 1]["SleepTime"]

# Compute Mean and Standard Deviation for Each Group
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

# Compute Pooled Standard Deviation
n1, n2 = len(group1), len(group2)
sp = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

# Compute Cohen’s d
cohen_d = (mean1 - mean2) / sp
print(f"Cohen's d: {cohen_d:.4f} (Effect Size)")
#%% md
# Cohen’s d value indicates a very small effect size, meaning that the difference in sleep time between the two groups is statistically negligible.
# 
# Given this minimal impact, we chose to remove the `SleepTime` column from the dataset.
# This allows us to focus on features with stronger predictive potential and avoid noise from weak signals.
# 
#%%
df = df.drop('SleepTime', axis=1)
print(f"SleepTime column removed. Dataset now has {df.shape[1]} columns.")
#%% md
# ### BMI vs Physical Health by Heart Disease Status
# 
# Next, we explore the relationship between **BMI** and **physical health**, and how that relationship differs between individuals with and without heart disease.
# 
# This can help us understand whether the interaction between excess weight and physical condition plays a role in heart disease risk.
# 
# 
#%%
# create separate plots for each Heart Disease category
plt.figure(figsize=(15, 10))
for i, health in enumerate(df['HeartDisease'].unique()):
    plt.subplot(2, 3, i + 1)
    subset = df[df['HeartDisease'] == health]
    sns.kdeplot(
        data=subset, x="BMI", y="PhysicalHealth",
        fill=True, thresh=0, levels=15, cmap="viridis"
    )
    plt.title(f'BMI vs Physical Health\nHeart Disease: {health}')

plt.tight_layout()
plt.show()
#%% md
# These density plots compare BMI and physical health days between individuals with and without heart disease.
# 
# Among those without heart disease, there's a clear concentration at lower BMI (20–30) and fewer physically unhealthy days — indicating generally healthier profiles.
# In contrast, the heart disease group shows two distinct patterns: one similar to the healthy group, and another with significantly more unhealthy days, suggesting increased physical burden.
# 
# This observation aligns with the correlation matrix: **physical health** had a moderate correlation with heart disease (0.17), while **BMI** showed a weaker one (0.05), reinforcing the idea that functional health status may be a more reliable indicator than body weight alone.
# 
#%% md
# ### Physical Health vs Mental Health by Heart Disease Status
# To explore the balance between physical and mental well-being, we plotted these two variables against each other, segmented by heart disease status.
# 
# We’re looking for patterns in how day-to-day health experiences relate to heart disease prevalence.
#%%
# Physical and Mental Health Impact
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(
    data=df[df['HeartDisease'] == 1],
    x="PhysicalHealth", y="MentalHealth",
    fill=True, thresh=0, levels=20, cmap="rocket",
)
plt.title(f'Physical Health vs Mental Health\nHeart Disease: 1')

plt.subplot(1, 2, 2)
sns.kdeplot(
    data=df[df['HeartDisease'] == 0],
    x="PhysicalHealth", y="MentalHealth",
    fill=True, thresh=0, levels=20, cmap="rocket",
)
plt.title(f'Physical Health vs Mental Health\nHeart Disease: 0')
plt.tight_layout()
plt.show()
#%% md
# The density plots show a clear difference between the two heart disease groups.
# 
# Among individuals without heart disease, most report very few physically or mentally unhealthy days — clustering near the origin.
# In contrast, those with heart disease exhibit more variation, with multiple concentrations at higher numbers of both physical and mental health days.
# This suggests that poor physical and mental health often **co-occur** in individuals with heart disease.
# 
# This pattern aligns with the correlation matrix, where **physical health** had a stronger correlation (0.17) than **mental health** (0.03).
# To further assess whether both variables contribute independently to the model — or if they overlap too much — we follow this with a **Variance Inflation Factor (VIF)** analysis to test for multicollinearity.
# 
#%% md
# #### Multicollinearity Analysis (VIF)
# 
# Before continuing, we assess whether some of the variables we’ve used so far might be redundant or highly correlated.
# 
# To do this, we calculate the **Variance Inflation Factor (VIF)** for each feature, focusing especially on variables like mental health and physical health, which appeared to overlap.
# 
#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Copy the dataset
df_vif = df.copy()

# Select only numerical features for VIF analysis
selected_features = ["PhysicalHealth", "MentalHealth"]

# Create a new DataFrame with only selected features
df_vif = df_vif[selected_features]

# Drop any missing values
df_vif = df_vif.dropna()

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(df_vif.columns))]

# Display the results
print(vif_data)
#%% md
# The results show that both **PhysicalHealth** and **MentalHealth** have low VIF values—indicating little to no multicollinearity.
# 
# Thus, despite appearing related, each feature provides distinct information and can be retained in the modeling phase.
#%% md
# ## Heart Disease and Interacting Health Factors
# Some health variables become more meaningful when examined together.
# In this section, we explore how combinations of physical health, mental health, BMI, general health, and asthma interact — and how those interactions differ between individuals with and without heart disease.
# 
# Rather than analyzing variables in isolation, we focus on joint distributions and subgroup analyzes to uncover more subtle patterns.
# 
#%% md
# ### 3D View: Physical & Mental Health vs. BMI by Heart Disease
# We begin this section with a 3D visualization combining three key variables:
# **physical health days**, **mental health days**, and **BMI**.
# 
# By comparing the spatial distributions of individuals with and without heart disease, we aim to see whether certain combinations of these features form identifiable risk zones.
# 
# 
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# First, reload the dataset
df_visualization = pd.read_csv('heart_2020_cleaned.csv')

plt.close('all')

# Convert HeartDisease column to consistent Yes/No strings
df_visualization['HeartDisease'] = df_visualization['HeartDisease'].map({'Yes': 'Yes', 'No': 'No'})

# Sample the data since plotting all points would be overwhelming
sample_size = 5000  # Adjust sample size as needed
df_sampled = df_visualization.sample(n=sample_size, random_state=42)

# Normalize features to 0-1 range for better visualization
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_sampled[['PhysicalHealth', 'MentalHealth', 'BMI']])
df_sampled[['PhysicalHealth_scaled', 'MentalHealth_scaled', 'BMI_scaled']] = scaled_features

# Create 3D plot with lighter background for better contrast
plt.close('all')
fig = plt.figure(figsize=(12, 10), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# Define distinct colors with high contrast
colors = {'No': 'blue', 'Yes': 'red'}

# Plot each category separately
for disease_status in df_sampled['HeartDisease'].unique():
    mask = df_sampled['HeartDisease'] == disease_status
    ax.scatter(
        df_sampled[mask]['PhysicalHealth_scaled'],
        df_sampled[mask]['MentalHealth_scaled'],
        df_sampled[mask]['BMI_scaled'],
        c=colors[disease_status],
        label=f'Heart Disease: {disease_status}',
        alpha=1.0,  # Full opacity for better visibility
        s=50  # Larger point size for visibility
    )

# Customize axes labels
ax.set_xlabel('Physical Health (normalized)', fontweight='bold')
ax.set_ylabel('Mental Health (normalized)', fontweight='bold')
ax.set_zlabel('BMI (normalized)', fontweight='bold')

plt.title('3D Relationship Between Physical Health, Mental Health, and BMI', fontsize=14)
plt.legend()

# Try different viewing angles to better see the data distribution
for angle in [(30, 45), (30, 120), (20, 210), (45, 300)]:
    ax.view_init(elev=angle[0], azim=angle[1])
    plt.draw()
    plt.pause(0.5)  # Display each angle for half a second

plt.show()
#%% md
# The 3D plot reveals that heart disease cases are more concentrated among individuals reporting a higher number of physically unhealthy days,
# while the association with mental health days appears weaker.
# 
# Points also show vertical clustering due to the discrete nature of the day-count variables.
# Heart disease cases span a wide range of BMI values, but there is some concentration in higher BMI areas.
# 
# Overall, the visualization supports earlier findings: **physical health** has a stronger correlation with heart disease (0.17) than **mental health** (0.03), highlighting the more dominant role of physical health burden.
# 
#%% md
# ### BMI Distribution by General Health and Heart Disease Status
# Next, we examine how **BMI** varies across different levels of **self-reported general health**, and how this pattern changes between individuals with and without heart disease.
# 
# This helps us understand whether body weight interacts meaningfully with how people perceive their overall health.
#%%
plt.figure(figsize=(15, 8))
sns.boxplot(x="GenHealth", y="BMI", hue="HeartDisease", data=df)
plt.title('BMI Distribution by General Health and Heart Disease Status')
plt.show()
#%% md
# Across all general health categories — from Excellent to Poor — individuals with better self-reported health tend to have lower BMI values.
# 
# BMI medians and distributions increase gradually as general health declines, with the highest values appearing in the Fair and Poor categories.
# Within each category, individuals with heart disease generally show slightly higher BMI than those without, though the difference is **most noticeable** in the lower health categories.
# 
# Despite these trends, the distributions between groups are still fairly similar overall, and this aligns with the **weak correlation (0.05)** between BMI and heart disease in the correlation matrix — suggesting that BMI alone is not a strong differentiator.
# 
#%% md
# ### Physical Health Distribution by Asthma and Heart Disease Status
# Asthma is a chronic condition that can limit physical activity and general well-being.
# Here, we look at how the number of physically unhealthy days varies based on asthma status and heart disease presence.
# 
# This allows us to see whether asthma exacerbates physical health difficulties in heart disease patients.
# 
#%%
sns.set_theme(style="ticks", palette="pastel")
plt.figure(figsize=(15, 8))
sns.violinplot(x="Asthma", y="PhysicalHealth", hue="HeartDisease", data=df)
plt.title('Distribution of Physical Health Issues across Asthma and Heart Disease Status')
plt.xlabel('Asthma Status')
plt.ylabel('Physical Health Issues (days)')
plt.show()
#%% md
# The violin plot illustrates the complex interaction between physical health issues, asthma, and heart disease.
# 
# Individuals with heart disease consistently report more physically unhealthy days, regardless of asthma status — with wider and higher distributions among heart disease cases.
# Asthma alone also appears to affect the distribution: asthmatic individuals show more symmetrical and concentrated patterns, while non-asthmatics have a more varied spread.
# 
# The combination of both conditions shows a **compounding effect**: people with both heart disease and asthma report the **highest concentration of physical health issues**, while those with neither condition report the fewest.
# 
# This pattern aligns with the correlation matrix: heart disease has a stronger correlation with physical health (0.17), while asthma’s correlation is weaker (0.04), reinforcing that heart disease plays a more dominant role in limiting physical well-being.
# 
#%% md
# ### Age vs. BMI in Asthma Patients by Heart Disease Status
# Finally, we focus on a specific subgroup: individuals with asthma.
# We examine how **age** and **BMI** interact within this group, segmented by heart disease status.
# 
# This allows us to assess whether weight gain across age behaves differently in asthma patients with or without heart disease.
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df_age_astma = pd.read_csv('heart_2020_cleaned.csv')

# age mapping (converting categorical age to numeric)
age_mapping = {
    '18-24': 21,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-64': 62,
    '65-69': 67,
    '70-74': 72,
    '75-79': 77,
    '80 or older': 82
}

# create new column with numeric age
df_age_astma['AgeNumeric'] = df_age_astma['AgeCategory'].map(age_mapping)

# filter data for skin cancer cases
astma = df_age_astma[df_age_astma['Asthma'] == 'Yes']

# create the joint plot with modified parameters
g = sns.jointplot(data=astma,
                  x="AgeNumeric",
                  y="BMI",
                  hue="HeartDisease",
                  kind="kde",
                  height=10)

plt.suptitle('Relationship between Age and BMI for Asthma Patients by Heart Disease Status',
             y=1.02,
             fontsize=12)

# improve axis labels
g.ax_joint.set_xlabel('Age', fontsize=10)
g.ax_joint.set_ylabel('BMI', fontsize=10)

plt.show()
#%% md
# This scatter plot explores how BMI varies with age among asthma patients, segmented by heart disease status.
# 
# A general upward trend in BMI with increasing age is observed for both groups.
# However, individuals with heart disease tend to have **slightly higher BMI** across most age ranges.
# This may suggest that asthma patients who also develop heart disease are more likely to be overweight or obese, possibly due to compounded health challenges or reduced physical activity.
# 
# That said, there is considerable overlap between the two groups, indicating that **BMI alone is not sufficient** to explain heart disease in asthma patients.
# 
#%% md
# ## PCA (Principal Component Analysis)
#%% md
# PCA is applied to reduce the dimensionality of health-related variables — specifically **BMI**, **PhysicalHealth**, **MentalHealth**, and **GenHealth** — while preserving the most significant variance in the data.
# 
# This technique helps uncover underlying structure and patterns, and allows us to assess whether variables such as **sex** and **age** correspond with distinct clusters in the transformed space.
# By projecting the data onto principal components, we evaluate how well these demographic categories align with health-related variation.
# 
#%% md
# ### PCA Projection of Health Metrics Colored by Sex
# 
# To explore whether sex plays a meaningful role in shaping health profiles, we apply PCA to four key health metrics:
# **BMI**, **PhysicalHealth**, **MentalHealth**, and **GenHealth**.
# 
# The resulting 2D projection helps visualize how individuals cluster in the principal component space, with colors representing male and female categories.
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Copy dataset
df_copy = df.copy()

# Convert categorical features to numerical
df_copy['Sex'] = df_copy['Sex'].replace({'Male': 1, 'Female': 0}).astype(int)

# Convert GenHealth to numerical values
genhealth_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}
df_copy['GenHealth'] = df_copy['GenHealth'].replace(genhealth_mapping)

# Select only the intended numerical columns
numerical_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'GenHealth']
X = df_copy[numerical_cols]  # Ensure we are only using 4 features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create DataFrame for PCA results
principalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add HeartDisease and Sex columns for analysis
principalDf['HeartDisease'] = df_copy['HeartDisease']
principalDf['Sex'] = df_copy['Sex']

# Plot PCA scatterplot
plt.figure(figsize=(15, 10))
sns.scatterplot(data=principalDf,
                x="PC1",
                y="PC2",
                hue="Sex",
                palette="Set1",
                alpha=0.7)

plt.title('PCA of Health Metrics Colored by Sex')
plt.show()
#%% md
# The PCA scatter plot shows a high degree of overlap between males and females, indicating that sex does not significantly separate within the reduced feature space. This suggests that the selected health metrics do not vary meaningfully based on sex, making it a weak differentiating factor in this context.
#%% md
# ### PCA Projection of Health Metrics Colored by Age Category
# We now repeat the PCA projection using the same four health metrics — **BMI**, **PhysicalHealth**, **MentalHealth**, and **GenHealth** —
# but this time color the points by **age category** instead of sex.
# 
# This helps us examine whether health-related variation in the dataset aligns with age groups, and whether PCA captures age-related patterns.
#%%
# create a DataFrame with principal components
principalDf2 = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])

# Add AgeCategory column for coloring (using the existing 10-year grouping)
principalDf2['AgeCategory'] = df_copy['AgeCategory']

# Create the scatter plot
plt.figure(figsize=(15, 10))
sns.scatterplot(data=principalDf2,
                x="PC 1",
                y="PC 2",
                hue="AgeCategory",
                palette="Set1",
                alpha=0.7)

# Customize the plot
plt.title('PCA of Health Metrics Colored by Age Category (10-year groups)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()

# Display explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Total explained variance: {sum(explained_variance):.2f}")

plt.show()
#%% md
# The age groups are distributed across the first two principal components (PC1 and PC2). The color coding by AgeCategory suggests that the variance in the dataset does not distinctly separate age groups, indicating that age may not strongly influence the primary variance captured by PCA. Additionally, the dispersed nature of points with no clear clustering suggests that the selected health metrics do not differentiate strongly across age categories in the reduced dimensionality space.
#%% md
# # Summary of Exploratory Data Analysis
# 
# Our exploratory analysis uncovered key insights into heart disease risk and its associated factors.
# 
# We began by observing a clear class imbalance: only 8.6% of individuals in the dataset were classified as having heart disease.
# This imbalance has important implications for modeling and evaluation strategies later on.
# 
# Several variables emerged as strongly associated with heart disease:
# - **Age** (0.23), **stroke** (0.20), **difficulty walking** (0.20), **diabetes** (0.18), and **physical health** (0.17) showed the highest positive correlations.
# - **General health** demonstrated a strong negative correlation (-0.24), highlighting its importance as a protective factor.
# 
# Our visual analyses revealed meaningful patterns:
# - **Smokers who don’t drink** had the highest heart disease prevalence (12.8%),
# - **Inactive individuals with poor general health** showed a heart disease rate of 35.5%, compared to just 2.1% in active individuals reporting excellent health.
# - The combination of **diabetes and stroke** resulted in the highest observed rate — 48.1%.
# 
# Based on these findings, we made targeted feature selection decisions:
# - We removed the **Race** variable due to its minimal correlation with heart disease (-0.04) and potential to introduce bias.
# - We also removed **SleepTime**, following visual inspection and a **Cohen's d** analysis which revealed a negligible effect size (-0.0202), suggesting it does not meaningfully distinguish between heart disease groups.
# 
# Finally, statistical tests such as correlation matrices, VIF analysis, and PCA confirmed that the selected variables contribute **independent and non-redundant** information.
# The PCA projections also showed that **sex** and **age** do not meaningfully separate individuals based on health metrics alone, reinforcing our focus on behavioral and medical features over demographics.
# 
# Altogether, the EDA provided a clear foundation for modeling — both in identifying impactful variables and in filtering out weaker ones.
# 
#%% md
# # 3. Predictive Modeling
# 
# With a clear understanding of the data and the most impactful variables identified during our exploratory analysis,
# we now move on to the next stage of the project: building models to predict heart disease risk.
# 
# We implemented a variety of machine learning algorithms from different modeling families —
# each based on fundamentally different mathematical principles. This diversity allows us to:
# 
# 1. Assess how different learning paradigms perform on this specific healthcare problem
# 2. Identify which modeling approach offers the best balance between precision and recall
# 3. Understand the trade-offs between interpretability and predictive power
# 
# The models we selected include:
# 
# - **Logistic Regression** – a linear, interpretable baseline model
# - **Random Forest** – an ensemble of decision trees that captures feature interactions
# - **Neural Network (MLP)** – a deep learning model that handles complex, non-linear patterns
# - **XGBoost** – a state-of-the-art gradient boosting model for structured data
# - **Hierarchical Clustering** – an unsupervised method to detect natural patient subgroups
# 
# We present the models in order of increasing complexity — from interpretable linear models to more flexible "black box" approaches.
# For each supervised algorithm, we trained and evaluated two versions:
# one using the original imbalanced data, and one using a SMOTE-balanced dataset.
# 
# This dual evaluation helps us understand how class imbalance impacts model performance,
# especially in a healthcare context where missing a positive case could have serious consequences.
# 
# 
# ## Data Preparation for Modeling
# Before training our models, we prepare the dataset to ensure effective learning and fair evaluation.
# This involves transforming categorical variables, handling severe class imbalance, and scaling numerical features —
# all essential for stable model performance.
#%% md
# ### Splitting and Encoding the Data
# We begin by converting categorical variables into numerical format.
# Binary variables (e.g., Smoking, Stroke) are mapped to 0/1, while multi-category features are transformed using `LabelEncoder`.
# 
# We then split the data into **features (X)** and **target (y)**, and divide it into **training (80%)** and **test (20%)** sets.
# 
#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings("ignore")

# Simple mapping for all categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        # Explicit mapping for binary columns
        if set(df[col].unique()).issubset({'Yes', 'No'}):
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else:
            # LabelEncoder for other categorical columns
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

# Making sure that we deleted those columns
df = df.drop(['Race', 'SleepTime'], axis=1, errors='ignore')

# Prepare features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%% md
# ### Addressing Class Imbalance with SMOTE
# As seen in the EDA, the dataset is highly imbalanced — only 8.6% of samples are labeled as having heart disease.
# This can lead models to ignore the minority class.
# 
# To correct for this, we apply **SMOTE** (Synthetic Minority Over-sampling Technique),
# which generates synthetic samples of the minority class based on feature-space similarities.
# 
#%%
# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original training set distribution: {pd.Series(y_train).value_counts()}")
print(f"Balanced training set distribution: {pd.Series(y_train_balanced).value_counts()}")

# Visualize the class distribution before and after balancing
plt.figure(figsize=(12, 5))

# Before balancing
plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette='Set2')
plt.title('Heart Disease Distribution Before Balancing')
plt.xlabel('Heart Disease')
plt.ylabel('Count')

# After balancing
plt.subplot(1, 2, 2)
sns.countplot(x=y_train_balanced, palette='Set2')
plt.title('Heart Disease Distribution After SMOTE')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
#%% md
# SMOTE transformed the dataset from an approximate 10.7:1 imbalance
# to a perfectly balanced 1:1 ratio — with 233,802 samples in each class in the training set.
# 
# This ensures that the model is exposed equally to both outcomes during training, improving its ability to detect heart disease cases.
# 
#%% md
# ### Feature Scaling
# After balancing, we apply feature scaling to all numeric columns to bring them to a similar range,
# which is especially important for distance-based models and neural networks.
#%%
# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print("Data is now ready for model training:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train_balanced shape: {y_train_balanced.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_test shape: {y_test.shape}")
#%% md
# The balanced training set now contains **467,604 samples** and **15 features**,
# while the unbalanced test set retains its original distribution with **63,913 samples**.
# 
# This setup allows the model to learn from balanced data while being evaluated on realistic, real-world class distributions.
#%% md
# ## Supervised Learning Models
#%% md
# ### Logistic Regression
# We begin with **logistic regression**, a simple yet powerful linear model often used as a baseline for binary classification tasks.
# 
# Its high interpretability and straightforward implementation make it a good starting point for evaluating how well the features in our dataset can distinguish between individuals with and without heart disease.
# 
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Train on original data
model_orig = LogisticRegression(max_iter=1000)
model_orig.fit(X_train, y_train)
y_pred_orig = model_orig.predict(X_test)

# Train on balanced data
model_balanced = LogisticRegression(max_iter=1000)
model_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = model_balanced.predict(X_test)

# Compare results
print("Logistic Regression Results:")
print("\nBefore SMOTE:")
print(classification_report(y_test, y_pred_orig))

print("\nAfter SMOTE:")
print(classification_report(y_test, y_pred_balanced))

# Create simple bar chart comparison for heart disease detection (class 1)
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [
    precision_score(y_test, y_pred_orig, pos_label=1),
    recall_score(y_test, y_pred_orig, pos_label=1),
    f1_score(y_test, y_pred_orig, pos_label=1)
]
after_values = [
    precision_score(y_test, y_pred_balanced, pos_label=1),
    recall_score(y_test, y_pred_balanced, pos_label=1),
    f1_score(y_test, y_pred_balanced, pos_label=1)
]

# Simple bar chart
plt.figure(figsize=(8, 5))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width / 2, before_values, width, label='Before SMOTE')
plt.bar(x + width / 2, after_values, width, label='After SMOTE')

plt.ylabel('Score')
plt.title('Heart Disease Detection Performance')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)

plt.show()
#%% md
# The model achieved a high overall accuracy (0.91), but this number is misleading due to the class imbalance.
# It performed well on the majority class (non-heart disease) with 0.99 recall, but poorly on detecting heart disease cases — with recall dropping to just 0.09.
# 
# This confirms the concern raised in the EDA: with only 8.6% of cases labeled positive, the model tends to ignore minority cases unless we address the imbalance directly.
#%% md
# ### Random Forest
# Next, we apply a **Random Forest**, an ensemble model based on decision trees.
# This model captures non-linear relationships and feature interactions that logistic regression may miss.
# 
# It also provides feature importance rankings, helping us understand which variables contribute most to prediction.
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Train on original data
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
rf_orig.fit(X_train, y_train)
y_pred_orig = rf_orig.predict(X_test)

# Train on balanced data
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = rf_balanced.predict(X_test)

# Compare results
print("Random Forest Results:")
print("\nBefore SMOTE:")
print(classification_report(y_test, y_pred_orig))

print("\nAfter SMOTE:")
print(classification_report(y_test, y_pred_balanced))

# Create simple bar chart comparison for heart disease detection (class 1)
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [
    precision_score(y_test, y_pred_orig, pos_label=1),
    recall_score(y_test, y_pred_orig, pos_label=1),
    f1_score(y_test, y_pred_orig, pos_label=1)
]
after_values = [
    precision_score(y_test, y_pred_balanced, pos_label=1),
    recall_score(y_test, y_pred_balanced, pos_label=1),
    f1_score(y_test, y_pred_balanced, pos_label=1)
]

# Simple bar chart
plt.figure(figsize=(8, 5))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width / 2, before_values, width, label='Before SMOTE')
plt.bar(x + width / 2, after_values, width, label='After SMOTE')

plt.ylabel('Score')
plt.title('Heart Disease Detection Performance')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)

plt.show()
#%% md
# The Random Forest classifier demonstrates strong performance metrics with an overall accuracy of 0.90. For heart disease detection, it achieves a precision of 0.32 and recall of 0.15. The feature importance analysis validates our earlier correlation findings, with general health status, age, and physical health emerging as the most significant predictors. This aligns with medical understanding of heart disease risk factors.
#%% md
# ### Neural Network
# To explore a more flexible, non-linear modeling approach, we implemented a **Multi-Layer Perceptron (MLP)** neural network.
# This architecture is well-suited for capturing complex interactions between features, especially in high-dimensional health data.
# 
# We used two hidden layers (100 and 50 neurons), and tested the model on both imbalanced and SMOTE-balanced datasets to evaluate its sensitivity to data distribution.
#%%
from matplotlib.patches import Circle, FancyArrowPatch

def plot_neural_network(layers_sizes, title="Neural Network Architecture"):
    # Set figure size
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca()
    ax.axis('off')

    # Layer positions
    n_layers = len(layers_sizes)
    layer_centers = np.linspace(0.2, 0.8, n_layers)

    # Vertical spacing for neurons
    max_neurons = max(layers_sizes)
    v_spacing = 0.9 / max(max_neurons, 1)

    # Draw layers and neurons
    for i, layer_size in enumerate(layers_sizes):
        layer_name = "Input" if i == 0 else "Output" if i == n_layers - 1 else f"Hidden {i}"

        # Neurons in current layer
        for j in range(layer_size):
            # Limit the number of displayed neurons if there are too many
            if layer_size <= 10 or (j < 4 or j >= layer_size - 4):
                x = layer_centers[i]
                y = 0.5 + (j - layer_size / 2) * v_spacing

                # Draw neuron
                circle = Circle((x, y), 0.02,
                             color='#3498db' if i == 0 else '#e74c3c' if i == n_layers - 1 else '#2ecc71',
                             ec='k', zorder=4)
                ax.add_patch(circle)

                # Connect to next layer
                if i < n_layers - 1:
                    next_layer_size = layers_sizes[i + 1]
                    next_x = layer_centers[i + 1]

                    # Connect to all neurons in next layer
                    for k in range(next_layer_size):
                        if next_layer_size <= 10 or (k < 4 or k >= next_layer_size - 4):
                            next_y = 0.5 + (k - next_layer_size / 2) * v_spacing
                            line = FancyArrowPatch((x, y), (next_x, next_y),
                                                  connectionstyle="arc3,rad=0.1",
                                                  arrowstyle="-", color='gray', alpha=0.2)
                            ax.add_patch(line)

            # Add dots to indicate more neurons not shown
            if layer_size > 10 and j == 4:
                plt.text(layer_centers[i], 0.5, "...", fontsize=20, ha='center')

        # Layer labels
        plt.text(layer_centers[i], 0.15, f"{layer_name}\n({layer_size} neurons)",
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Draw our neural network
input_size = 16  # Number of features in your model (adjust as needed)
plot_neural_network([input_size, 100, 50, 1], "MLP Architecture for Heart Disease Prediction")
#%%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Neural Network before SMOTE
mlp_orig = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
    activation='relu',             # ReLU activation function
    solver='adam',                 # Optimization algorithm
    alpha=0.0001,                  # Regularization parameter
    batch_size='auto',             # Automatic batch size
    learning_rate='adaptive',      # Adaptive learning rate
    max_iter=500,                  # Maximum number of iterations
    random_state=42                # Random seed for reproducibility
)

# Train the model on imbalanced data
mlp_orig.fit(X_train, y_train)
y_pred_orig = mlp_orig.predict(X_test)

# Neural Network after SMOTE
mlp_balanced = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)

# Train the model on SMOTE-balanced data
mlp_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = mlp_balanced.predict(X_test)

# Compare results
print("Neural Network Results:")
print("\nBefore SMOTE:")
print(classification_report(y_test, y_pred_orig))

print("\nAfter SMOTE:")
print(classification_report(y_test, y_pred_balanced))

# Create a bar chart for comparing heart disease detection (class 1)
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [
    precision_score(y_test, y_pred_orig, pos_label=1),
    recall_score(y_test, y_pred_orig, pos_label=1),
    f1_score(y_test, y_pred_orig, pos_label=1)
]
after_values = [
    precision_score(y_test, y_pred_balanced, pos_label=1),
    recall_score(y_test, y_pred_balanced, pos_label=1),
    f1_score(y_test, y_pred_balanced, pos_label=1)
]

# Create simple bar chart
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width / 2, before_values, width, label='Before SMOTE', color='skyblue')
plt.bar(x + width / 2, after_values, width, label='After SMOTE', color='salmon')

# Add numerical values above bars
for i, v in enumerate(before_values):
    plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

for i, v in enumerate(after_values):
    plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

plt.ylabel('Score')
plt.title('Neural Network Performance on Heart Disease Detection')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
#%% md
# The neural network's performance changed dramatically before and after SMOTE:
# - Without balancing, recall for heart disease was only 11%, despite high overall accuracy.
# - After SMOTE, recall jumped to 65%, showing much better detection of heart disease cases — though precision dropped to 20%.
# 
# This sharp contrast highlights how neural networks are especially sensitive to data imbalance.
# Compared to other models, the MLP learned more complex patterns from the balanced data, making it effective for early risk detection — even at the cost of some false positives.
# 
#%% md
# ### XGBoost
# We then applied **XGBoost**, a gradient boosting model known for its speed, regularization, and strong performance on structured data.
# 
# Its iterative tree-based optimization allows it to capture both linear and non-linear patterns, often outperforming other models on tabular datasets.
#%%
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
# Train on original data
xgb_orig = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
xgb_orig.fit(X_train, y_train)
y_pred_orig = xgb_orig.predict(X_test)

# Train on balanced data
xgb_balanced = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
xgb_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = xgb_balanced.predict(X_test)

# Compare results
print("XGBoost Results:")
print("\nBefore SMOTE:")
print(classification_report(y_test, y_pred_orig))

print("\nAfter SMOTE:")
print(classification_report(y_test, y_pred_balanced))

# Create simple bar chart comparison for heart disease detection (class 1)
metrics = ['Precision', 'Recall', 'F1-Score']
before_values = [
    precision_score(y_test, y_pred_orig, pos_label=1),
    recall_score(y_test, y_pred_orig, pos_label=1),
    f1_score(y_test, y_pred_orig, pos_label=1)
]
after_values = [
    precision_score(y_test, y_pred_balanced, pos_label=1),
    recall_score(y_test, y_pred_balanced, pos_label=1),
    f1_score(y_test, y_pred_balanced, pos_label=1)
]

# Simple bar chart
plt.figure(figsize=(8, 5))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width / 2, before_values, width, label='Before SMOTE')
plt.bar(x + width / 2, after_values, width, label='After SMOTE')

plt.ylabel('Score')
plt.title('Heart Disease Detection Performance')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)

plt.show()
#%% md
# Before SMOTE, XGBoost performed similarly to the other models — high accuracy (0.92) but very low recall (0.08) for heart disease.
# After SMOTE, recall soared to 0.73, while precision fell to 0.20 — a tradeoff we also saw in the MLP.
# 
# With an F1-score of 0.32 after balancing, XGBoost emerged as one of the strongest models for detecting heart disease, offering a solid balance between sensitivity and overall performance.
# 
#%% md
# ## Unsupervised Analysis
# 
# While supervised models rely on labeled data to learn patterns and make predictions,
# unsupervised techniques like clustering help us uncover natural groupings in the data —
# without using predefined outcome labels.
# 
# In this section, we apply **hierarchical clustering** to identify distinct patient profiles based on shared health characteristics.
# This complementary analysis allows us to explore whether meaningful risk groups emerge from the data structure itself,
# and whether these clusters align with known heart disease risk.
# 
# ### Hierarchical Clustering of Patient Profiles
# 
# Hierarchical clustering groups individuals based on similarity across multiple features, using **Euclidean  distance** to handle a mix of numeric and categorical variables.
# 
# This technique allows us to discover naturally occurring subpopulations —
# which may help in designing targeted prevention strategies or refining future predictive models.
# 
# We analyze the results both **before and after applying SMOTE**, to understand how class imbalance affects the formation and interpretation of these clusters.
# 
# 
#%%
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# Check if X_train is a DataFrame or numpy array
if isinstance(X_train, pd.DataFrame):
    # If DataFrame, use .iloc for indexing
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train), size=5000, replace=False)
    X_train_sample = X_train.iloc[sample_indices].values
    y_train_sample = y_train.iloc[sample_indices].values if isinstance(y_train, pd.Series) else y_train[sample_indices]
else:
    # If numpy array, use direct indexing
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train), size=5000, replace=False)
    X_train_sample = X_train[sample_indices]
    y_train_sample = y_train[sample_indices]

# Calculate distance matrix (using Euclidean distance for symmetric matrix)
distances_orig = pairwise_distances(X_train_sample, metric='euclidean')

# Make sure the matrix is symmetric
distances_orig = (distances_orig + distances_orig.T) / 2

# Perform hierarchical clustering with 5 clusters
n_clusters = 5
cluster_orig = AgglomerativeClustering(n_clusters=n_clusters,
                                 metric='precomputed',
                                 linkage='complete')
clusters_orig = cluster_orig.fit_predict(distances_orig)

# Calculate heart disease rates by cluster
df_results_orig = pd.DataFrame({
    'Cluster': clusters_orig,
    'HeartDisease': y_train_sample
})
heart_disease_by_cluster_orig = df_results_orig.groupby('Cluster')['HeartDisease'].mean() * 100

# ---- Hierarchical Clustering on SMOTE Balanced Data ----
# Sample the balanced dataset
if isinstance(X_train_balanced, pd.DataFrame):
    # If DataFrame, use .iloc for indexing
    np.random.seed(42)
    balanced_sample_indices = np.random.choice(len(X_train_balanced), size=5000, replace=False)
    X_train_balanced_sample = X_train_balanced.iloc[balanced_sample_indices].values
    y_train_balanced_sample = y_train_balanced.iloc[balanced_sample_indices].values if isinstance(y_train_balanced, pd.Series) else y_train_balanced[balanced_sample_indices]
else:
    # If numpy array, use direct indexing
    np.random.seed(42)
    balanced_sample_indices = np.random.choice(len(X_train_balanced), size=5000, replace=False)
    X_train_balanced_sample = X_train_balanced[balanced_sample_indices]
    y_train_balanced_sample = y_train_balanced[balanced_sample_indices]

# Calculate distance matrix (using Euclidean distance for symmetric matrix)
distances_balanced = pairwise_distances(X_train_balanced_sample, metric='euclidean')

# Make sure the matrix is symmetric
distances_balanced = (distances_balanced + distances_balanced.T) / 2

# Perform hierarchical clustering with 5 clusters
cluster_balanced = AgglomerativeClustering(n_clusters=n_clusters,
                                    metric='precomputed',
                                    linkage='complete')
clusters_balanced = cluster_balanced.fit_predict(distances_balanced)

# Calculate heart disease rates by cluster
df_results_bal = pd.DataFrame({
    'Cluster': clusters_balanced,
    'HeartDisease': y_train_balanced_sample
})
heart_disease_by_cluster_bal = df_results_bal.groupby('Cluster')['HeartDisease'].mean() * 100

# ---- Create dendrograms -----
# Prepare data for dendrograms
condensed_dist_orig = squareform(distances_orig)
linkage_matrix_orig = linkage(condensed_dist_orig, method='complete')

condensed_dist_bal = squareform(distances_balanced)
linkage_matrix_bal = linkage(condensed_dist_bal, method='complete')

# Create dendrograms side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Dendrogram before SMOTE (left)
dendrogram(linkage_matrix_orig, ax=ax1)
ax1.set_title('Hierarchical Clustering Dendrogram (Before SMOTE)')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')

# Dendrogram after SMOTE (right)
dendrogram(linkage_matrix_bal, ax=ax2)
ax2.set_title('Hierarchical Clustering Dendrogram (After SMOTE)')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Distance')

plt.tight_layout()
plt.show()

# ---- Create heart disease rate graphs ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Before SMOTE - left side
bars_orig = ax1.bar(range(n_clusters), heart_disease_by_cluster_orig, color='skyblue')
ax1.set_title('Heart Disease Rate by Cluster (Before SMOTE)')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Heart Disease Rate (%)')
ax1.set_xticks(range(n_clusters))

# Add percentage labels on bars
for bar, percentage in zip(bars_orig, heart_disease_by_cluster_orig):
    ax1.text(bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.5,
            f'{percentage:.1f}%',
            ha='center', va='bottom')

# After SMOTE - right side
bars_bal = ax2.bar(range(n_clusters), heart_disease_by_cluster_bal, color='skyblue')
ax2.set_title('Heart Disease Rate by Cluster (After SMOTE)')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Heart Disease Rate (%)')
ax2.set_xticks(range(n_clusters))

# Add percentage labels on bars
for bar, percentage in zip(bars_bal, heart_disease_by_cluster_bal):
    ax2.text(bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.5,
            f'{percentage:.1f}%',
            ha='center', va='bottom')

plt.tight_layout()
plt.show()
#%% md
# The clustering revealed five distinct patient profiles.
# In the original, imbalanced data, heart disease rates varied modestly across clusters (7.1%–26.9%).
# After SMOTE, these rates increased substantially (45.5%–72.3%), with clearer separation between risk levels.
# 
# The dendrograms showed how balancing influenced the clustering structure, enabling more meaningful groupings.
# The bottom graphs illustrate how SMOTE clearly amplifies distinctions between clusters, making risk groups more identifiable.
# 
# These clusters likely represent patients sharing critical risk factors, which could guide targeted interventions or future feature selection.
# Overall, this analysis complements our supervised models by identifying natural risk strata—potentially guiding future interventions or model refinement.
#%% md
# # 4. Model Comparison and Interpretation
# 
# After training multiple machine learning models, we now shift our focus to evaluating their performance side by side.
# This includes comparing predictive accuracy, recall, precision, and F1-scores for detecting heart disease — particularly under the impact of SMOTE balancing.
# 
# We also examine **model interpretability**, both at the global level using feature importance (via SHAP),
# and at the individual level through case-specific SHAP analysis that helps us understand how decisions are made for specific patients.
# 
# ## Comparing Model Performance Metrics
# 
# To assess how well each model handles heart disease detection, we compare the **precision**, **recall**, and **F1-score**
# for each classifier on the **SMOTE-balanced** test set.
# 
# The following chart summarizes these metrics, focusing on the minority class (heart disease cases).
# 
# 
# 
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame with model performance metrics after SMOTE
metrics_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'Neural Network', 'XGBoost'],
    'Accuracy': [0.70, 0.82, 0.74, 0.73],
    'Precision (No HD)': [0.97, 0.93, 0.96, 0.97],
    'Recall (No HD)': [0.70, 0.87, 0.75, 0.73],
    'F1 (No HD)': [0.81, 0.90, 0.84, 0.83],
    'Precision (HD)': [0.19, 0.20, 0.20, 0.20],
    'Recall (HD)': [0.73, 0.35, 0.65, 0.73],
    'F1 (HD)': [0.30, 0.26, 0.30, 0.32]
}

metrics_df = pd.DataFrame(metrics_data)

# Display the comparison table
print("Model Performance Comparison (After SMOTE):")
print(metrics_df.to_string(index=False))

# Create a bar plot for comparison
plt.figure(figsize=(15, 8))
metrics_to_plot = ['Accuracy', 'Precision (HD)', 'Recall (HD)', 'F1 (HD)']
plot_data = metrics_df.melt(id_vars=['Model'],
                            value_vars=metrics_to_plot,
                            var_name='Metric',
                            value_name='Score')

# Plot the comparison using seaborn
sns.barplot(x='Model', y='Score', hue='Metric', data=plot_data)
plt.title('Model Performance Comparison (After SMOTE)')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%% md
# ## Summary of Model Performance
# 
# Comparing the models reveals clear differences in how each algorithm handles the trade-off between sensitivity (recall) and specificity (precision):
# 
# - **XGBoost** emerged as the most balanced model, with an F1-score of 0.32 — combining a strong recall (0.73) with moderate precision (0.20).
# - **Logistic Regression** achieved the highest recall (0.73), making it effective for catching more heart disease cases, though at the cost of precision (0.19).
# - **Neural Network** showed slightly weaker performance with recall of 0.65 and an F1-score of 0.30.
# - **Random Forest**, while yielding the highest overall accuracy (0.82), had lower recall (0.15) for heart disease, making it less suitable when sensitivity is critical.
# 
# Despite the improvements brought by SMOTE, all models still performed significantly better on the majority class (non-heart disease),
# with F1-scores ranging from 0.81 to 0.92 — compared to 0.26 to 0.32 for heart disease cases.
# 
# These results reflect the inherent challenge of predicting rare medical conditions, and emphasize the need to balance recall with precision in real-world applications.
# 
# **Practical recommendations** based on modeling goals:
# - If **maximizing recall** is most important (e.g., screening), choose **Logistic Regression**.
# - If **balancing precision and recall** is the goal, **XGBoost** provides the best overall compromise.
# - If **minimizing false positives** is the priority, **Random Forest** may be more appropriate — though at the cost of missing more true cases.
# 
#%% md
# ## Global Model Interpretability with SHAP
# 
# To understand which features influenced our model predictions, we applied **SHAP (SHapley Additive exPlanations)** —
# a model-agnostic method for interpreting complex machine learning outputs.
# 
# The SHAP summary plot below ranks features by their overall impact on the model's predictions,
# and shows how different feature values push the prediction toward higher or lower risk.
# 
#%%
import shap
import os
import warnings
import sys

# Disable all warnings
warnings.filterwarnings('ignore')

# Redirect stdout temporarily to suppress progress output
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Disable kmeans iterations display
shap.utils._legacy.kmeans.kmeans = lambda X, k, **kwargs: (X[:k], None)

try:
    # Convert X_test back to DataFrame with column names
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    # Create SHAP explainer without verbose parameter
    explainer = shap.Explainer(xgb_balanced, X_test_df)
    shap_values = explainer(X_test_df)
finally:
    # Restore stdout
    sys.stdout.close()
    sys.stdout = original_stdout

# Create beeswarm plot
shap.plots.beeswarm(shap_values)
#%% md
# The summary plot shows that **age** is the most influential feature:
# higher age (red) significantly increases the risk of heart disease, while younger age (blue) reduces it.
# 
# **General Health** is the second most impactful feature, with poor health ratings increasing risk.
# Other contributors include **physical activity**, **diabetes**, and **smoking**, each playing a role in how the model evaluates risk.
# 
#%% md
# ## Individual-Level SHAP Analysis
# 
# While the global SHAP summary shows overall feature importance, it does not explain individual predictions.
# To address this, we performed case-specific SHAP analysis using **Waterfall plots** for four representative patients:
# 
# 1. **True Positive** – Correctly identified heart disease case
# 2. **True Negative** – Correctly identified healthy case
# 3. **False Positive** – Incorrectly flagged healthy patient
# 4. **False Negative** – Missed heart disease case
# 
# These plots reveal how specific features influenced each individual decision — helping us understand model behavior in real-world contexts.
# 
#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import warnings
import matplotlib as mpl

# Suppress warnings
warnings.filterwarnings("ignore")

# Predictions using existing balanced model
y_pred = xgb_balanced.predict(X_test)
y_pred_proba = xgb_balanced.predict_proba(X_test)[:, 1]

# Create results DataFrame
results_df = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted': y_pred,
    'Probability': y_pred_proba
})

# Find representative cases
true_positive_indices = results_df[(results_df['True_Label'] == 1) & (results_df['Predicted'] == 1)].index
true_positive = true_positive_indices[0] if len(true_positive_indices) > 0 else None

true_negative_indices = results_df[(results_df['True_Label'] == 0) & (results_df['Predicted'] == 0)].index
true_negative = true_negative_indices[0] if len(true_negative_indices) > 0 else None

false_positive_indices = results_df[(results_df['True_Label'] == 0) & (results_df['Predicted'] == 1)].index
false_positive = false_positive_indices[0] if len(false_positive_indices) > 0 else None

false_negative_indices = results_df[(results_df['True_Label'] == 1) & (results_df['Predicted'] == 0)].index
false_negative = false_negative_indices[0] if len(false_negative_indices) > 0 else None

# Collect case indices and labels
case_indices = [idx for idx in [true_positive, true_negative, false_positive, false_negative] if idx is not None]
case_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative'][:len(case_indices)]

# Create SHAP explainer
X_test_df = pd.DataFrame(X_test.reset_index(drop=True), columns=X.columns)
explainer = shap.TreeExplainer(xgb_balanced)

# Define consistent figure size for all plots - making wider to handle the left margin
fixed_figsize = (6.5, 3.5)

# Set consistent values for all plots
border_width = 2
border_color = 'black'
font_size = 10

# Reset any previous matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)
# Force all plots to have exactly the same dimensions
mpl.rcParams['savefig.bbox'] = 'standard'  # Don't use tight bbox
mpl.rcParams['savefig.pad_inches'] = 0.1   # Consistent padding

# Create and save waterfall plots with consistent borders
for i, idx in enumerate(case_indices):
    # Get feature values for this specific case
    x_instance = X_test_df.iloc[[idx]]

    # Get SHAP values
    shap_values = explainer(x_instance)

    # Create figure with fixed size
    fig = plt.figure(figsize=fixed_figsize)

    # Create waterfall plot with smaller inner plot
    shap.plots.waterfall(
        shap_values[0],
        max_display=5,
        show=False
    )

    # Add title
    case_type = case_labels[i]
    true_label = "Has Heart Disease" if results_df['True_Label'].iloc[idx] == 1 else "No Heart Disease"
    pred_label = "Has Heart Disease" if results_df['Predicted'].iloc[idx] == 1 else "No Heart Disease"
    prob = results_df['Probability'].iloc[idx] * 100

    plt.title(f'Case Analysis: {case_type}\n'
              f'True Label: {true_label}, Prediction: {pred_label}, Probability: {prob:.1f}%',
              fontsize=font_size, fontweight='bold')

    # Add a border to the figure
    fig.patch.set_edgecolor(border_color)
    fig.patch.set_linewidth(border_width)

    plt.subplots_adjust(left=0.25, right=0.80, top=0.80, bottom=0.20)

    # Save with consistent size
    filename = f"case_{case_type.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=100, bbox_inches=None)  # No auto-cropping
    plt.close()

# Display all saved charts
from IPython.display import Image, display
for case_type in case_labels:
    display(Image(filename=f"case_{case_type.replace(' ', '_')}.png"))
#%% md
# ## Interpretation of Individual Model Decisions
# 
# The SHAP Waterfall plots offer a deeper view into how the model processes feature contributions:
# 
# - In the **True Positive**, strong contributions from **Age**, **GenHealth**, and **Diabetic** pushed the prediction above the threshold.
# - In the **True Negative**, protective factors like young age, good general health, and low asthma impact led to a low predicted probability.
# - The **False Positive** case showed overreliance on GenHealth and PhysicalActivity, potentially overestimating risk despite contradictory signals like asthma.
# - The **False Negative** case involved a patient with good health indicators that offset risk signals from age and other factors — leading to a missed prediction.
# 
# These examples illustrate both the model’s strengths in identifying classic risk profiles, and its limitations in handling edge cases with mixed indicators.
# They also highlight opportunities to improve model robustness and reduce misclassification in future iterations.
# Future improvements might focus on better handling of unusual risk factor combinations and atypical disease presentations.
#%% md
# # 5. Final Reflections and Future Directions
# 
# ## From Data to Impact: Reconnecting with David
# 
# At the start of this journey, we imagined David — a man in his 50s, seemingly healthy, but at risk for heart disease.
# Our goal was to understand whether data alone could have warned him earlier, and how machine learning might help identify people like him before it's too late.
# 
# This project set out to explore that question through real data, rigorous analysis, and modern predictive tools.
# 
# ---
# 
# ## What We Did
# 
# We began with a large and detailed dataset, rich in lifestyle, demographic, and health-related features.
# After careful preprocessing — including encoding variables, restructuring age groups, filtering out noise, and balancing classes — we built a clean, consistent foundation for analysis.
# 
# Our exploratory data analysis revealed key trends:
# - Heart disease risk rises with age, poor general health, diabetes, and stroke history.
# - Lifestyle choices like physical activity and smoking play a measurable role.
# - Variables like sleep duration and race had little predictive value, and were removed.
# 
# We then trained and evaluated multiple machine learning models — from interpretable logistic regression to complex neural networks and gradient boosting.
# We tested both imbalanced and SMOTE-balanced datasets to understand how sensitivity to minority classes changes across models.
# 
# ---
# 
# ## What We Learned
# 
# Some key insights emerged:
# 
# - **XGBoost** offered the best balance of recall and precision for detecting heart disease.
# - **Logistic Regression** was best when the priority is catching as many positive cases as possible.
# - **Neural Networks** were sensitive to data balancing, performing well with SMOTE but poorly without.
# - **Random Forest**, while highly accurate overall, was less effective in identifying high-risk individuals.
# 
# We also found that **no model could overcome the limitations of the data** entirely.
# Heart disease prediction is fundamentally difficult — not just due to class imbalance, but because real-world risk is shaped by genetics, environment, and subtle interactions not captured in survey data.
# 
# ---
# 
# ## Beyond the Metrics: What SHAP Taught Us
# 
# Using SHAP analysis, we moved beyond performance metrics and into the **reasoning** behind model decisions.
# 
# - Globally, **age** and **general health** were the strongest drivers of prediction.
# - On the individual level, SHAP revealed how combinations of features — such as smoking, diabetes, or asthma — tipped the model's decision one way or the other.
# - We saw where the model succeeded… and where it failed — especially in edge cases with mixed signals.
# 
# These insights help us not only build better models, but also **trust** them — a critical requirement in medical applications.
# 
# ---
# 
# ## Final Thoughts: What This Means for David — and for Us
# 
# If David had access to a model like ours, would it have caught his risk in time?
# 
# Maybe.
# If he had reported poor general health, a stroke history, or difficulty walking, the model likely would have raised a flag.
# But if he seemed relatively healthy on the surface — as many people do — it might have missed him entirely.
# 
# That’s the core tension in predictive health:
# **We can build good models, but no model is perfect.**
# 
# Still, every false positive we accept may be a real life we save.
# In medicine, it's often better to **flag too many** than to miss the one.
# 
# ---
# 
# ## Looking Forward
# 
# This project taught us a lot — about data, modeling, and most importantly, about the **limitations and potential** of machine learning in healthcare.
# 
# To improve further, future work might include:
# - Richer datasets with genetic, environmental, or clinical markers
# - Temporal data (changes over time)
# - Collaboration with medical professionals for guided feature engineering
# - Ensemble or hybrid models that blend interpretability with performance
# 
# ---
# 
# ## Closing Note
# 
# While no model is perfect, every step toward better prediction brings us closer to personalized, preventive care — where data and medicine work hand in hand.
# 