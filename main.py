#%% md
# # Intro
# As computer science students with a deep interest in data science, we, Hila Giladi (312557606) and Kfir Shuster (315695122), have chosen to focus on heart disease prediction and analysis. We are committed to leveraging data science to better understand risk factors and potentially help improve early detection and prevention strategies. Through careful analysis of various health metrics and lifestyle factors, we aim to contribute to the broader understanding of heart disease risk factors and their complex interactions.
# 
# # The Problem
# Our analysis of the data reveals a complex picture of the challenges in understanding heart disease factors. While approximately 8.6% of the population in our dataset suffers from heart disease, our analysis uncovers an intricate web of relationships between various risk factors, ranging from demographic factors such as age and gender, through lifestyle habits like smoking and physical activity, to existing medical conditions such as diabetes and stroke. This complexity emphasizes the need for a comprehensive and multidimensional approach to understanding heart disease risk.
# 
# # The Importance of the Solution
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
# # Data Collection and Selection Process
# We utilized a comprehensive dataset containing 319,795 records with 18 different variables, covering a wide range of metrics:
# - Physiological measures: BMI, physical health
# - Lifestyle habits: smoking, alcohol consumption, physical activity, sleep hours
# - Health conditions: diabetes, stroke, asthma, kidney disease
# - Demographic characteristics: age, sex, race
# 
# This dataset was selected for its extensive scope and rich variety of variables, allowing us to examine the complex relationships between various factors and heart disease. The data provides a solid foundation for in-depth analysis and the development of predictive models that may help improve our understanding of heart disease risk factors and enhance early detection capabilities.
# 
# # Data Analysis
# Let's transform our collected data into a structured pandas data frame to examine the information we've gathered. This will give us a clear view of our dataset's contents and help us understand what we're working with.
#%%
import pandas as pd

df = pd.read_csv('heart_2020_cleaned.csv')
display(df)
#%% md
# # Data Processing Overview
# Before we can analyze data and build models, preprocessing of the data is necessary. This stage is essential for ensuring the quality of analysis, as it transforms raw data into a consistent and structured format. Data processing includes converting categorical variables to numerical ones, reorganizing age categories, and handling outliers. At this stage, we will also clean and filter the data, handle missing values, and prepare the data for in-depth exploratory analysis.
# 
#%% md
# # Data Preprocessing:
# ## Converting Categorical Variables
# 1. Convert binary variables from Yes/No to 1/0:
#    - Applied to health conditions and behaviors like HeartDisease, Smoking, AlcoholDrinking etc.
#    - Special handling for KidneyDisease to ensure integer type conversion
# 
# 2. Restructure age categories into broader 10-year groups:
#    - Original categories (e.g., "18-24", "25-29") consolidated into decade spans
#    - New categories range from "18-29" to "80+"
#    - Includes error handling for unknown categories
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
#%% md
# ## Data Cleaning: Filter Sleep Duration Values
# Removed unrealistic sleep duration values by limiting the range to 1-16 hours per day, eliminating extreme outliers that likely represent data entry errors or measurement mistakes.
#%%
df = df[(df['SleepTime'] >= 1) & (df['SleepTime'] <= 16)]
#%% md
# # Summary of Data Processing
# We have successfully completed the initial data processing phase. We converted binary variables (Yes/No) to numbers (1/0), grouped age categories into more meaningful decade groups (18-29, 30-39, etc.), and removed unrealistic sleep duration values (limiting to 1-16 hours per day). These preprocessing steps ensure that our data is now in an appropriate format for further analysis and visualization. The cleaned dataset maintains the essential information while eliminating potential sources of error from extreme outliers or inconsistent formatting.
#%% md
# # Exploratory Data Analysis (EDA) Overview
# Exploratory data analysis is a cornerstone of any data science project, allowing us to understand the central characteristics and patterns in our data. In this stage, we will explore the distribution of heart disease in the population and examine the relationships between heart disease and various factors such as demographic data (age, sex), lifestyle habits (smoking, alcohol consumption, physical activity), and existing health conditions (diabetes, stroke, physical and mental health). Through visualizations and statistical analyses, we will uncover insights that will guide our variable selection and model choices, and provide a deeper understanding of risk factors for heart disease.
#%% md
# # EDA
# ## Basic histogram of the prediction column
# ### Distribution of heart disease cases in our population
# Create a basic histogram to display the distribution of heart disease cases in our population. The purpose of this graph is to provide a clear picture of the ratio between people with and without heart disease in our dataset.
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
# The graph reveals a significant class imbalance where 91.4% (approximately 280,000 individuals) do not have heart disease, while only 8.6% (around 25,000 individuals) have heart disease.
#%% md
# ### Distribution of Heart Disease by Race
# Create a bar plot to show the distribution of heart disease cases across different racial groups, allowing us to identify any potential patterns or disparities in heart disease prevalence among different ethnic groups
#%%
sns.countplot(data=df, x='Race', hue='HeartDisease', palette='YlOrBr')
plt.xlabel('Race')
plt.ylabel('Frequency')
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.show()
#%% md
# 
# The graph shows that White individuals make up the largest portion of the dataset with approximately 220,000 cases, while other racial groups have significantly smaller representations. All racial groups maintain a similar proportion of heart disease cases relative to their population size.
# This visualization, while providing demographic information, doesn't offer much predictive value for our heart disease analysis. The similar proportions of heart disease cases across all racial groups suggest that race alone is not a strong differentiating factor for heart disease risk in our dataset. This observation is further supported by the correlation matrix we examined earlier, which showed race had a weak negative correlation (-0.04) with heart disease.
# Due to this low informational value and potential for introducing bias without adding predictive power, we'll remove this feature from our dataset now to ensure it doesn't influence our subsequent analyses and modeling.
#%%
# Remove Race column from dataset as it doesn't provide significant predictive value
df = df.drop('Race', axis=1)
print(f"Race column removed. Dataset now has {df.shape[1]} columns.")
display(df)
#%% md
# ### Distribution of Heart Disease by Sex
# Create a bar plot comparing the distribution of heart disease between males and females to understand if there are any gender-based differences in heart disease prevalence.
#%%
sns.countplot(data=df, x='Sex', hue='HeartDisease', palette='YlOrBr')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()
#%% md
# The visualization shows that females have a slightly higher total frequency (approximately 155,000 cases) compared to males (approximately 135,000 cases). Both genders display a similar pattern of heart disease distribution, with the majority not having heart disease.
#%% md
# ### Distribution of Heart Disease by Age Category
# Create a bar plot to examine how heart disease prevalence varies across different age groups, using ordered age categories to show the progression of risk with age.
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
# The 60-69 age group has the highest frequency with approximately 60,000 cases, followed by the 50-59 age group. The incidence of heart disease (represented by lighter brown) increases notably with age, showing higher proportions in older age groups (60-69, 70-79, and 80+).
#%% md
# ## Correlation Matrix of Health Variables
# Create a correlation matrix heatmap to visualize the relationships between various health-related variables, helping identify which factors are most strongly associated with heart disease.
# 
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
# This correlation matrix shows the relationships between various health-related variables, with stronger correlations indicated by darker colors and numerical values ranging from -1 to 1. The most notable correlations with heart disease (HeartDisease) include: age category (0.23), stroke (0.20), difficulty walking (0.20), physical health (0.17), and diabetic status (0.18), while showing negative correlations with general health (-0.24) and physical activity (-0.10). There's a particularly strong negative correlation (-0.48) between physical health and general health, suggesting that poor physical health significantly impacts overall health perception. Interestingly, factors like BMI (0.05) and alcohol drinking (-0.03) (-0.04) show relatively weak correlations with heart disease.
#%% md
# ## Investigating Multicollinearity Using VIF Analysis
# To ensure the reliability of our predictive models, we conducted a Variance Inflation Factor (VIF) analysis on key health variables. The VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity with other predictors.
# Our analysis focused on General Health status and Physical Health, which showed moderate correlation in our earlier correlation matrix analysis (-0.48).
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
# The results indicate low multicollinearity between these variables, with VIF values of approximately 1.30 for both features. In statistical practice, VIF values below 5 (or even 2.5 in more conservative approaches) are generally considered acceptable. Our values are well below these thresholds, suggesting that each variable contributes unique information to our model.
# When we removed GenHealth_numeric from the analysis, the VIF for PhysicalHealth predictably dropped to 1.0, confirming the absence of multicollinearity when only one predictor remains (apart from the constant).
# This analysis supports our decision to retain both variables in our predictive models, as they each provide distinct information about patient health status that could be valuable for heart disease prediction, despite their moderate negative correlation.
#%% md
# ## Statistical Significance Testing: Comparing Physical Activity and Difficulty Walking
#%% md
# Let's run T-test on the columns PhysicalActivity, DiffWalking for Equal Means:
# Both of the columns are very similar in their meaning and has strong correlations.
# This test can help you understand if the difference between the two columns is statistically significant. If the distributions are very similar, it might indicate that one feature can be removed without losing much predictive power.
#%%
from scipy.stats import ttest_ind

t_stat, p_value_ttest = ttest_ind(df['PhysicalActivity'], df['DiffWalking'])

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value of T-test: {p_value_ttest:.3f}")
#%% md
# The T-test results show a very large T-statistic of 664.097 and a p-value of 0.000, indicating a statistically significant difference between the means of PhysicalActivity and DiffWalking. This suggests that the two variables provide distinct information and are not redundant. Therefore, it may not be appropriate to drop either column, as both could be valuable for predictive modeling, representing different aspects of the data.
#%% md
# ## Heart Disease Distribution by Lifestyle Habits
# Create a bar plot showing the percentage of heart disease cases across different combinations of smoking and alcohol consumption habits.
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
# Smokers who don't drink have the highest prevalence of heart disease (12.8%), while non-smokers who drink show the lowest (3.0%). This aligns with the correlation matrix which showed smoking had a positive correlation with heart disease (0.11), indicating increased risk. Interestingly, alcohol drinking showed a very weak negative correlation (-0.03), explaining why categories involving drinking (non-smoker drinker at 3.0% and smoker drinker at 6.6%) show lower heart disease percentages than their non-drinking counterparts.
#%% md
# ## Distribution of Heart Disease by Physical Activity and Health Status
# Create a bar plot comparing heart disease rates across different combinations of physical activity levels and general health status.
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
# The data shows a progression from 2.1% (active, excellent health) to 35.5% (inactive, poor health) in heart disease risk. This pattern aligns with the correlation matrix, which showed negative correlations between heart disease and both physical activity (-0.10) and general health (-0.24), indicating that better health status and physical activity are associated with lower heart disease risk. The stronger correlation with general health is reflected in the more dramatic increases in heart disease rates as health status declines
#%% md
# ## Distribution of Heart Disease by Physical and Mental Health Days
# Create a bar plot showing how heart disease prevalence varies with different combinations of physical and mental health issue days.
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
# Heart disease rates increase from 4.0% (perfect physical health, minimal mental health issues) to 24.4% (over 15 days of both issues). The correlation matrix supports these patterns, showing physical health had a stronger correlation (0.17) with heart disease compared to mental health (0.03). This explains why we see larger jumps in heart disease rates when physical health days increase versus changes in mental health days
#%% md
# ## Heart Disease by Diabetes and Stroke Status
# 
# Create a bar plot comparing heart disease rates across different combinations of diabetes and stroke conditions.
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
# The progression from 5.8% (neither condition) to 48.1% (both conditions) aligns with the correlation matrix, where both diabetes (0.18) and stroke (0.20) showed positive correlations with heart disease. The slightly higher correlation for stroke is reflected in its stronger individual impact (31.1% vs 19.3% for diabetes alone).
#%% md
# ## BMI Distribution by Heart Disease Status
# 
# Create a density plot comparing BMI distributions between people with and without heart disease.
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
# The KDE plot of BMI distribution by heart disease status shows a similar right-skewed pattern for both groups, with individuals having heart disease slightly skewed towards a higher BMI. While this suggests a potential link between BMI and heart disease, the significant overlap between distributions indicates that BMI alone may not be a strong predictor. Further statistical analysis, such as an ANOVA test, can help determine if the difference in BMI between those with and without heart disease is statistically significant.
#%% md
# ## ANOVA Testing: Comparing BMI Distribution Between Heart Disease Groups
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
# The ANOVA test indicates a statistically significant difference in BMI between individuals with and without heart disease (F = 860.4963, p < 0.0001). This suggests that BMI is a relevant factor in heart disease prediction
#%% md
# ## Kidney Disease Distribution by Heart Disease Status
# 
# Create a density plot showing the distribution of kidney disease among individuals with and without heart disease.
#%%
sns.kdeplot(data=df, x='KidneyDisease', hue='HeartDisease', fill=True, common_norm=False)

plt.title('KidneyDisease Distribution by Heart Disease Status')
plt.xlabel('KidneyDisease')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
#%% md
# The correlation matrix showed a positive correlation (0.15) between kidney disease and heart disease, which is reflected in the relatively higher density at 1 for the heart disease group, though the overall prevalence remains low in both groups.
#%% md
# ## Sleep Time Distribution by Heart Disease Status
# 
# Create a density plot comparing sleep patterns between those with and without heart disease.
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
# This density plot shows the distribution of sleep hours comparing individuals with and without heart disease, with a red dashed line indicating the recommended 8 hours of sleep. The distribution is multimodal (showing multiple peaks) with the highest concentration around 7-8 hours of sleep for both groups. People without heart disease (shown in blue) have slightly higher peaks at the recommended sleep duration, while those with heart disease (shown in grey) show a more spread out distribution with lower peaks. However, the overall similar patterns between both groups align with the very weak correlation (0.01) found in the correlation matrix between sleep time and heart disease. The plot suggests that sleep duration alone may not be a strong predictor of heart disease risk, though there's a slight tendency for people without heart disease to maintain more regular sleep patterns closer to the recommended 8 hours.
#%% md
# ## BMI vs Sleep Time by Heart Disease Status
# 
# Create two separate density plots comparing the BMI-sleep relationship for people with and without heart disease.
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
# These two density plots compare the relationship between BMI and sleep time for people with heart disease (left) and without heart disease (right). In both groups, there are similar patterns of horizontal bands showing common sleep durations between 5-10 hours. However, there are some notable differences: people with heart disease (left plot) show a slightly higher concentration in the higher BMI ranges (30-40), indicated by the brighter red areas, while those without heart disease (right plot) have their highest concentrations in the lower BMI ranges (20-30). Both plots show multiple bands of sleep duration, suggesting that sleep patterns are similar regardless of BMI or heart disease status, which aligns with the weak correlations we saw earlier in the correlation matrix (BMI with heart disease: 0.05, sleep time with heart disease: 0.01).
#%% md
# ## Cohen's d Effect Size Analysis: Sleep Time and Heart Disease
# Let's assess whether sleep duration is essential for prediction using Cohen’s d. This metric quantifies the effect size by measuring the difference between two group means in terms of standard deviation, offering insight into the practical significance of the variation.
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
# The Cohen’s d value of -0.0202 indicates a very small effect size, meaning the difference in sleep duration between individuals with and without heart disease is minimal and likely not practically significant. This suggests that while there may be a statistical difference, sleep duration alone is not a strong differentiator for heart disease risk.
# Given this extremely small effect size and the negligible correlation we observed earlier (0.01), we'll remove the SleepTime variable from our dataset to simplify our model and focus on more influential predictors of heart disease.
# 
#%%
df = df.drop('SleepTime', axis=1)
print(f"SleepTime column removed. Dataset now has {df.shape[1]} columns.")
#%% md
# ## BMI vs Physical Health by Heart Disease Status
# 
# Create density plots comparing the relationship between BMI and physical health days for those with and without heart disease.
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
# These density plots compare BMI and Physical Health days between individuals with and without heart disease. The left plot (Heart Disease: 0) shows a concentrated cluster at lower BMI (20-30) and fewer physical health issue days, indicating healthier patterns. The right plot (Heart Disease: 1) reveals two distinct concentrations: one with low physical health days and normal BMI, and another showing higher physical health days, suggesting more health challenges in people with heart disease. These patterns align with the correlation matrix, which showed physical health had a moderate correlation with heart disease (0.17), while BMI showed a weaker correlation (0.05), highlighting that physical health issues may be a better indicator of heart disease risk than BMI alone.
#%% md
# ## Physical Health vs Mental Health by Heart Disease Status
# 
# 
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
# The right plot (without heart disease) shows a strong concentration near zero for both health metrics, indicating most healthy individuals experience few poor health days. The left plot (with heart disease) shows more dispersion with multiple hotspots, including significant densities at higher numbers of both physical and mental health days. This pattern aligns with the correlation matrix where physical health showed stronger correlation with heart disease (0.17) than mental health (0.03), though the visualization suggests these factors often occur together in heart disease patients.
# To further assess potential multicollinearity and ensure both features contribute independently to the model, we will use the Variance Inflation Factor (VIF), which confirms that these variables do not exhibit significant collinearity.
#%% md
# ## Multicollinearity Analysis: Variance Inflation Factor (VIF)
# The following code calculates the Variance Inflation Factor (VIF) for our numerical health metrics to assess potential multicollinearity between these variables. VIF helps identify how much the variance of an estimated regression coefficient increases if predictors are correlated, with higher values indicating stronger multicollinearity issues.
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
# The VIF analysis for PhysicalHealth and MentalHealth resulted in low values (1.20 for both), indicating no significant multicollinearity between them. This suggests that both features contribute independently to the model and are not redundant. Since they provide unique information, there is no need to remove either feature, and both can be retained for further analysis and model training.
#%% md
# # 3D Relationship between Physical Health, Mental Health, BMI, and Heart Disease
# Create a 3D scatter plot to visualize the complex relationship between physical health, mental health, BMI, and heart disease status, using different colors to distinguish between cases.
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
# The 3D plot reveals that heart disease cases (red points) are more prevalent among individuals with higher physical health issues, while showing a weaker association with mental health metrics. Points form vertical clustering patterns due to discrete day-count values in the original data. The visualization confirms the stronger correlation between physical health and heart disease (0.17) compared to mental health (0.03). While heart disease cases appear across various BMI values, they show some concentration in higher ranges, demonstrating the complex, non-linear interactions between these health factors.
#%% md
# # BMI Distribution by General Health and Heart Disease Status
# Create a boxplot showing BMI distributions across different general health categories, split by heart disease status.
# 
# 
#%%
plt.figure(figsize=(15, 8))
sns.boxplot(x="GenHealth", y="BMI", hue="HeartDisease", data=df)
plt.title('BMI Distribution by General Health and Heart Disease Status')
plt.show()
#%% md
# Across all general health categories (Excellent, Very good, Good, Fair, Poor), there's a consistent pattern where individuals with better general health ratings tend to have lower BMI values. The medians and distributions of BMI gradually increase as general health ratings decline, with Fair and Poor health categories showing the highest median BMIs and widest ranges. When comparing those with and without heart disease, people with heart disease consistently show slightly higher BMI medians within each health category, though this difference is most pronounced in the Fair and Poor health categories. All health categories display outliers at high BMI values, with particularly extreme outliers (reaching BMI values of 80+) appearing in the Fair and Poor health categories. However, the relatively similar distributions between heart disease and non-heart disease groups across health categories aligns with the weak correlation (0.05) between BMI and heart disease found in the correlation matrix, suggesting that while BMI has some relationship with both general health and heart disease, it's not a strongly determining factor on its own.
#%% md
# # Physical Health Distribution by Asthma and Heart Disease Status
# Create a violin plot showing physical health distributions across asthma status, split by heart disease presence.
#%%
sns.set_theme(style="ticks", palette="pastel")
plt.figure(figsize=(15, 8))
sns.violinplot(x="Asthma", y="PhysicalHealth", hue="HeartDisease", data=df)
plt.title('Distribution of Physical Health Issues across Asthma and Heart Disease Status')
plt.xlabel('Asthma Status')
plt.ylabel('Physical Health Issues (days)')
plt.show() #todo - change text
#%% md
# The violin plot demonstrates the complex relationship between physical health issues, asthma, and heart disease. The plot shows that individuals with heart disease (shown in brown) consistently experience more days of poor physical health compared to those without heart disease (blue), regardless of their asthma status. Among those with heart disease, the distribution is wider and shows a higher concentration of days with physical health problems. When looking at asthma's impact, asthmatic individuals display more symmetrical and concentrated distributions of physical health issues, while non-asthmatics show more spread in their patterns. The combined presence of both conditions appears to have a compounding effect, with individuals having both heart disease and asthma showing the highest concentration of physical health issues, while those with neither condition report the fewest days of poor physical health. These patterns align with the correlation matrix findings, where heart disease showed a stronger correlation with physical health (0.17) compared to asthma's weaker correlation (0.04), indicating that heart disease has a more significant impact on physical health than asthma.
#%% md
# # Relationship between Age and BMI for Asthma Patients by Heart Disease Status
# Create a contour plot with marginal distributions to visualize how age and BMI relate specifically for asthma patients, with separate distributions for those with and without heart disease.
# 
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
# This contour plot with marginal distributions illustrates the relationship between age and BMI specifically for asthma patients, separated by heart disease status. The central plot shows density contours (blue for no heart disease, orange for heart disease) while the top and right margins display the distributions of age and BMI respectively. For asthma patients without heart disease (blue contours), the distribution is concentrated between ages 20-70 with BMI ranging from 20-40, showing highest density around ages 30-50 and BMI 25-35. Those with both asthma and heart disease (orange contours) tend to be older, with distributions shifted towards the 50-80 age range, while maintaining similar BMI ranges, though slightly skewing higher. The marginal distributions clearly demonstrate that among asthma patients, those with heart disease tend to be older, with a subtle trend toward higher BMI values. This aligns with the correlation matrix findings, which showed weak correlations between heart disease and both BMI (0.05) and asthma (0.04), suggesting these relationships exist but aren't strongly predictive.
#%% md
# # PCA (Principal Component Analysis)
# 
# 
#%% md
# PCA is applied to reduce the dimensionality of the health metrics (BMI, PhysicalHealth, MentalHealth, GenHealth) while retaining the most significant variance. This helps identify underlying patterns in the data and determine whether these metrics effectively differentiate individuals based on sex and age. By transforming the data into principal components, we can evaluate the importance of each feature and assess whether sex and age plays a meaningful role in the overall variance of health-related factors.
#%% md
# ## Focused PCA with Health Metrics Colored by Sex Category
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
# ## Focused PCA with Health Metrics Colored by Age Category
# Finally, we'll examine the same health metrics (BMI, PhysicalHealth, MentalHealth, GenHealth) but now coloring by age category to understand how these health relationships vary across different age groups.
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
# Our exploratory analysis revealed significant insights regarding heart disease and its risk factors. We identified a clear imbalance in the data, with only 8.6% of samples classified as having heart disease, which requires special consideration in the model-building stage. We found important correlations with heart disease: age (0.23), stroke (0.20), difficulty walking (0.20), diabetic status (0.18), and physical health (0.17) showed the strongest positive correlations, while general health status (-0.24) showed a strong negative correlation.
# 
# Our visualizations highlighted interesting patterns: smokers who don't drink exhibit the highest rate of heart disease (12.8%); inactive people with poor general health show a heart disease rate of 35.5% compared to just 2.1% in active people with excellent health; and the combination of diabetes and stroke leads to a heart disease rate of 48.1%.
# 
# During the EDA process, we made important feature selection decisions based on our findings. We removed the Race variable as it showed minimal predictive value (correlation of -0.04) and could potentially introduce bias. Similarly, SleepTime was removed after Cohen's d analysis revealed a very small effect size (-0.0202), indicating that sleep duration alone is not a strong differentiator for heart disease risk.
# 
# PCA analysis and other statistical tests confirmed that our remaining variables contribute unique information to the model without significant multicollinearity, providing a solid foundation for our modeling phase.
#%% md
# # Model Building and Training Overview
# After gaining a deep understanding of our data, we are ready to build models that will attempt to predict the risk of heart disease. In this stage, we will first address the challenge of data imbalance using the SMOTE technique, which will create synthetic samples of heart disease cases to balance the training set. We will then develop a variety of models - from relatively simple models like logistic regression, through decision tree-based models like Random Forest, to complex neural networks. Additionally, we will explore natural patterns in the data using a hierarchical clustering algorithm. Comparing different models will allow us to understand the advantages and disadvantages of each approach and choose the most appropriate model for our needs.
#%% md
# # Data Balancing using SMOTE
# As we observed in our EDA, there is a significant class imbalance in our dataset (91.4% without heart disease vs 8.6% with heart disease). This imbalance can affect model performance, particularly the ability to detect the minority class (heart disease). We'll use SMOTE (Synthetic Minority Over-sampling Technique) to balance our dataset.
# 
#%% md
# 
# ## Arrange the data to input and target and train and set
# In this step, we're preparing our heart disease dataset by converting categorical variables to numeric format.
# Binary variables like 'Smoking' are mapped directly (Yes=1, No=0), while multi-category variables
# are transformed using LabelEncoder. Then, we split our features (X) and target (HeartDisease),
# and divide the data into training (80%) and testing (20%) sets.
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
# ## Apply SMOTE to Training Data
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
# The SMOTE balancing transformed your highly imbalanced dataset from a 10:1 ratio (234,055 negative vs 21,781 positive cases) to a perfectly balanced 1:1 distribution with 234,055 samples in each class. This balanced dataset should help your model better detect heart disease cases without bias toward the majority class, as SMOTE created synthetic samples rather than simply duplicating existing minority examples.
#%% md
# ## Scale Features
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
# The data preparation is now complete and ready for model training. Our balanced training dataset
# consists of 468,110 samples with 17 features after SMOTE application. The balanced target variable
# matches this with 468,110 labels. Our test dataset contains 63,959 samples with the same 17 features,
# which remained unbalanced to represent real-world data distribution. This preprocessing pipeline
# ensures our model will train on balanced data while being evaluated on realistic data conditions.
#%% md
# Now we'll train and evaluate models using both datasets,
# always testing on the original imbalanced test data to reflect real-world conditions
#%% md
# # Model Building and Algorithm Selection
# 
# In this phase, we implemented a variety of machine learning algorithms from different families to predict heart disease risk. We strategically selected diverse algorithms that employ fundamentally different mathematical techniques. This approach allows us to:
# 
# 1. Assess how different learning paradigms perform on this specific healthcare problem
# 2. Identify which modeling approach offers the best balance of precision and recall for heart disease detection
# 3. Understand the trade-offs between interpretability and predictive power
# 
# We selected the following algorithms, each representing a different modeling family:
# 
# - **Logistic Regression**: A linear model that serves as our baseline and offers high interpretability
# - **Random Forest**: An ensemble of decision trees that captures non-linear relationships and feature interactions
# - **Neural Network (MLP)**: A deep learning approach that can model complex patterns without explicit feature engineering
# - **XGBoost**: A gradient boosting framework that excels at structured data and usually achieves state-of-the-art performance on tabular datasets
# - **Hierarchical Clustering**: An unsupervised learning approach to identify natural groupings in our dataset without using the target variable
# 
# Our presentation follows a logical progression from simpler to more complex algorithms, reflecting the historical development of machine learning techniques while also moving from highly interpretable models to more "black box" approaches. This pedagogical structure allows us to examine how increasing model complexity affects predictive performance, particularly for detecting the minority class (heart disease cases).
# 
# For each supervised algorithm, we trained and evaluated two versions: one using the original imbalanced dataset and another using SMOTE-balanced data. This allowed us to assess how class balancing affects each model's ability to identify heart disease cases, which is critical in healthcare applications where failing to detect a positive case could have serious consequences.
#%% md
# # Logistic Regression
# Implement a logistic regression model as our baseline classifier for predicting heart disease, as it's effective for binary classification problems.
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
# The classification report reveals significant class imbalance issues in our model's performance. While achieving a misleadingly high overall accuracy of 0.91, the model shows a stark contrast in its predictive capabilities: it excels at identifying non-heart disease cases (0.92 precision, 0.99 recall) but performs poorly in detecting heart disease cases (0.51 precision, 0.09 recall). This mirrors the imbalance we saw in our initial data exploration, where only 8.6% of cases had heart disease, making the model biased towards predicting the majority class.
#%% md
# # Random Forest
# Implement a Random Forest classifier to capture complex interactions between health variables and provide feature importance rankings.
# 
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
# The Random Forest classifier demonstrates strong performance metrics with an overall accuracy of 0.92. For heart disease detection, it achieves a precision of 0.61 and recall of 0.15. The feature importance analysis validates our earlier correlation findings, with general health status, age, and physical health emerging as the most significant predictors. This aligns with medical understanding of heart disease risk factors.
# 
#%% md
# # Neural Network
#%% md
# To further diversify our algorithmic approaches, we implemented a Multi-Layer Perceptron (MLP) neural network. Neural networks can capture complex non-linear relationships between variables, making them potentially valuable for heart disease prediction where risk factors may interact in intricate ways. Our model architecture consists of an input layer corresponding to our health metrics, two hidden layers (100 and 50 neurons respectively) for learning complex patterns, and a single output neuron for binary classification. Below we visualize this architecture before examining the model's performance on both balanced and imbalanced datasets.
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
# The MLP neural network exhibited unique behavior compared to other models, with dramatic changes before and after implementing SMOTE. The original model suffered from extreme imbalance, with a low sensitivity of only 6% for heart disease cases despite an overall accuracy of 91%, indicating significant bias toward the majority class. After SMOTE, a complete reversal occurred - sensitivity soared to 76%, while precision dropped to 22%, with a significant improvement in the F1 score (from 0.11 to 0.34).
# 
# Compared to other models in the study, the neural network demonstrates heightened sensitivity to data balancing techniques, a characteristic that distinguishes this algorithm family. While models like Random Forest showed only moderate improvement after SMOTE, the neural network responded much more dramatically. This phenomenon stems from the unique architecture of neural networks, allowing them to learn complex patterns and relationships between variables. With balanced data, the network better learned heart disease characteristics, though with a tendency toward overgeneralization. In the medical context of heart disease detection, the SMOTE-enhanced model may be preferred using a "better safe than sorry" approach, as it identifies more potential cases, even at the cost of false alarms. The flexibility and ability of neural networks to learn complex, non-linear patterns make them a valuable tool in health data analysis, but require careful tuning of hyperparameters and data balancing techniques.
#%% md
# # XGBoost
# Implement an XGBoost classifier which uses gradient boosting with decision trees. This advanced algorithm combines multiple weak prediction models to create a stronger predictive model, with regularization techniques to prevent overfitting. XGBoost is known for its performance, speed, and ability to handle complex patterns.
# 
# 
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
# The XGBoost model demonstrated a pattern similar to other models, with significant improvement in heart disease detection after balancing the data using SMOTE. Before SMOTE, the model achieved high overall accuracy (0.91) but struggled to identify heart disease cases with very low recall of 0.08, though precision for positive cases was relatively good (0.57). After data balancing with SMOTE, there was a substantial change in performance: the recall for heart disease detection dramatically increased to 0.61 (a 7.6-fold improvement), but at the cost of reduced precision to 0.22.
# 
# These results highlight XGBoost's strong ability to learn from balanced data, as it shows a better balance between precision and recall compared to Random Forest, which achieved lower recall (0.23) but higher precision (0.30). XGBoost demonstrates better recall than Random Forest but lower than the high recall of the Logistic Regression model (0.77), positioning it as a middle-ground solution in terms of heart disease detection capability.
# 
# The F1-score of 0.33 (after SMOTE) indicates a significant improvement compared to 0.14 (before SMOTE), demonstrating the importance of addressing data imbalance. In summary, XGBoost offers an advanced boosting approach that provides a good balance between identifying heart disease cases and minimizing false alarms, making it a valuable addition to the range of models in the project.
#%% md
# # Hierarchical Clustering
# - Perform hierarchical clustering to identify natural groupings in our health data, using Gower distance to handle mixed numeric and categorical variables.
# - Creates 5 distinct clusters of patients
# - Provides cluster sizes and characteristics
# - Shows mean/mode values of features in each cluster
# - Helps identify natural groupings of patients with similar health profiles
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

# Calculate distance matrix (using euclidean distance for symmetric matrix)
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

# Calculate distance matrix (using euclidean distance for symmetric matrix)
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
# Hierarchical clustering analysis of heart disease data revealed distinct patient groups based on health characteristics. Before SMOTE balancing, clusters showed moderate differences in disease rates (6.5%-42.1%), with one cluster identifying higher-risk patients. After SMOTE, disease prevalence increased dramatically across all clusters (42.5%-100%), with one cluster containing only heart disease cases.
# The bottom graphs display heart disease rates in each cluster - the left graph shows the original data with lower rates reflecting the natural imbalance in the data, while the right graph presents the disease distribution after SMOTE with significantly higher values highlighting how the balancing technique enables better identification of risk groups.
# The dendrograms illustrate the hierarchical cluster structure and demonstrate how data balancing affects natural grouping. This unsupervised analysis complements our supervised models by identifying natural relationships between features, allowing more targeted approaches to heart disease prevention.
#%% md
# # Comparing the Models
# After training multiple models from different algorithm families, we observed distinct patterns in their performance characteristics. The figure below compares the precision, recall, and F1-score for heart disease detection across all models after SMOTE balancing.
# 
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame with model performance metrics after SMOTE
metrics_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'Neural Network', 'XGBoost'],
    'Accuracy': [0.70, 0.82, None, 0.73],
    'Precision (No HD)': [0.97, 0.93, None, 0.97],
    'Recall (No HD)': [0.70, 0.87, None, 0.73],
    'F1 (No HD)': [0.81, 0.90, None, 0.83],
    'Precision (HD)': [0.19, 0.20, None, 0.20],
    'Recall (HD)': [0.73, 0.35, None, 0.73],
    'F1 (HD)': [0.30, 0.26, None, 0.32]
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
# # Model Summary and Analysis
# 
# When comparing the models' performance after SMOTE balancing, we observed distinct patterns in how these diverse algorithm families approach the heart disease prediction task.
# 
# **__Random Forest__** emerged as the best performer in terms of overall accuracy (**0.90**), but its precision for heart disease cases dropped to **0.23**, making it less cautious in labeling patients as having heart disease than initially thought. While it still maintains a relatively high F1-score (**0.27**), it no longer leads in balancing precision and recall.
# 
# **__Logistic Regression__** demonstrated a recall of **0.70** for heart disease cases, making it effective at identifying positive cases, though with a low precision of **0.18**. This trade-off resulted in an F1-score of **0.29**, showing that it prioritizes detecting heart disease cases at the cost of more false positives.
# 
# **__Neural Network__** saw a significant drop in recall (**0.58**, down from 0.76 in previous results), reducing its effectiveness in identifying heart disease cases. While it maintains a relatively high precision (**0.20**) compared to **__Logistic Regression__**, its F1-score (**0.30**) suggests it is no longer as competitive in recall-driven scenarios.
# 
# **__XGBoost__** provided the best balance between precision (**0.22**) and recall (**0.61**), resulting in an F1-score (**0.33**) that closely matches the top performers. This makes it a strong compromise model when balancing sensitivity and specificity is crucial.
# 
# Despite applying SMOTE to address class imbalance, all models still showed significantly better performance for non-heart disease cases (**F1 scores between 0.81-0.92**) compared to heart disease cases (**F1 scores between 0.27-0.33**). While SMOTE improved recall for heart disease cases, the challenge of accurately detecting them remains.
# 
# Our analysis reveals that these diverse modeling approaches capture different aspects of the heart disease prediction problem. The linear approach (**__Logistic Regression__**) remains useful for maximizing recall, while **__Neural Network__** is now less competitive. Tree-based methods (**__Random Forest__** and **__XGBoost__**) demonstrate varying trade-offs between precision and recall, with **__XGBoost__** emerging as the most balanced option.
# 
# Ultimately, the choice between these models depends on clinical priorities:
# - **If maximizing recall is most important** → **__Logistic Regression__** (0.70 recall) is preferable.
# - **If balancing precision and recall is key** → **__XGBoost__** (F1-score 0.33) is the best choice.
# - **If prioritizing high accuracy and minimizing false positives** → **__Random Forest__** is a safer option.
#%% md
# # Interpretability of the Models
# 
#%% md
# After testing and comparing different classification models for heart disease prediction, we want to understand which variables most significantly influenced our models' behavior. To gain deeper insights, we'll use SHAP (SHapley Additive exPlanations) values, which provide a sophisticated way to interpret machine learning models.
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
# The plot above sorts features by the sum of SHAP value magnitudes over all samples, and uses these values to show the distribution of impacts each feature has on the model output. Colors represent the feature value (red high, blue low). It clearly shows that older age (shown in red) significantly increases the likelihood of heart disease, while younger age (shown in blue) decreases it. General Health is the second most important feature, where poor health status (red) increases the risk of heart disease. Interestingly, features like physical activity and mental health show more moderate effects on the model's output.
#%% md
# ## Enhancing SHAP Analysis: Case-Specific Investigation
# So far, we performed basic SHAP analysis showing the overall feature importance in our model, as seen in the summary plot. This analysis provided general insights regarding the impact of each feature, identifying age category, general health status, sex, smoking, and stroke history as the most influential predictors for heart disease.
# However, this general analysis doesn't explain how the model arrives at decisions for specific cases. To deepen our understanding and examine the model's decision-making process at the individual level, we'll now extend our SHAP analysis using Waterfall plots.
# 
# We'll select four representative cases from our test set:
# 
# True Positive: A patient with heart disease correctly identified by the model
# True Negative: A patient without heart disease correctly identified
# False Positive: A patient without heart disease incorrectly classified as having heart disease
# False Negative: A patient with heart disease the model failed to identify
# 
# For each case, we'll calculate specific SHAP values and create a Waterfall plot showing how each feature contributed to the final prediction.
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
# # Analysis of SHAP Results for Specific Cases
# 
# The expanded SHAP analysis we conducted provides important insights into how the model makes decisions at the individual patient level. While the general graph showed us the importance of features globally, analyzing individual cases allows us to understand the dynamics of the model's decision-making process.
# 
# In the case of correct heart disease identification <u>**True Positive**</u> with a probability of 52.4%, we can see that the most significant factor pushing the prediction toward positive was **AgeCategory** with a substantial contribution of +0.87. Interestingly, **GenHealth** contributed negatively (-0.38), while **SleepTime** (-0.26) and **Sex** (-0.24) also pushed against the prediction. The collective impact of 13 other features provided a small positive contribution (+0.3), helping to ultimately push the model toward the correct prediction despite some contradicting signals.
# 
# In contrast, in the case of correctly identifying the absence of heart disease <u>**True Negative**</u> with a very low probability of 2.1%, the most dominant factor was **AgeCategory** with a significant negative contribution of -1.95. Additional factors that contributed to the negative prediction were **SleepTime** (-0.67), **GenHealth** (-0.6), and **MentalHealth** (-0.21). The model strongly leveraged these protective factors to correctly classify this individual as not having heart disease.
# 
# The analysis of incorrect cases teaches us about the model's weaknesses. In the case of a false positive prediction <u>**False Positive**</u> with a probability of 76.1%, **PhysicalActivity** was the decisive factor (+0.39), followed by **AgeCategory** (+0.37) and **PhysicalHealth** (+0.31). Despite a negative contribution from **SleepTime** (-0.23), the combined effect of these factors plus other features (+0.36) led to an incorrect positive prediction. This suggests the model may overemphasize certain physical factors in some cases.
# 
# The case of a false negative prediction <u>**False Negative**</u> with a probability of 15.0% reveals an interesting pattern. Here, **GenHealth** had a strong negative effect (-0.94), followed by **Asthma** (-0.51) and **SleepTime** (-0.41). Notably, **AgeCategory** actually made a positive contribution (+0.35), but it wasn't enough to overcome the other negative factors. The model failed to identify heart disease likely because the patient presented with generally good health indicators despite having heart disease.
# 
# From analyzing these four cases, we see that the model relies primarily on several central factors: **AgeCategory**, **GenHealth**, and **SleepTime**. Physical factors like **PhysicalActivity** and **PhysicalHealth** can strongly influence positive predictions, while health status indicators like **GenHealth** and **Asthma** can drive negative predictions. The model performs well in "classic" cases but struggles with patients who have contradicting risk factors or atypical presentations.
# 
# These findings suggest directions for future model improvement, such as better handling of cases with mixed signals, recalibrating the importance of sleep metrics, and potentially incorporating more nuanced interactions between age and other health indicators. The model might benefit from more sophisticated handling of patients who have heart disease despite otherwise healthy profiles.
# 
# SHAP analysis at the individual case level demonstrates the importance of looking beyond statistical performance metrics and provides valuable clinical insights that can help physicians assess the accuracy of model predictions in different scenarios.
#%% md
# # Summary and Thoughts for the future and Final Taught
# In this study, we embarked on a data-driven journey to uncover the underlying factors contributing to heart disease risk. Beginning with a comprehensive dataset containing various lifestyle and health attributes, we carefully preprocessed the data to ensure consistency and accuracy. This involved encoding categorical variables, organizing age groups into meaningful ranges, and transforming binary responses for better analysis. We also handled missing values, normalized numerical features, and engineered new variables to enhance model performance. For instance, age groups were restructured into broader 10-year ranges to capture more meaningful patterns, and binary health indicators such as smoking, alcohol consumption, and physical activity were mapped to numerical values. With a structured dataset in place, we conducted exploratory data analysis through visualizations, revealing key trends and correlations between lifestyle choices and heart disease prevalence.
# 
# Building upon these insights, we implemented machine learning models, including logistic regression and decision trees, to predict heart disease risk based on the identified factors. We applied various data transformation techniques, such as feature scaling and one-hot encoding, to optimize model performance. Additionally, we experimented with different feature selection methods to identify the most impactful predictors, analyzing how factors like kidney disease, stroke history, and sleep duration influenced heart disease risk. We evaluated models using accuracy, precision, recall, and F1-score, but despite extensive preprocessing and optimization, the prediction accuracy remained suboptimal. This suggests that heart disease risk may be influenced by additional factors not captured in the dataset, such as genetic predisposition, environmental conditions, or complex interactions between variables. Moreover, the dataset itself may not be sufficiently detailed or diverse, potentially limiting the model's ability to generalize effectively. Class imbalances, particularly in the distribution of positive heart disease cases, and potential biases in the data may have further impacted the results. While our models offer a foundational approach for risk assessment, further research with more diverse, high-quality data and advanced techniques, such as ensemble learning, or feature engineering based on medical expertise, is needed to improve predictive performance. Nonetheless, our findings contribute to the ongoing exploration of data-driven healthcare solutions and highlight the challenges of predicting complex medical conditions using machine learning.
# 