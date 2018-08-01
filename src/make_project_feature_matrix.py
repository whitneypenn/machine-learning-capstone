import pandas as pd
import numpy as np

#input data
projects = pd.read_csv("Capstone_Data/io/Projects.csv")
schools = pd.read_csv("Capstone_Data/io/Schools.csv")
print('data loaded')

#### Get School Information ####
'''W: West, N: Northeast, M: Midwest, S: South '''
us_states_to_regions = {
    'Alabama': 'S',
    'Alaska': 'W',
    'Arizona': 'W',
    'Arkansas': 'S',
    'California': 'W',
    'Colorado': 'W',
    'Connecticut': 'N',
    'Delaware': 'S',
    'Florida': 'S',
    'Georgia': 'S',
    'Hawaii': 'W',
    'Idaho': 'W',
    'Illinois': 'M',
    'Indiana': 'M',
    'Iowa': 'M',
    'Kansas': 'M',
    'Kentucky': 'S',
    'Louisiana': 'S',
    'Maine': 'N',
    'Maryland': 'S',
    'Massachusetts': 'N',
    'Michigan': 'M',
    'Minnesota': 'M',
    'Mississippi': 'S',
    'Missouri': 'M',
    'Montana': 'W',
    'Nebraska': 'M',
    'Nevada': 'W',
    'New Hampshire': 'N',
    'New Jersey': 'N',
    'New Mexico': 'W',
    'New York': 'N',
    'North Carolina': 'S',
    'North Dakota': 'M',
    'Ohio': 'M',
    'Oklahoma': 'S',
    'Oregon': 'W',
    'Pennsylvania': 'N',
    'Rhode Island': 'N',
    'South Carolina': 'S',
    'South Dakota': 'M',
    'Tennessee': 'S',
    'Texas': 'S',
    'Utah': 'W',
    'Vermont': 'N',
    'Virginia': 'S',
    'Washington': 'W',
    'West Virginia': 'S',
    'Wisconsin': 'M',
    'Wyoming': 'W',
    'District of Columbia': 'S',
}

schools['Region'] = schools['School State'].map(us_states_to_regions)
metro_type = pd.get_dummies(schools['School Metro Type'])
region_dummies = pd.get_dummies(schools['Region'])
schools = pd.concat([schools, metro_type, region_dummies], axis=1)
print('schools dataframe made')

#merge projects with school information
projects = pd.merge(projects, schools, on='School ID', how='left')
print('projects merged with school information')

## Remove Rows with null values in these columns:
projects.dropna(subset = ['Project Title', 'Project Essay', 'Project Short Description', 'Project Need Statement','Project Resource Category','School Metro Type','Region'], inplace=True) #removes a total of 45 rows
print('na rows removed')

## Subset Data
projects = projects.sample(frac=.01, random_state=3)

projects_keep_categories = projects[['Project Title', 'Project Resource Category', 'Project Subject Category Tree', 'Project Subject Subcategory Tree','Project Type', 'School Metro Type','Region', 'Project Grade Level Category']]
projects_keep_categories.to_csv('projects_with_categorical_data.csv')

## Columns to Dummies:
resource_types = pd.get_dummies(projects['Project Resource Category']) #17 columns
subject_category = pd.get_dummies(projects['Project Subject Category Tree']) #51 columns
subject_subcategories = pd.get_dummies(projects['Project Subject Subcategory Tree']) #431 columns
project_type = pd.get_dummies(projects['Project Type']) #3 columns
grade_level = pd.get_dummies(projects['Project Grade Level Category']) #5 Columns

## Add Columns Together
project_dummies = pd.concat([projects, resource_types, subject_category, subject_subcategories, project_type, grade_level], axis=1)

#drop the dummied columns
project_dummies.drop(['Project Resource Category', 'Project Subject Category Tree', 'Project Subject Subcategory Tree','Project Type', 'Project Current Status', 'Project Fully Funded Date', 'Project Posted Date', 'Teacher Project Posted Sequence', 'Project Essay', 'Project Short Description', 'Project Need Statement', 'Project Grade Level Category', 'School ID', 'School Name', 'School Metro Type', 'School Percentage Free Lunch', 'School State', 'School Zip', 'School City', 'School County', 'School District', 'Region', 'Teacher ID', 'Project Expiration Date' ], axis=1, inplace=True)

#Save Categorical Data to CSV
project_dummies.to_csv('projects_with_dummy_data.csv')
print('projects saved to CSV')

#Make corpus
school_info = projects[['School ID', 'Project Title', 'Project Essay', 'School Metro Type', 'Region']]
school_info.to_csv('project_essays.csv', index=False)
print('essays saved to CSV')
