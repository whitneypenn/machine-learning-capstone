import pandas as pd
import numpy as np

#input data
projects = pd.read_csv("Capstone_Data/io/Projects.csv")
schools = pd.read_csv("Capstone_Data/io/Schools.csv")

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

#merge projects with school information
projects = pd.merge(projects, schools, on='School ID', how='left')

## Columns to Dates ##
projects['Project Posted Date'] = pd.to_datetime(projects['Project Posted Date'])
projects['Project Expiration Date'] = pd.to_datetime(projects['Project Expiration Date'])

## Remove Rows with null values in these columns:
projects.dropna(subset = ['Project Title', 'Project Essay', 'Project Short Description', 'Project Need Statement','Project Resource Category' ], inplace=True) #removes a total of 45 rows

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

project_dummies.to_csv('projects_with_categorical_data.csv')


## Do the projects cluster better using TFIDFs of the need statements or
## using dummies of the categories that already exist?


## Actually useful features to cluster on:
