import pandas as pd
import numpy as np

def combine_projects_with_relevant_school_information(projects_dataframe, schools_dataframe):
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

    schools_dataframe['Region'] = schools_dataframe['School State'].map(us_states_to_regions)
    metro_type = pd.get_dummies(schools_dataframe['School Metro Type'])
    region_dummies = pd.get_dummies(schools_dataframe['Region'])
    schools_dataframe = pd.concat([schools_dataframe, metro_type, region_dummies], axis=1)
    print('schools dataframe made')

    #merge projects with school information
    projects_dataframe = pd.merge(projects_dataframe, schools_dataframe, on='School ID', how='left')
    print('projects merged with school information')

    ## Remove Rows with null values in these columns:
    projects_dataframe.dropna(subset = ['Project Title', 'Project Essay', 'Project Short Description', 'Project Need Statement','Project Resource Category','School Metro Type','Region'], inplace=True) #removes a total of 45 rows
    print('na rows removed')

    return projects_dataframe

def subset_data(total_projects, percentage_of_data=.01, random_state=None):
    subset_projects = total_projects.sample(frac=.01, random_state=random_state)
    return subset_projects

def get_categorical_data(subset_projects, file_name=None):
    projects_with_categories = projects[['Project Title', 'Project Resource Category', 'Project Subject Category Tree', 'Project Subject Subcategory Tree','Project Type', 'School Metro Type','Region', 'Project Grade Level Category']]
    if file_name:
        projects_with_categories.to_csv('{}.csv'.format(file_name))
        print('projects saved to CSV: {}.csv'.format(file_name))
    return projects_with_categories

def categorical_columns_to_dummies(subset_projects, file_name=None):
    resource_types = pd.get_dummies(subset_projects['Project Resource Category']) #17 columns
    subject_category = pd.get_dummies(subset_projects['Project Subject Category Tree']) #51 columns
    subject_subcategories = pd.get_dummies(subset_projects['Project Subject Subcategory Tree']) #431 columns
    project_type = pd.get_dummies(subset_projects['Project Type']) #3 columns
    grade_level = pd.get_dummies(subset_projects['Project Grade Level Category']) #5 Columns

    ## Add Columns Together
    project_dummies = pd.concat([subset_projects, resource_types, subject_category, subject_subcategories, project_type, grade_level], axis=1)

    #drop the dummied columns
    project_dummies.drop(['Project Resource Category', 'Project Subject Category Tree', 'Project Subject Subcategory Tree','Project Type', 'Project Current Status', 'Project Fully Funded Date', 'Project Posted Date', 'Teacher Project Posted Sequence', 'Project Essay', 'Project Short Description', 'Project Need Statement', 'Project Grade Level Category', 'School ID', 'School Name', 'School Metro Type', 'School Percentage Free Lunch', 'School State', 'School Zip', 'School City', 'School County', 'School District', 'Region', 'Teacher ID', 'Project Expiration Date' ], axis=1, inplace=True)

    #Save Categorical Data to CSV
    if file_name:
        project_dummies.to_csv('{}.csv'.format(file_name))
        print('projects saved to CSV: {}.csv'.format(file_name))
    return project_dummies

def make_corpus(subset_projects, file_name=None):
    corpus_with_info = subset_projects[['School ID', 'Project Title', 'Project Essay', 'School Metro Type', 'Region']]
    if file_name:
        corpus_with_info.to_csv('{}.csv'.format(file_name), index=False)
        print('essays saved to CSV: {}.csv'.format(file_name))
    return corpus_with_info

if __name__ == '__main__':
    #input data
    print('hello!')
    projects = pd.read_csv("Capstone_Data/io/Projects.csv")
    print('projects loaded')
    schools = pd.read_csv("Capstone_Data/io/Schools.csv")
    print('schools loaded')

    full_projects = combine_projects_with_relevant_school_information(projects, schools)

    ## Subset Data
    #Training Data
    projects = subset_data(full_projects, percentage_of_data=.01, random_state=3)
    #Testing Data
    test_projects = subset_data(full_projects, percentage_of_data=.001, random_state=4)
    print('data subset')

    #Categorical Data for KModes
    projects_keep_categories = get_categorical_data(projects, file_name='data/categorical_data')
    test_projects_keep_categories = get_categorical_data(projects, file_name='data/test_categorical_data')
    print(' categories made')

    #Dummy Data for KMeans
    project_dummies = categorical_columns_to_dummies(projects, file_name='data/projects_with_dummy_data')
    test_project_dummies = categorical_columns_to_dummies(projects , file_name='data/test_projects_with_dummy_data')
    print(' dummies made')

    #Make corpus for LDA
    project_corpus = make_corpus(projects, file_name='data/project_essays')
    test_project_corpus = make_corpus(test_projects, file_name='data/test_project_essays')
    print('corpi made')
