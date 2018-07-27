import pandas as pd

# Read the Data
donations = pd.read_csv('Capstone_Data/io/Donations.csv')
donors = pd.read_csv("Capstone_Data/io/Donors.csv")
projects = pd.read_csv("Capstone_Data/io/Projects.csv")
resources = pd.read_csv("Capstone_Data/io/Resources.csv")
schools = pd.read_csv("Capstone_Data/io/Schools.csv")
teachers = pd.read_csv("Capstone_Data/io/Teachers.csv")

#merge the data frames
donations_and_donors = pd.merge(donations, donors, on='Donor ID', how='left')
donations_donors_projects = pd.merge(donations_and_donors, projects, on='Project ID', how='left')
donations_donors_projects_schools = pd.merge(donations_donors_projects, schools, on='School ID', how='left')

#save the data frame
donations_donors_projects_schools.to_csv('donations_information.csv')
