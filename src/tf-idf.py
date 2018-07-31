import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#input data
essays = pd.read_csv('project_essays.csv')
