import os
import zipfile

# Download dataset using Kaggle API
os.system('kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p .')

# Unzip the dataset
with zipfile.ZipFile('fake-and-real-news-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Verify files
print("Files in directory:", os.listdir('.'))