# Turn imported .txt files (the format they are by default) into .csv files
import os

# The directory where the files are
path = "data/raw_data/"

# Loop through all the files in the working directory
for filename in os.listdir(path):
    # Split the filename into name and extension
    base, ext = os.path.splitext(filename)
    # Check if the extension is .txt
    if ext == ".txt":
        # If so, rename the file with a .csv extension
        os.rename(path + filename, path + base + ".csv")