import numpy as np
import os

from config import *

# This programme combines all separately generated data into one single file

# Path to the directory containing the files
# Warning: folder should have been created in the first place
combine_folder_path = 'combined_n{}/'.format(n)

for i in range(len(theta100)):
    # List to hold the arrays
    arrays = []

    # Loop through each file in the directory
    for file_name in os.listdir(folder.format(theta100[i])):
        # Check if the file has the correct prefix and suffix
        if file_name.startswith("results_n{}_".format(n)) and file_name.endswith(".npy"):
            # Load the array from the file
            file_path = os.path.join(folder.format(theta100[i]), file_name)
            array = np.load(file_path)
            # Append the array to the list
            arrays.append(array)

    # Concatenate all arrays into one
    output_array = np.concatenate(arrays)
    # Save the concatenated array
    np.save(os.path.join(combine_folder_path, "combinedOutput_n{}_point{}.npy".format(n, theta100[i])), output_array)

print("All arrays have been concatenated and saved.")

