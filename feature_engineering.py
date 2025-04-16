import numpy as np
from parse_pdb import parse_pdb_files

# Directory containing PDB files
directory = 'pdb_files'

# Parse PDB files and extract features
features = parse_pdb_files(directory)

# Function to calculate distances between atoms
def calculate_distances(features):
    distances = []
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            if i < j:  # Avoid duplicate calculations
                coord1 = feature1['coordinates']
                coord2 = feature2['coordinates']
                distance = np.linalg.norm(coord1 - coord2)
                distances.append(distance)
    return distances

# Calculate distances between atoms
distances = calculate_distances(features)

# Print first 5 distances as a sample
print('Sample distances:', distances[:5]) 