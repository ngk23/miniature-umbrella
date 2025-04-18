import numpy as np
from parse_pdb import parse_pdb_files

directory = 'pdb_files'
features = parse_pdb_files(directory)
def calculate_distances(features):
    distances = []
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            if i < j:  
                coord1 = feature1['coordinates']
                coord2 = feature2['coordinates']
                distance = np.linalg.norm(coord1 - coord2)
                distances.append(distance)
    return distances

distances = calculate_distances(features)

print('Sample distances:', distances[:5]) 