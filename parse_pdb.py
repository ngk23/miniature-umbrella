from Bio.PDB import PDBParser
import os

# Directory containing PDB files
directory = 'pdb_files'

# Initialize PDB parser
parser = PDBParser(QUIET=True)

# Function to parse PDB files and extract features
def parse_pdb_files(directory):
    features = []
    for filename in os.listdir(directory):
        if filename.endswith('.ent'):  # PDB files have .ent extension
            filepath = os.path.join(directory, filename)
            structure = parser.get_structure(filename, filepath)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            # Extract features like atom name, coordinates, etc.
                            features.append({
                                'atom_name': atom.get_name(),
                                'coordinates': atom.get_coord(),
                                'residue_name': residue.get_resname(),
                                'chain_id': chain.get_id(),
                            })
    return features

# Parse PDB files and extract features
features = parse_pdb_files(directory)

# Print extracted features
for feature in features[:5]:  # Print first 5 features as a sample
    print(feature) 