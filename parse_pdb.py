from Bio.PDB import PDBParser
import os

directory = 'pdb_files'
parser = PDBParser(QUIET=True)
def parse_pdb_files(directory):
    features = []
    for filename in os.listdir(directory):
        if filename.endswith('.ent'):  
            filepath = os.path.join(directory, filename)
            structure = parser.get_structure(filename, filepath)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            features.append({
                                'atom_name': atom.get_name(),
                                'coordinates': atom.get_coord(),
                                'residue_name': residue.get_resname(),
                                'chain_id': chain.get_id(),
                            })
    return features

features = parse_pdb_files(directory)
for feature in features[:5]:  
    print(feature) 