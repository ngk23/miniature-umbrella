from Bio.PDB import PDBList

# Create a PDBList object
pdb_list = PDBList()

# List of PDB IDs to download
pdb_ids = ['1TUP', '4HHB', '1A8O']  # Replace with your desired PDB IDs

# Download each PDB file
for pdb_id in pdb_ids:
    pdb_list.retrieve_pdb_file(pdb_id, pdir='pdb_files', file_format='pdb') 