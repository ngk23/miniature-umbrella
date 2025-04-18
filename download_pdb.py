from Bio.PDB import PDBList

pdb_list = PDBList()
pdb_ids = ['1TUP', '4HHB', '1A8O']  
for pdb_id in pdb_ids:
    pdb_list.retrieve_pdb_file(pdb_id, pdir='pdb_files', file_format='pdb') 