from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Define the SMILES string
smi = r' C/C(=C\C=C\C=C(/C)\C=C\C=C(/C)\C(=O)C[C@]12[C@](O1)(C[C@H](CC2(C)C)O)C)/C=C/C=C(\C)/C=C=C3[C@](C[C@H](CC3(C)C)OC(=O)C)(C)O  '  # Example structure
mol = Chem.MolFromSmiles(smi)

# Define EState SMARTS patterns and their corresponding colors (Approximations)
patterns = [
    (Chem.MolFromSmarts('[#6]=[#6]'), (1, 1, 0)),      # EState FP 11 (Yellow) - C=C double bond
    (Chem.MolFromSmarts('[#7]'), (0, 1, 0.5)),         # EState FP 30 (Lime) - Any nitrogen
    (Chem.MolFromSmarts('[#8]=[#6]'), (0, 1, 1)),      # EState FP 35 (Cyan) - C=O
    (Chem.MolFromSmarts('[#6](-[#1])(-[#6])'), (1, 0, 1)),  # EState FP 13 (Magenta) - Secondary Carbon
    (Chem.MolFromSmarts('[#8H]'), (1, 0.5, 0))         # EState FP 34 (Orange) - Hydroxyl (OH)
]

hit_ats = set()  # Use a set to avoid duplicates
hit_bonds = set()
atom_cols = {}
bond_cols = {}

# Loop through each SMARTS pattern to find matches
for patt, color in patterns:
    atom_matches = mol.GetSubstructMatches(patt)  # Get all matches
    for match in atom_matches:
        hit_ats.update(match)  # Add matched atoms
        
        # Assign colors to atoms
        for at in match:
            atom_cols[at] = color  # Color for atoms
        
        # Find bonds for the matched atoms
        for i in range(len(match)):
            for j in range(i + 1, len(match)):
                bond = mol.GetBondBetweenAtoms(match[i], match[j])
                if bond:
                    hit_bonds.add(bond.GetIdx())  # Add bond index
                    bond_cols[bond.GetIdx()] = color  # Color for bonds

# Convert sets to lists for the drawing
hit_ats = list(hit_ats)
hit_bonds = list(hit_bonds)

# Create the drawing
d = rdMolDraw2D.MolDraw2DCairo(500, 500)
rdMolDraw2D.PrepareAndDrawMolecule(d, mol, 
                                    highlightAtoms=hit_ats, 
                                    highlightBonds=hit_bonds,
                                    highlightAtomColors=atom_cols,
                                    highlightBondColors=bond_cols)

# Save the drawing as an image file
d.FinishDrawing()
with open("Fucoxanthin_highlighted.png", "wb") as f:
    f.write(d.GetDrawingText())  # Write the drawing to file