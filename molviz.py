import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
import os
import base64

def mol2png(df, width=300, height=300):
    for idx, row in df.iterrows():
        smiles = row['canonical_smiles']
        sources = row["source"]
        print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(width, height))
        img.save(os.path.join("mols",f"{sources}_{idx}.png"))

def mol23d(df):
    for idx, row in df.iterrows():
        smiles = row['canonical_smiles']
        source = row["source"]
        print(smiles)
        
        # Convert SMILES to molecule and add hydrogens
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        if not AllChem.EmbedMolecule(mol, AllChem.ETKDG()):
            AllChem.UFFOptimizeMolecule(mol)

        # Convert the molecule to a 3Dmol.js format for visualization
        block = Chem.MolToMolBlock(mol)
        
        # Set up 3D viewer
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(block, 'mol')
        viewer.setStyle({'stick': {}})
        viewer.zoomTo()
        viewer.show() 
        viewer.png()

def main():
    df = pd.read_csv("inhibitors.csv", index_col=0)
    mol23d(df)
    print("done")
    

main()