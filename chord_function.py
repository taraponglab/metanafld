import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from d3blocks import D3Blocks


def similarity_matrix(df, cutpoint=0.4):
    """
    df = dataframe result
    cutpoint  = similarity level you want to cut
    """
    # Generate molecular fingerprints using RDKit
    molecules = [Chem.MolFromSmiles(smi) for smi in df['CANONICAL_SMILES']]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in molecules]

    # Compute the Tanimoto similarity matrix
    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
    for i in range(len(fingerprints)):
        for j in range(i, len(fingerprints)):
            similarity = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Ensure matrix symmetry

    # Extract chemical names for labeling
    chemicals = df['CHEMICAL_NAME'].tolist()

    # Create links between chemicals based on similarity, with transparency (alpha) based on similarity
    chord_links = []
    for i in range(len(chemicals)):
        for j in range(i + 1, len(chemicals)):
            similarity = similarity_matrix[i, j]
            if similarity > cutpoint:  # Filter low similarities
                # Use chemical names instead of indices
                chord_links.append((chemicals[i], chemicals[j], similarity))

    # Convert chord links to DataFrame with chemical names for source and target, and value (similarity) and alpha
    chord_data = pd.DataFrame(chord_links, columns=['source', 'target', 'weight'])
    chord_data.to_csv(f"chord_{cutpoint}.csv")
    # Initialize
    d3 = D3Blocks() # ls /tmp/d3blocks/chord.html
    d3.chord(chord_data) #convert -density 500 Chord_D3blocks.svg Chord_D3blocks.png
    return chord_data

df = pd.read_csv("dataset.csv")
chord_df = similarity_matrix(df)