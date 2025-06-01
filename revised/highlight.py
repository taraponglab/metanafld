from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import os
import pandas as pd

def highlight_substructures(smiles, substructure_dict, compound_name="compound"):
    """
    smiles: str, the SMILES string of the molecule
    substructure_dict: dict, keys are names, values are SMARTS patterns
    compound_name: str, name for saving the image file
    Returns: list of substructure names found in the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Assign a color for each substructure
    import random
    random.seed(42)
    colors = {}
    color_palette = [
        (1,0,0), (0,1,0), (0,0,1), (1,0.5,0), (0.5,0,1), (0,0.7,0.7),
        (0.7,0.7,0), (0.7,0,0.7), (0,0.5,1), (1,0,0.5), (0.5,1,0)
    ]
    for i, key in enumerate(substructure_dict):
        colors[key] = color_palette[i % len(color_palette)]

    # Find atoms to highlight for each substructure
    highlight_atoms = []
    highlight_atom_colors = {}
    found_substructures = []
    for i, (name, smarts) in enumerate(substructure_dict.items()):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            found_substructures.append(name)
        for match in matches:
            highlight_atoms.extend(match)
            for idx in match:
                highlight_atom_colors[idx] = colors[name]

    # Draw the molecule
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(set(highlight_atoms)),
        highlightAtomColors=highlight_atom_colors
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    img = Image.open(BytesIO(png))
    plt.imshow(img)
    plt.axis('off')
    plt.close()
    # Save image
    img.save(os.path.join("molecules_highlight", f"{compound_name}_highlighted.png"))
    return found_substructures

# Example usage:
substructure_dict = {
    "Alkene": "[CX3;$([H2]),$([H1][#6]),$(C([#6])[#6])]=[CX3;$([H2]),$([H1][#6]),$(C([#6])[#6])]",   # SubFP5	    # (1,0,0)      # Red
    "Primary carbon": "[CX4H3][#6]",                                                                 # SubFP1       # (0,1,0)      # Green
    "Primary alcohol": "[OX2H][CX4H2;!$(C([OX2H])[O,S,#7,#15])]",                                    # SubFP13      # (0,0,1)      # Blue
    "Vinylogous ester": "[#6X3](=[OX1])[#6X3]=,:[#6X3][#6;!$(C=[O,N,S])]",                           # SubFP137     # (1,0.5,0)    # Orange
    "Alkylarylether": "[OX2](c)[CX4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])]",                            # SubFP18      # (0.5,0,1)    # Purple
    "Carboxylic acid": "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[$([OX2H]),$([OX1-])]",                   # SubFP84      # (0,0.7,0.7)  # Teal
    "Secondary carbon": "[CX4H2]([#6])[#6]",                                                         # SubFP2       # (0.7,0.7,0)  # Olive
    "Conjugated double bond": "*=*[*]=,#,:[*]",                                                      # SubFP287     # (0.7,0,0.7)  # Magenta
    "Heterocyclic": "[!#6;!R0]",                                                                     # SubFP275     # (0,0.5,1)    # Sky Blue
    "Aromatic": "a"                                                                                  # SubFP274     # (1,0,0.5)    # Pink
    # If you add more, next will be (0.5,1,0) # Lime
}

# Read the CSV
df = pd.read_csv("nafld.csv")

# Create output directory if it doesn't exist
os.makedirs("molecules_highlight", exist_ok=True)

for idx, row in df.iterrows():
    name = str(row["Name"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
    smiles = row["canonical_smiles"]
    try:
        found = highlight_substructures(smiles, substructure_dict, compound_name=name)
        print(f"Highlighted: {name}")
        print(f"  Substructures found: {', '.join(found) if found else 'None'}")
    except Exception as e:
        print(f"Failed for {name}: {e}")