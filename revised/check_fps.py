import pandas as pd
import os

fps = ['AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP','Combined']

for x in fps:
    print("#" * 100)
    # Normal features
    normal_path = os.path.join("nafld", f"{x}.csv")
    if os.path.exists(normal_path):
        df = pd.read_csv(normal_path, index_col=0)
        print(f"{x} (normal): {df.shape}")
    else:
        print(f"{x} (normal): File not found")

    # Reduced features
    reduced_path = os.path.join("nafld", f"{x}_reduced.csv")
    if os.path.exists(reduced_path):
        df_red = pd.read_csv(reduced_path, index_col=0)
        print(f"{x} (reduced): {df_red.shape}")
    else:
        print(f"{x} (reduced): File not found")