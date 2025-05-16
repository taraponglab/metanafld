#importing libraries
import pandas as pd
import numpy as np
from padelpy import padeldescriptor
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import os
from glob import glob
from astartes.molecules import train_test_split_molecules

def canonical_smiles(df, smiles_column):
    df['canonical_smiles'] = df[smiles_column].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    return df
def compute_fps(df, name):
    xml_files = glob(os.path.join(name,"*.xml"))
    xml_files.sort()
    FP_list = [
    'AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP']
    fp = dict(zip(FP_list, xml_files))
    print(fp)
    df['canonical_smiles'].to_csv(os.path.join(name, 'smiles.smi'), sep='\t', index=False, header=False)
    #Calculate fingerprints
    for i in FP_list:
        padeldescriptor(mol_dir=os.path.join(name, 'smiles.smi'),
                    d_file=os.path.join(name, i+'.csv'),
                    descriptortypes= fp[i],
                    retainorder=True, 
                    removesalt=True,
                    threads=2,
                    detectaromaticity=True,
                    standardizetautomers=True,
                    standardizenitro=True,
                    fingerprints=True
                    )
        Fingerprint = pd.read_csv(os.path.join(name, i+'.csv')).set_index(df.index)
        Fingerprint = Fingerprint.drop('Name', axis=1)
        Fingerprint.to_csv(os.path.join(name, i+'.csv'))
        print(i+'.csv', 'done')
    #load at pc
    fp_at = pd.read_csv(os.path.join(name,'AD2D.csv'    ) , index_col=0)
    fp_es = pd.read_csv(os.path.join(name,'EState.csv'  ) , index_col=0)
    fp_ke = pd.read_csv(os.path.join(name,'KRFP.csv'    ) , index_col=0)
    fp_pc = pd.read_csv(os.path.join(name,'PubChem.csv' ) , index_col=0)
    fp_ss = pd.read_csv(os.path.join(name,'SubFP.csv'   ) , index_col=0)
    fp_cd = pd.read_csv(os.path.join(name,'CDKGraph.csv') , index_col=0)
    fp_cn = pd.read_csv(os.path.join(name,'CDK.csv'     ) , index_col=0)
    fp_kc = pd.read_csv(os.path.join(name,'KRFPC.csv'   ) , index_col=0)
    fp_ce = pd.read_csv(os.path.join(name,'CDKExt.csv'  ) , index_col=0)
    fp_sc = pd.read_csv(os.path.join(name,'SubFPC.csv'  ) , index_col=0)
    fp_ac = pd.read_csv(os.path.join(name,'AP2DC.csv'   ) , index_col=0)
    fp_ma = pd.read_csv(os.path.join(name,'MACCS.csv'   ) , index_col=0)
    
    fp_at.to_csv(os.path.join(name, 'xat.csv'))
    fp_es.to_csv(os.path.join(name, 'xes.csv'))
    fp_ke.to_csv(os.path.join(name, 'xke.csv'))
    fp_pc.to_csv(os.path.join(name, 'xpc.csv'))
    fp_ss.to_csv(os.path.join(name, 'xss.csv'))
    fp_cd.to_csv(os.path.join(name, 'xcd.csv'))
    fp_cn.to_csv(os.path.join(name, 'xcn.csv'))
    fp_kc.to_csv(os.path.join(name, 'xkc.csv'))
    fp_ce.to_csv(os.path.join(name, 'xce.csv'))
    fp_sc.to_csv(os.path.join(name, 'xsc.csv'))
    fp_ac.to_csv(os.path.join(name, 'xac.csv'))
    fp_ma.to_csv(os.path.join(name, 'xma.csv'))
    return fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma


def main():
    for i in ['nafld']: #'ames', 
        df = pd.read_csv(i+".csv", index_col=0)
        df = df.drop_duplicates(subset = ['canonical_smiles'], keep=False)
        print(i,":",  len(df))
        df.to_csv(os.path.join(i+".csv"))
        y_label = df['class'].values
        y_label = pd.DataFrame(y_label, columns=['class'], index=df.index)
        y_label.to_csv(os.path.join(i, 'y_label.csv'))
        #fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma = compute_fps(df, i)
if __name__ == "__main__":
    main()
    