#!/usr/bin/env python
"""
Fingerprints a compound database

Taks in a database in .sdf format, and creates a
pickled file with a Pandas dataframe that contains
the columns: 

'ID': An identifier extracted from the SDF file
'SMILES': SMILES string for the molecule
'RDKit': RDKit fingerprint 
'ECFP4': ECFP4 (Morgan) Fingerprint
'MACCS': MACCS fingerprint

Author: Gustavo Seabra
        Department of Medicinal Chemistry
        College of Pharmacy
        Univerity of Florida
        Gainesville, FL, USA
        seabra - at - cop - dot - ufl - dot - edu

"""

import sys
import pandas as pd
import numpy as np
import time
import gc
from multiprocessing import Pool, cpu_count
from pathlib import Path

# For the preprocessing
import preprocess

# RDKit
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys 
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# Suppress RDKit output
from rdkit import RDLogger
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)

# Cosmetic: progress bars
from tqdm import trange, tnrange
from tqdm.auto import tqdm


def fmt_time(t_seconds):
    t_hours, t_rem = divmod(t_seconds, 3600)
    t_minutes, t_seconds = divmod(t_rem,60)
    return f"{int(t_hours):0>2}:{int(t_minutes):0>2}:{t_seconds:05.2f}"

# Add Fingerprints
def get_rdkfingerprints(mol):
    fp = RDKFingerprint(mol)
    return np.array(list(map(int,fp.ToBitString())),dtype='i1')

def get_morganfingerprints(mol):
    fp = GetMorganFingerprintAsBitVect(mol,2)
    return np.array(list(map(int,fp.ToBitString())),dtype='i1')

def get_MACCSfingerprints(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(list(map(int,fp.ToBitString())),dtype='i1')

def add_fingerprints(frame):
    tqdm.pandas(desc="Generating RDKit Fingerprints:")
    frame['RDKit'] = frame['ROMol'].progress_apply(get_rdkfingerprints)
    
    tqdm.pandas(desc="Generating Morgan Fingerprints:")
    frame['ECFP4'] = frame['ROMol'].progress_apply(get_morganfingerprints)

    tqdm.pandas(desc="Generating MACCS Fingerprints:")
    frame['MACCS'] = frame['ROMol'].progress_apply(get_MACCSfingerprints)
    
    return frame

# ---

def add_id_and_smiles_columns(df,id_column):
    # Adds ID and SMILES column
    df["ID"]     = df['ROMol'].apply(lambda x: x.GetProp(id_column))
    df['SMILES'] = df['ROMol'].apply(Chem.MolToSmiles)
    return df

def preprocess_df(df):
    df, _dup = preprocess.preprocess_db(df, standardize_ROMol=False)
    del(_dup)
    return df

def fingerprint_pipeline(df, n_procs=10):

    df_split = np.array_split(df, n_procs)
    with Pool(n_procs) as p:
        # Cleans the DataFrame
        start_preprocess = time.time()
        print(f"Preprocessing DataFrame in {n_procs} parallel processes... ", end="", flush=True)
        df_split = p.map(preprocess_df,df_split)
        preprocess_time = time.time() - start_preprocess
        print(f"Done. ({fmt_time(preprocess_time)})")

        # Apply fingerprints
        start_fp = time.time()
        print("Applying fingerprints... ", end="", flush=True)
        df = pd.concat(p.map(add_fingerprints,df_split))
        del(df_split)
        fp_time = time.time() - start_fp
        print(f"Done. ({fmt_time(fp_time)})")

    return df, preprocess_time, fp_time
    
## --------------------

if __name__ == "__main__":

    import argparse, textwrap

    start_all = time.time()
    parser = argparse.ArgumentParser(
             description=textwrap.dedent("""\
                 Cleans and fingerprints a database. Outputs a pickled Pandas 
                 DataFrame, with the columns:

                    'ID':     An identifier extracted from the SDF file
                    'SMILES': SMILES string for the molecule
                    'RDKit':  RDKit fingerprint 
                    'ECFP4':  ECFP4 (Morgan) Fingerprint
                    'MACCS':  MACCS fingerprint """),
            formatter_class=argparse.RawTextHelpFormatter,
                                    epilog="Enjoy.")
    parser.add_argument("input_file", help="SDF  database file")
    parser.add_argument("out_name" , help="The output file name will be outname_df.pkl.bz2")
    parser.add_argument("--id_col"  , default='_Name',
                         help="Column containing the molecule IDs (default: infer from file)")
    parser.add_argument("--mem_chunk_size", type=int, default=100_000,
                        help="mem_chunk size to split the file (default: %(default)s).")
    parser.add_argument("--max_file_size", type=int, default=300_000,
                        help="Max file size. If the database is larger, split the output (default: %(default)s).")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Number of parallel process to use")
    parser.add_argument("-v", "--verbose" , action='store_true', help="Increase printout")
    args = parser.parse_args()

    input_file     = args.input_file
    out_name       = args.out_name
    id_column      = args.id_col
    n_procs        = args.parallel
    mem_chunk_size = args.mem_chunk_size
    max_file_size  = args.max_file_size
    verbose        = args.verbose

    print("Fingerprinting file: ", input_file)
    print("Database name:       ", out_name)
    print("ID Column:           ", id_column)
    print("# of processes:      ", n_procs)
    print("\n---")

    # Initialize the Molecule Supplier
    allowed_types = ['.sdf', '.smi']
    file_type = Path(input_file).suffix
    supplier = []
    assert file_type in allowed_types, f"File type {file_type} not recognized."
    if file_type == '.sdf':
        supplier  = Chem.rdmolfiles.SDMolSupplier(input_file)
    elif file_type == '.smi':
        assert id_column == '_Name', "SMILES file must not have id_column assigned."
        supplier  = Chem.rdmolfiles.SmilesMolSupplier(input_file)
    db_size = len(supplier)
    supplier.reset()

    # We will need to split the work into different files
    full_file_splits, file_remainder = divmod(db_size, max_file_size)
    n_file_splits = full_file_splits
    if file_remainder > 0: n_file_splits += 1

    print(f"The full database contains {db_size:,} records.")
    print(f"The maximum (output) file size is {max_file_size:,} records.")
    if n_file_splits > 1:
        print(f" ==> The final results will be split into {n_file_splits} files.")

    preprocess_time = 0
    fp_time      = 0
    save_time    = 0
    read_time    = 0

    name_add = ""
    final_size = 0
    for file_split in range(n_file_splits):
        print("="*60)
        print(f"Processing split # {file_split}.")


        this_file_size = min(db_size,max_file_size)
        if file_split == n_file_splits - 1: this_file_size = file_remainder

        # To allow for procesing files of large size, we
        # split the data into mem_chunks
        if n_file_splits > 1: name_add = f"_{file_split:02d}"

        out_file = f"{out_name}{name_add}_fp.pkl.bz2"
        full_df = pd.DataFrame()

        full_mem_chunks, remainder = divmod(this_file_size, mem_chunk_size)
        n_mem_chunks = full_mem_chunks 
        if remainder > 0: n_mem_chunks += 1

        print((f"This split has {this_file_size:,} molecules, and will be split " 
            f"into {n_mem_chunks} mem_chunks of {mem_chunk_size:,} molecules (max) each."))

        for mem_chunk in range(n_mem_chunks):

            print("-"*60)
            print(f"Processing mem_chunk {mem_chunk+1}.")
            mem_chunk_df = []

            # ----  Reads molecules
            begin_read = time.time()
            print("Reading molecules... ", end="", flush=True)
            mols_read = 0
            pbar = tqdm(total=mem_chunk_size)
            while mols_read < mem_chunk_size:

                if file_type == '.sdf':
                    if supplier.atEnd(): break
                try:
                    this_mol = next(supplier)
                except:
                    break
                if this_mol is not None: 
                    res = Chem.SanitizeMol(this_mol, catchErrors=True)
                    if not bool(res): 
                        mem_chunk_df.append(this_mol)
                        mols_read += 1
                        pbar.update(1)
            pbar.close()

            mem_chunk_df = pd.DataFrame({'ROMol':mem_chunk_df})
            mem_chunk_df = add_id_and_smiles_columns(mem_chunk_df, id_column)
            this_read_time = time.time() - begin_read
            read_time += this_read_time
            print(f"Done. ({fmt_time(this_read_time)})", flush=True)
            # ---- Done reading molecules

            # Process the database
            mem_chunk_df, mem_chunk_preprocess_time, mem_chunk_fp_time = fingerprint_pipeline(mem_chunk_df, n_procs)
            
            # Accumulates the result into a larger dataframe
            mem_chunk_df.drop(columns=['ROMol','InChI Key'], inplace=True)
            preprocess_time += mem_chunk_preprocess_time
            fp_time         += mem_chunk_fp_time
            full_df = full_df.append(mem_chunk_df, ignore_index=True)
            del(mem_chunk_df)
            gc.collect()
        #---End mem_chunk Loop 


        final_size += len(full_df)
        print(f"File {out_file} has {len(full_df):,} distinct molecules.")
        print(f"Currently, the total database has {final_size:,} molecules.\n")

        # Save the final DataFrame to a pkl file
        start_save = time.time()
        print("Saving Pickled DataFrame... ", end="", flush=True)
        full_df.to_pickle(out_file)
        del(full_df)
        gc.collect()
        this_save_time = time.time() - start_save
        save_time += this_save_time
        print(f"Done. ({fmt_time(this_save_time)})", flush=True)
    #---End file_split loop

    total_time = time.time() - start_all

    print(f"Finished processing file {input_file} with {db_size:,} molecules initially.")
    print(f"After processing, the final database has {final_size:,} molecules.")
    print('='*33)
    print(f"TIMINGS          {'HH:MM:SS.SS':>16}")
    print('-'*33)
    print(f"Reading          {fmt_time(read_time   ):>16}")
    print(f"Cleaning         {fmt_time(preprocess_time):>16}")
    print(f"Fingerprinting   {fmt_time(fp_time     ):>16}")
    print(f"Saving           {fmt_time(save_time   ):>16}")
    print(f"TOTAL            {fmt_time(total_time  ):>16}")
    print('='*33)
    print("Done")

