# Tools for fingerprinting chemical databases

```
Author: Gustavo Seabra 
        Department of Medicinal Chemistry
        College of Pharmacy
        Univerity of Florida
        Gainesville, FL, USA
        seabra - at - cop - dot - ufl - dot - edu
```

The `apply_fp.py` script will read an SDF file,
apply 3 types of fingerprints (RDKit, MACCS and ECFP4),
and save the data as a pickled dataframe.

The script can use parallel processing for speeding up
the fingerprinting, and also automatically splits the
database if it is too large to fit in memory.

Before fingerprinting, we pre-process the database as
usual for screening projects:
1. Standardize format
2. Removes molecules with atoms other than 'C','N','O','H','S','P','As','Se','F','Cl','Br' and 'I'
3. Removes salts
4. Removes fragments / mixtures
5. Removes charges by adding / removing H
6. Canonicalize tautomers
This is done with the script `preprocess.py`.

For details on how to run, please do:
`python apply_fp.py --help`

Files included here:
```
.
├── 0README.md   (this file)
├── __init__.py
├── apply_fp.py  (Execute this to apply the fingerprints)
├── preprocess.py (Tools to preprocess a database to remove invalid/undesired molecules)
├── chembl-26_phase-1.sdf (Example: 565 molecules in Phase-1 extracted from ChEMBL26)
└── chembl-26_fp.pkl.bz2 (Example: results of fingerprinting the chembl-26_phase-1.sdf file)
```

Enjoy!
