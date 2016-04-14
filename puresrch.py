# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:39:39 2016

@author: bmanubay
"""

import thermopyl as th 
from thermopyl import thermoml_lib
import cirpy
import numpy as np
import pandas as pd
from sklearn.externals.joblib import Memory

mem = Memory(cachedir="/home/bmanubay/.thermoml/")

@mem.cache
def resolve_cached(x, rtype):
   return cirpy.resolve(x, rtype)

df = th.pandas_dataframe()
dt = list(df.columns)

bad_filenames = ["/home/bmanubay/.thermoml/j.fluid.2013.12.014.xml"]  # This file confirmed to have possible data entry errors.
df = df[~df.filename.isin(bad_filenames)]

experiments = ["Mass density, kg/m3","Speed of sound, m/s", "Relative permittivity at zero frequency", "Activity coefficient", "Specific heat capacity at constant pressure, J/K/kg", "Molar heat capacity at constant pressure, J/K/mol", "Molar heat capacity at constant volume, J/K/mol", "Molar volume, m3/mol", "Specific volume, m3/kg", "Molar enthalpy, kJ/mol"]

ind_list = [df[exp].dropna().index for exp in experiments]
ind = reduce(lambda x,y: x.union(y), ind_list)
df = df.ix[ind]

name_to_formula = pd.read_hdf("/home/bmanubay/.thermoml/compound_name_to_formula.h5", 'data')
name_to_formula = name_to_formula.dropna()

# Extract rows with two components
df["n_components"] = df.components.apply(lambda x: len(x.split("__")))
df = df[df.n_components == 1]
df.dropna(axis=1, how='all', inplace=True)


df["formula"] = df.components.apply(lambda chemical: name_to_formula[chemical])

heavy_atoms = ["C", "O"]
desired_atoms = ["H"] + heavy_atoms

df["n_atoms"] = df.formula.apply(lambda formula_string : thermoml_lib.count_atoms(formula_string))
df["n_heavy_atoms"] = df.formula.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, heavy_atoms))
df["n_desired_atoms"] = df.formula.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, desired_atoms))
df["n_other_atoms"] = df.n_atoms - df.n_desired_atoms

df = df[df.n_other_atoms == 0]

df = df[df.n_heavy_atoms > 0]
df = df[df.n_heavy_atoms <= 10]
df.dropna(axis=1, how='all', inplace=True)

df["smiles"] = df.components.apply(lambda x: resolve_cached(x, "smiles"))  # This should be cached via sklearn.
df = df[df.smiles != None]
df = df[df["smiles"].str.contains('=O') == False] # Getting rid of data sets with C=O and C=C occurrences
df = df[df["smiles"].str.contains('#') == False]
df = df[df["smiles"].str.contains('O=') == False]
df = df[df["smiles"].str.contains('=C') == False]
df = df[df["smiles"].str.contains('C=') == False]
df.dropna(subset=["smiles"], inplace=True)
df = df.ix[df.smiles.dropna().index]

    
df["cas"] = df.components.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "cas")))  # This should be cached via sklearn.
df = df[df.cas != None]
df = df.ix[df.cas.dropna().index]

# Neither names (components) nor smiles are unique.  Use CAS to ensure consistency.
cannonical_smiles_lookup = df.groupby("cas").smiles.first()
cannonical_components_lookup = df.groupby("cas").components.first()


df["smiles"] = df.cas.apply(lambda x: cannonical_smiles_lookup[x])
df["components"] = df.cas.apply(lambda x: cannonical_components_lookup[x])

# Extract rows with temperature between 128 and 399 K
df = df[df['Temperature, K'] > 250.]
df = df[df['Temperature, K'] < 400.]

# Extract rows with pressure between 101.325 kPa and 101325 kPa
df = df[df['Pressure, kPa'] > 100.]
df = df[df['Pressure, kPa'] < 102000.]

# Strip rows not in liquid phase
df = df[df['phase']=='Liquid']

df.dropna(axis=1, how='all', inplace=True)

df["filename"] = df["filename"].map(lambda x: x.lstrip('/home/bmanubay/.thermoml/').rstrip('.xml'))

dfbig = pd.concat([df['filename'], df["components"], df["Mass density, kg/m3"], df["Mass density, kg/m3_std"], df["Speed of sound, m/s"], df["Speed of sound, m/s_std"], df["Relative permittivity at zero frequency"], df["Relative permittivity at zero frequency_std"], df["Molar heat capacity at constant pressure, J/K/mol"], df["Molar heat capacity at constant pressure, J/K/mol_std"], df["Molar volume, m3/mol"], df["Molar volume, m3/mol_std"], df["Specific volume, m3/kg"], df["Specific volume, m3/kg_std"], df["Molar enthalpy, kJ/mol"], df["Molar enthalpy, kJ/mol"]] , axis=1, keys=["filename", "components", "Mass density, kg/m3", "Mass density, kg/m3_std", "Speed of sound, m/s", "Speed of sound, m/s_std", "Relative permittivity at zero frequency", "Relative permittivity at zero frequency_std", "Specific heat capacity at constant pressure, J/K/kg", "Specific heat capacity at constant pressure, J/K/kg_std", "Molar heat capacity at constant pressure, J/K/mol", "Molar heat capacity at constant pressure, J/K/mol_std", "Molar heat capacity at constant volume, J/K/mol", "Molar heat capacity at constant volume, J/K/mol_std", "Molar volume, m3/mol", "Molar volume, m3/mol_std", "Specific volume, m3/kg", "Specific volume, m3/kg_std", "Molar enthalpy, kJ/mol", "Molar enthalpy, kJ/mol"])
dfbig.groupby(["filename"])
a = dfbig["filename"].value_counts()
b = dfbig["components"].value_counts()

df1 = pd.concat([df['filename'], df["components"], df["Mass density, kg/m3"], df["Mass density, kg/m3_std"]], axis=1, keys=["filename", "components", "Mass density, kg/m3", "Mass density, kg/m3_std"])
df1["Mass density, kg/m3_std"].replace('nan', np.nan, inplace=True)
df1 = df1[np.isnan(df1["Mass density, kg/m3_std"])==False]
a1 = df1["filename"].value_counts()
b1 = df1["components"].value_counts()

df2 = pd.concat([df['filename'], df["components"], df["Speed of sound, m/s"], df["Speed of sound, m/s_std"]], axis=1, keys=["filename", "components", "Speed of sound, m/s", "Speed of sound, m/s_std"])
df2["Speed of sound, m/s_std"].replace('nan', np.nan, inplace=True)
df2 = df2[np.isnan(df2["Speed of sound, m/s_std"])==False]
a2= df2["filename"].value_counts()
b2 = df2["components"].value_counts()

df3 = pd.concat([df['filename'], df["components"], df["Relative permittivity at zero frequency"], df["Relative permittivity at zero frequency_std"]], axis=1, keys=["filename", "components", "Relative permittivity at zero frequency", "Relative permittivity at zero frequency_std"])
df3["Relative permittivity at zero frequency_std"].replace('nan', np.nan, inplace=True)
df3 = df3[np.isnan(df3["Relative permittivity at zero frequency_std"])==False]
a3 = df3["filename"].value_counts()
b3 = df3["components"].value_counts()

df6 = pd.concat([df['filename'], df["components"], df["Molar heat capacity at constant pressure, J/K/mol"], df["Molar heat capacity at constant pressure, J/K/mol_std"]], axis=1, keys=["filename", "components", "Molar heat capacity at constant pressure, J/K/mol", "Molar heat capacity at constant pressure, J/K/mol_std"])
df6["Molar heat capacity at constant pressure, J/K/mol_std"].replace('nan', np.nan, inplace=True)
df6 = df6[np.isnan(df6["Molar heat capacity at constant pressure, J/K/mol_std"])==False]
a6 = df6["filename"].value_counts()
b6 = df6["components"].value_counts()

df8 = pd.concat([df['filename'], df["components"], df["Molar volume, m3/mol"], df["Molar volume, m3/mol_std"]], axis=1, keys=["filename", "components", "Molar volume, m3/mol", "Molar volume, m3/mol_std"])
df8["Molar volume, m3/mol_std"].replace('nan', np.nan, inplace=True)
df8 = df8[np.isnan(df8["Molar volume, m3/mol_std"])==False]
a8 = df8["filename"].value_counts()
b8 = df8["components"].value_counts()

df9 = pd.concat([df['filename'], df["components"], df["Specific volume, m3/kg"], df["Specific volume, m3/kg_std"]], axis=1, keys=["filename", "components", "Specific volume, m3/kg", "Specific volume, m3/kg_std"])
df9["Specific volume, m3/kg_std"].replace('nan', np.nan, inplace=True)
df9 = df9[np.isnan(df9["Specific volume, m3/kg_std"])==False]
a9 = df9["filename"].value_counts()
b9 = df9["components"].value_counts()

df10 = pd.concat([df['filename'], df["components"], df["Molar enthalpy, kJ/mol"], df["Molar enthalpy, kJ/mol_std"]], axis=1, keys=["filename", "components", "Molar enthalpy, kJ/mol", "Molar enthalpy, kJ/mol_std"])
df10["Molar enthalpy, kJ/mol_std"].replace('nan', np.nan, inplace=True)
df10 = df10[np.isnan(df10["Molar enthalpy, kJ/mol_std"])==False]
a10 = df10["filename"].value_counts()
b10 = df10["components"].value_counts()

dfbig.to_csv("/home/bmanubay/.thermoml/tables/Ken/Ken_pure_sets_all.csv")
a.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts_all.csv")
b.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_all.csv")

df1.to_csv("/home/bmanubay/.thermoml/tables/Ken/dens_pure.csv")
a1.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_dens.csv")
b1.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_dens.csv")

df2.to_csv("/home/bmanubay/.thermoml/tables/Ken/sos_pure.csv")
a2.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_sos.csv")
b2.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_sos.csv")

df3.to_csv("/home/bmanubay/.thermoml/tables/Ken/dielec_pure.csv")
a3.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_dielec.csv")
b3.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_dielec.csv")

df6.to_csv("/home/bmanubay/.thermoml/tables/Ken/cpmol_pure.csv")
a6.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_cpmol.csv")
b6.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_cpmol.csv")

df7.to_csv("/home/bmanubay/.thermoml/tables/Ken/cvmol_pure.csv")
a7.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_cvmol.csv")
b7.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_cvmol.csv")

df8.to_csv("/home/bmanubay/.thermoml/tables/Ken/vmol_pure.csv")
a8.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_vmol.csv")
b8.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_vmol.csv")

df9.to_csv("/home/bmanubay/.thermoml/tables/Ken/vspec_pure.csv")
a9.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_vspec.csv")
b9.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_vspec.csv")

df10.to_csv("/home/bmanubay/.thermoml/tables/Ken/hmol_pure.csv")
a10.to_csv("/home/bmanubay/.thermoml/tables/Ken/purename_counts_hmol.csv")
b10.to_csv("/home/bmanubay/.thermoml/tables/Ken/purecomp_counts_hmol.csv")

