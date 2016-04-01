# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:29:39 2016

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

experiments = ["Mass density, kg/m3", "Partial molar enthalpy, kJ/mol", "Partial molar volume, m3/mol", "Partial pressure, kPa", "Excess molar Gibbs energy, kJ/mol", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol", "Excess molar heat capacity, J/K/mol", "Excess molar volume, m3/mol", "(Relative) activity", "Activity coefficient", "Speed of sound, m/s"]

ind_list = [df[exp].dropna().index for exp in experiments]
ind = reduce(lambda x,y: x.union(y), ind_list)
df = df.ix[ind]

name_to_formula = pd.read_hdf("/home/bmanubay/.thermoml/compound_name_to_formula.h5", 'data')
name_to_formula = name_to_formula.dropna()

# Extract rows with two components
df["n_components"] = df.components.apply(lambda x: len(x.split("__")))
df = df[df.n_components == 2]
df.dropna(axis=1, how='all', inplace=True)

counts_data = {}
counts_data["0.  Two Components"] = df.count()[experiments]

# Split components into separate columns (to use name_to_formula)
df["x1"], df["x2"] =  zip(*df["components"].str.split('__').tolist())
del df["components"]
df['x2'].replace('', np.nan, inplace=True)
df.dropna(subset=['x2'], inplace=True)

df["formula1"] = df.x1.apply(lambda chemical: name_to_formula[chemical])
df["formula2"] = df.x2.apply(lambda chemical: name_to_formula[chemical])

heavy_atoms = ["C", "O"]
desired_atoms = ["H"] + heavy_atoms

df["n_atoms1"] = df.formula1.apply(lambda formula_string : thermoml_lib.count_atoms(formula_string))
df["n_heavy_atoms1"] = df.formula1.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, heavy_atoms))
df["n_desired_atoms1"] = df.formula1.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, desired_atoms))
df["n_other_atoms1"] = df.n_atoms1 - df.n_desired_atoms1
df["n_atoms2"] = df.formula2.apply(lambda formula_string : thermoml_lib.count_atoms(formula_string))
df["n_heavy_atoms2"] = df.formula2.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, heavy_atoms))
df["n_desired_atoms2"] = df.formula2.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, desired_atoms))
df["n_other_atoms2"] = df.n_atoms2 - df.n_desired_atoms2

df = df[df.n_other_atoms1 == 0]
df = df[df.n_other_atoms2 == 0]

counts_data["1.  Druglike Elements"] = df.count()[experiments]

df = df[df.n_heavy_atoms1 > 0]
df = df[df.n_heavy_atoms1 <= 10]
df = df[df.n_heavy_atoms2 > 0]
df = df[df.n_heavy_atoms2 <= 10]
df.dropna(axis=1, how='all', inplace=True)

counts_data["2.  Heavy Atoms"] = df.count()[experiments]

df["smiles1"] = df.x1.apply(lambda x: resolve_cached(x, "smiles"))  # This should be cached via sklearn.
df = df[df.smiles1 != None]
df = df[df["smiles1"].str.contains('C=O') == False] # Getting rid of data sets with C=O and C=C occurrences
df = df[df["smiles1"].str.contains('(C)=O') == False]
df = df[df["smiles1"].str.contains('O=C') == False]
df = df[df["smiles1"].str.contains('O=(C)') == False]
df = df[df["smiles1"].str.contains('C=C') == False]
df = df[df["smiles1"].str.contains('(C)=C') == False]
df = df[df["smiles1"].str.contains('C=(C)') == False] 
df.dropna(subset=["smiles1"], inplace=True)
df = df.ix[df.smiles1.dropna().index]
df["smiles2"] = df.x2.apply(lambda x: resolve_cached(x, "smiles"))  # This should be cached via sklearn.
df = df[df.smiles2 != None]
df = df[df["smiles2"].str.contains('C=O') == False] # Getting rid of data sets with C=O and C=C occurrences
df = df[df["smiles2"].str.contains('(C)=O') == False]
df = df[df["smiles2"].str.contains('O=C') == False]
df = df[df["smiles2"].str.contains('O=(C)') == False]
df = df[df["smiles2"].str.contains('C=C') == False]
df = df[df["smiles2"].str.contains('(C)=C') == False]
df = df[df["smiles2"].str.contains('C=(C)') == False]
df.dropna(subset=["smiles2"], inplace=True)
df = df.ix[df.smiles2.dropna().index]

    
df["cas1"] = df.x1.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "cas")))  # This should be cached via sklearn.
df = df[df.cas1 != None]
df = df.ix[df.cas1.dropna().index]
df["cas2"] = df.x2.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "cas")))  # This should be cached via sklearn.
df = df[df.cas2 != None]
df = df.ix[df.cas2.dropna().index]


# Neither names (components) nor smiles are unique.  Use CAS to ensure consistency.
cannonical_smiles_lookup1 = df.groupby("cas1").smiles1.first()
cannonical_components_lookup1 = df.groupby("cas1").x1.first()
cannonical_smiles_lookup2 = df.groupby("cas2").smiles2.first()
cannonical_components_lookup2 = df.groupby("cas2").x2.first()


df["smiles1"] = df.cas1.apply(lambda x: cannonical_smiles_lookup1[x])
df["x1"] = df.cas1.apply(lambda x: cannonical_components_lookup1[x])
df["smiles2"] = df.cas2.apply(lambda x: cannonical_smiles_lookup2[x])
df["x2"] = df.cas2.apply(lambda x: cannonical_components_lookup2[x])

# Extract rows with temperature between 128 and 399 K
df = df[df['Temperature, K'] > 128.]
df = df[df['Temperature, K'] < 400.]

counts_data["3.  Temperature"] = df.count()[experiments]

# Extract rows with pressure between 101.325 kPa and 101325 kPa
df = df[df['Pressure, kPa'] > 100.]
df = df[df['Pressure, kPa'] < 102000.]

counts_data["4.  Pressure"] = df.count()[experiments]

# Strip rows not in liquid phase
df = df[df['phase']=='Liquid']

counts_data["5.  Liquid state"] = df.count()[experiments]


df.dropna(axis=1, how='all', inplace=True)

df.to_csv("/home/bmanubay/.thermoml/tables/full_filtered_data.csv")

experiments = ["Mass density, kg/m3", "Partial molar enthalpy, kJ/mol", "Partial molar volume, m3/mol", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol", "Excess molar heat capacity, J/K/mol", "Excess molar volume, m3/mol", "Activity coefficient", "Speed of sound, m/s"]

mu = df.groupby(["x1", "x2", "smiles1", "smiles2", "cas1", "cas2", "Temperature, K", "Pressure, kPa"])[experiments].mean()

counts_data["6.  Aggregate T, P"] = mu.count()[experiments]

counts_data = pd.DataFrame(counts_data).T

q = mu.reset_index()
q = q.ix[q[experiments].dropna().index]
q.to_csv("/home/bmanubay/.thermoml/tables/more_data.csv")

counts_data.ix["7.  Density+PME+PMV+PP+EMG+EME+EMHC+EMV"] = len(q)

print counts_data.to_latex()
