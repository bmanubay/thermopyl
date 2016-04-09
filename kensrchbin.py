# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:33:13 2016

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

experiments = ["Mass density, kg/m3", "Partial molar enthalpy, kJ/mol", "Partial molar volume, m3/mol", "Partial pressure, kPa", "Excess molar Gibbs energy, kJ/mol", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol", "Excess molar heat capacity, J/K/mol", "Excess molar volume, m3/mol", "(Relative) activity", "Activity coefficient", "Speed of sound, m/s", "Relative permittivity at zero frequency"]

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
df = df[df["smiles1"].str.contains('=O') == False] # Getting rid of data sets with C=O and C=C occurrences
df = df[df["smiles1"].str.contains('#') == False]
df = df[df["smiles1"].str.contains('O=') == False]
df = df[df["smiles1"].str.contains('=C') == False]
df = df[df["smiles1"].str.contains('C=') == False]
df.dropna(subset=["smiles1"], inplace=True)
df = df.ix[df.smiles1.dropna().index]
df["smiles2"] = df.x2.apply(lambda x: resolve_cached(x, "smiles"))  # This should be cached via sklearn.
df = df[df.smiles2 != None]
df = df[df["smiles2"].str.contains('=O') == False] # Getting rid of data sets with C=O and C=C occurrences
df = df[df["smiles2"].str.contains('#') == False]
df = df[df["smiles2"].str.contains('O=') == False]
df = df[df["smiles2"].str.contains('=C') == False]
df = df[df["smiles2"].str.contains('C=') == False]
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

df["filename"] = df["filename"].map(lambda x: x.lstrip('/home/bmanubay/.thermoml/').rstrip('.xml'))


dfbig = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Mass density, kg/m3"], df["Mass density, kg/m3_std"], df["Excess molar enthalpy (molar enthalpy of mixing), kJ/mol"], df["Excess molar enthalpy (molar enthalpy of mixing), kJ/mol_std"], df["Excess molar heat capacity, J/K/mol"], df["Excess molar heat capacity, J/K/mol_std"], df["Excess molar volume, m3/mol"], df["Excess molar volume, m3/mol_std"], df["Activity coefficient"], df["Activity coefficient_std"], df["Speed of sound, m/s"], df["Speed of sound, m/s_std"], df["Relative permittivity at zero frequency"], df["Relative permittivity at zero frequency_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Mass density, kg/m3", "Mass density, kg/m3_std", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol_std", "Excess molar heat capacity, J/K/mol", "Excess molar heat capacity, J/K/mol_std", "Excess molar volume, m3/mol", "Excess molar volume, m3/mol_std", "Activity coefficient", "Activity coefficient_std", "Speed of sound, m/s", "Speed of sound, m/s_std", "Relative permittivity at zero frequency", "Relative permittivity at zero frequency_std"])

dfbig.groupby(["filename"])

a = dfbig["filename"].value_counts()
b = dfbig["x1"].value_counts()
c = dfbig["x2"].value_counts()
d = dfbig["components"].value_counts()

df1 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Mass density, kg/m3"], df["Mass density, kg/m3_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Mass density, kg/m3", "Mass density, kg/m3_std"])
df1["Mass density, kg/m3_std"].replace('nan', np.nan, inplace=True)
df1 = df1[np.isnan(df1["Mass density, kg/m3_std"])==False]
a1 = df1["filename"].value_counts()
b1 = df1["x1"].value_counts()
c1 = df1["x2"].value_counts()
d1 = df1["components"].value_counts()

df2 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Excess molar enthalpy (molar enthalpy of mixing), kJ/mol"], df["Excess molar enthalpy (molar enthalpy of mixing), kJ/mol_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol", "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol_std"])
df2["Excess molar enthalpy (molar enthalpy of mixing), kJ/mol_std"].replace('nan', np.nan, inplace=True)
df2 = df2[np.isnan(df2["Excess molar enthalpy (molar enthalpy of mixing), kJ/mol_std"])==False]
a2 = df2["filename"].value_counts()
b2 = df2["x1"].value_counts()
c2 = df2["x2"].value_counts()
d2 = df2["components"].value_counts()

df3 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Excess molar heat capacity, J/K/mol"], df["Excess molar heat capacity, J/K/mol_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Excess molar heat capacity, J/K/mol", "Excess molar heat capacity, J/K/mol_std"])
df3["Excess molar heat capacity, J/K/mol_std"].replace('nan', np.nan, inplace=True)
df3 = df3[np.isnan(df3["Excess molar heat capacity, J/K/mol_std"])==False]
a3 = df3["filename"].value_counts()
b3 = df3["x1"].value_counts()
c3 = df3["x2"].value_counts()
d3 = df3["components"].value_counts()

df4 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Excess molar volume, m3/mol"], df["Excess molar volume, m3/mol_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Excess molar volume, m3/mol", "Excess molar volume, m3/mol_std"])
df4["Excess molar volume, m3/mol_std"].replace('nan', np.nan, inplace=True)
df4 = df4[np.isnan(df4["Excess molar volume, m3/mol_std"])==False]
a4 = df4["filename"].value_counts()
b4 = df4["x1"].value_counts()
c4 = df4["x2"].value_counts()
d4 = df4["components"].value_counts()

df5 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Activity coefficient"], df["Activity coefficient_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Activity coefficient", "Activity coefficient_std"])
df5["Activity coefficient_std"].replace('nan', np.nan, inplace=True)
df5 = df5[np.isnan(df5["Activity coefficient_std"])==False]
a5 = df5["filename"].value_counts()
b5 = df5["x1"].value_counts()
c5 = df5["x2"].value_counts()
d5 = df5["components"].value_counts()

df6 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Speed of sound, m/s"], df["Speed of sound, m/s_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Speed of sound, m/s", "Speed of sound, m/s_std"])
df6["Speed of sound, m/s_std"].replace('nan', np.nan, inplace=True)
df6 = df6[np.isnan(df6["Speed of sound, m/s_std"])==False]
a6 = df6["filename"].value_counts()
b6 = df6["x1"].value_counts()
c6 = df6["x2"].value_counts()
d6 = df6["components"].value_counts()

df7 = pd.concat([df['filename'], df['x1'], df['x2'], df["components"], df["Mole fraction"], df["Relative permittivity at zero frequency"], df["Relative permittivity at zero frequency_std"]], axis=1, keys=["filename", "x1", "x2", "components", "Mole fraction", "Relative permittivity at zero frequency", "Relative permittivity at zero frequency_std"])
df7["Relative permittivity at zero frequency_std"].replace('nan', np.nan, inplace=True)
df7 = df7[np.isnan(df7["Relative permittivity at zero frequency_std"])==False]
a7 = df7["filename"].value_counts()
b7 = df7["x1"].value_counts()
c7 = df7["x2"].value_counts()
d7 = df7["components"].value_counts()


dfbig.to_csv("/home/bmanubay/.thermoml/tables/Ken/Ken_binary_sets.csv")
a.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts.csv")
b.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts.csv")
c.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts.csv")
d.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts.csv")

df1.to_csv("/home/bmanubay/.thermoml/tables/Ken/dens_bin.csv")
a1.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts1.csv")
b1.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts1.csv")
c1.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts1.csv")
d1.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts1.csv")

df2.to_csv("/home/bmanubay/.thermoml/tables/Ken/eme_bin.csv")
a2.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts2.csv")
b2.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts2.csv")
c2.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts2.csv")
d2.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts2.csv")

df3.to_csv("/home/bmanubay/.thermoml/tables/Ken/emhp_bin.csv")
a3.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts3.csv")
b3.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts3.csv")
c3.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts3.csv")
d3.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts3.csv")

df4.to_csv("/home/bmanubay/.thermoml/tables/Ken/emv_bin.csv")
a4.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts4.csv")
b4.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts4.csv")
c4.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts4.csv")
d4.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts4.csv")

df5.to_csv("/home/bmanubay/.thermoml/tables/Ken/actcoeff_bin.csv")
a5.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts5.csv")
b5.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts5.csv")
c5.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts5.csv")
d5.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts5.csv")

df6.to_csv("/home/bmanubay/.thermoml/tables/Ken/sos_bin.csv")
a6.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts6.csv")
b6.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts6.csv")
c6.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts6.csv")
d6.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts6.csv")

df7.to_csv("/home/bmanubay/.thermoml/tables/Ken/dielec_bin.csv")
a7.to_csv("/home/bmanubay/.thermoml/tables/Ken/name_counts7.csv")
b7.to_csv("/home/bmanubay/.thermoml/tables/Ken/x1_counts7.csv")
c7.to_csv("/home/bmanubay/.thermoml/tables/Ken/x2_counts7.csv")
d7.to_csv("/home/bmanubay/.thermoml/tables/Ken/mix_counts7.csv")