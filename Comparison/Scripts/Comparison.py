"""
Compare the overlap of biomarkers from each model
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib_venn import venn2

#%%read data
LGB_top_50_genes = pd.read_csv(snakemake.input.LGB_top_50_genes)
ACE_top_50_genes = pd.read_csv(snakemake.input.ACE_top_50_genes)

LGB_top_50_3m = pd.read_csv(snakemake.input.LGB_top_50_3m)
LGB_top_50_18m = pd.read_csv(snakemake.input.LGB_top_50_18m)
LGB_top_50_24m = pd.read_csv(snakemake.input.LGB_top_50_24m)

ACE_top_50_3m = pd.read_csv(snakemake.input.ACE_top_50_3m)
ACE_top_50_18m = pd.read_csv(snakemake.input.ACE_top_50_18m)
ACE_top_50_24m = pd.read_csv(snakemake.input.ACE_top_50_24m)

Log_3m = pd.read_csv(snakemake.input.Log_3m)
Log_18m = pd.read_csv(snakemake.input.Log_18m)
Log_24m = pd.read_csv(snakemake.input.Log_24m)

#c%%ompare shared biomarker numbers among models for each age group
Log_3m_top_50 = Log_3m[:50]
Log_18m_top_50 = Log_18m[:50]
Log_24m_top_50 = Log_24m[:50]

plt.figure()
venn3((set(LGB_top_50_3m["Top_50_Genes"]), set(ACE_top_50_3m["gene"]), set(Log_3m_top_50["gene"])), set_labels=["LGB", "ACE", "Reg"])
plt.title("Model_comparision_3m")
plt.savefig(snakemake.output.venn_models_3m)
plt.close()

plt.figure()
venn3((set(LGB_top_50_18m["Top_50_Genes"]), set(ACE_top_50_18m["gene"]), set(Log_18m_top_50["gene"])), set_labels=["LGB", "ACE", "Reg"])
plt.title("Model_comparision_18m")
plt.savefig(snakemake.output.venn_models_18m)
plt.close()

plt.figure()
venn3((set(LGB_top_50_24m["Top_50_Genes"]), set(ACE_top_50_24m["gene"]), set(Log_24m_top_50["gene"])), set_labels=["LGB", "ACE", "Reg"])
plt.title("Model_comparision_24m")
plt.savefig(snakemake.output.venn_models_24m)
plt.close()



# %%
