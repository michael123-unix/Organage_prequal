import numpy as np
import pandas as pd
import anndata as ad

print("loading and preparing data...")

adata = ad.read(snakemake.input.raw_data)

#set layer (model needs a specified input layer)
adata.layers["raw_counts"] = adata.X

#convert NA to np.nan for observations and data, first all cathegories to strings, then string replacement
for i in adata.obs.columns:
    if adata.obs[i].dtype == "category":
        adata.obs[i] = adata.obs[i].astype(str)

adata.obs.replace("NA", np.nan, inplace=True)
adata.obs[adata.obs["cell_ontology_class"]=="NA"]
adata.obs["cell_ontology_class"].count()/len(adata.obs["cell_ontology_class"])

#background and foreground
background = ["cell_ontology_class", "sex", "tissue"]
salient = ["age"]

#background as cathegorical when applicable
for i in background:
    adata.obs[background] = adata.obs[background].astype("category")   
adata.obs[background].dtypes

#age as numeric
adata.obs["age"] = adata.obs["age"].str.extract("(\d+)").astype(int)

#male mice only to reduce dataset size
print("restricting analysis to male mice only due to memory and storage limitations")
adata_male_mice = adata[adata.obs.sex=="male"]

#filter out cells of low abundance
print("exclude cells with abundances < 700")
cell_filter = adata_male_mice.obs["cell_ontology_class"].value_counts()>700
abundant_cells_index = cell_filter[cell_filter]
abundant_cells_index.index.nunique()
abundant_cells = abundant_cells_index.index.tolist()
type(abundant_cells)

adata_male_mice.shape
mask = adata_male_mice.obs.cell_ontology_class.isin(abundant_cells)
adata_male_mice = adata_male_mice[mask, :].copy()
adata_male_mice.shape

#define data for train test split
print("isolating test data as an idependent evaluation set")
train_size = int(adata_male_mice.shape[0]*0.8)
indices = np.arange(adata_male_mice.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

train_data = adata_male_mice[train_idx,:].copy()
test_data = adata_male_mice[test_idx,:].copy()

print("exporting data")
train_data.write(snakemake.output.train_data)
test_data.write(snakemake.output.test_data)

