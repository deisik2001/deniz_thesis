import anndata
import pandas as pd
import scanpy as sc
from pytorch_lightning import loggers as pl_loggers

from scvi.model._scvi import SCVI

tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
prior = "vamp"

adata = anndata.read_h5ad("../Data/sim_no_be/data.h5ad")

SCVI.setup_anndata(adata, batch_key="sample_id")

model = SCVI(adata, n_latent=5, n_layers=3, prior_distribution=prior,
             deeply_inject_covariates=False)
model.train(max_epochs=400, check_val_every_n_epoch=1,train_size=0.8, validation_size=0.2, logger=tb_logger)
adata.obsm["latent"] = model.get_latent_representation()

df = pd.DataFrame(adata.obsm["latent"],
                  columns=[f"latent_{i}" for i in range(adata.obsm["latent"].shape[1])],
                  index=adata.obs_names)

sc.pp.neighbors(adata, use_rep="latent")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["sample_id", "celltype"], save=f"{prior}.png")
