# wsinsight-train

End-to-end, **headless** CLI to train WSInsight CellViT cell-classification
heads from paired 10x Xenium + H&E. No GUI, no QuPath required.

```bash
pip install wsinsight-train
wsitrain check --input /data/breast            # preflight
wsitrain run   --input /data/breast --tissue breast
```

Pipeline (DAG): `annotate → segment → transfer → tile → split → train → export → report`.
KurtoRank labels Xenium clusters; cells are segmented on H&E (Cellpose by
default, StarDist optional); labels transfer by coordinate join; tiles train a
CellViT head exported to TorchScript.

Input layout:

```
input/<sample>/
  outs/          # raw 10x Xenium bundle
  he.ome.tif     # registered H&E
```
