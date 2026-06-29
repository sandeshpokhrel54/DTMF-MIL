# KMeans and SDF/RBF Preprocessing

This folder contains the preprocessing pipeline used by DTMf-MIL.

1. Fit KMeans using only the training slides for each CV fold.
2. Assign every train/val/test patch in that fold to one of the fold-specific KMeans clusters.
3. Build a tissue grid from patch coordinates.
4. For each cluster, compute a signed distance field, RBF channels, SDF gradients, and local SDF statistics.
5. Save one `*_sdf70.npz` file per slide.

The saved `spatial_features` array has shape `[num_patches, 70]`, corresponding to `5 clusters x 14 features`.

Use `cv_kmeans.py` and `cv_extract_sdf.py` for the generic feature layout. Use the `_hamid.py` variants for the Hamid VA feature folder layout.
