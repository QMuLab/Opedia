# README

This is an osteosarcoma single-cell annotation tool based on the scGPT model. The model has been fine-tuned on a subset of our integrated single-cell RNA sequencing dataset comprising more than 360,000 osteosarcoma cells. The related manuscript has been submitted. 

## 1. Environment Preparation
- Ensure scGPT related dependencies are installed from [scGPT GitHub repository](https://github.com/bowang-lab/scGPT). 


## 2. Input File Preparation
- Prepare single-cell RNA sequencing data file in [.h5ad] format.
- Replace the path in the code at `'custom_file_path'` location.

## 3. Model Checkpoint Configuration
- Ensure trained model files exist in the `./ckpt/` directory.
- Required files include:
  - [model.pt](https://drive.google.com/file/d/1TNhdCv1sFgZf7c_4dOxP7CBwrTJDAdkq/view?usp=sharing) - Model weights file - Please download this file from Google Drive [here](https://drive.google.com/file/d/1TNhdCv1sFgZf7c_4dOxP7CBwrTJDAdkq/view?usp=sharing).
  - [vocab.json](./ckpt/vocab.json) - Vocabulary file.
  - [id2type.json](./ckpt/id2type.json) - ID to type mapping file.

## 4. Modify Input File Path
In the [inference.py] file, locate the following code:

```python
adata = sc.read_h5ad('custom_file_path')
```

Replace `'custom_file_path'` with the actual input file path.

## 5. Inference
Execute the following command:
```bash
python inference.py
```

## 6. Output Results
- Inference results will be saved in the `./save/` directory as [rst.csv](./save/rst.csv) file.
- The CSV file contains classification prediction rankings for each cell.

## 7. Demo
- Demo input file: `./demo/demo_input.h5ad` (50 cells)
- To run the demo, set the input path in `inference.py` to:

```python
adata = sc.read_h5ad('./demo/demo_input.h5ad')
```

- Run:

```bash
python inference.py
```

- Expected output:
  - A result file `./save/rst.csv`.
  - Each row corresponds to one cell.
  - Each column (`rank_1`, `rank_2`, ...) contains the ranked predicted cell types.
  - The script prints the total runtime in seconds after inference finishes.

- Example output format:

```text
rank_1,rank_2,rank_3,...
MSC,Osteoblast-like,Adipocyte,...
MSC,Osteoblast-like,MDM,...
```

## 8. Notes
- Input [.h5ad] file should conform to standard single-cell data format (cell x gene).
- Ensure gene names are gene symbols.
- Output results display cell types ranked by confidence scores.
## 9. Reference and contact
We finetuned scGPT model for osteosarcoma. The original publication for scGPT is as follows.

`Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature methods, 21(8), 1470-1480.`

The demo input file `./demo/demo_input.h5ad` was downloaded from the Broad Institute Single Cell Portal study [SCP542](https://singlecell.broadinstitute.org/single_cell/study/SCP542/pan-cancer-cell-line-heterogeneity?cluster=tSNE%20Bone%20Cancer&spatialGroups=--&annotation=Cell_line--group--study&subsample=all) and extracted for this demo.

If you encounter any questions in using our model, please contact quanhua.muATpolyu.edu.hk.
