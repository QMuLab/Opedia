# README

This is an osteosarcoma single-cell annotation tool based on the scGPT model.

## 1. Environment Preparation
- Ensure scGPT related dependencies are installed from [scGPT GitHub repository](https://github.com/bowang-lab/scGPT)


## 2. Input File Preparation
- Prepare single-cell RNA sequencing data file in [.h5ad] format
- Replace the path in the code at `'custom_file_path'` location

## 3. Model Checkpoint Configuration
- Ensure trained model files exist in the `./ckpt/` directory
- Required files include:
  - [model.pt](./ckpt/model.pt) - Model weights file - Please download this file from [here](https://drive.google.com/file/d/1TNhdCv1sFgZf7c_4dOxP7CBwrTJDAdkq/view?usp=sharing)
  - [vocab.json](./ckpt/vocab.json) - Vocabulary file
  - [id2type.json](./ckpt/id2type.json) - ID to type mapping file

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
- Inference results will be saved in the `./save/` directory as [rst.csv](./save/rst.csv) file
- The CSV file contains classification prediction rankings for each cell

## 7. Notes
- Input [.h5ad] file should conform to standard single-cell data format (cell x gene)
- Ensure gene names are gene symbols
- Output results display cell types ranked by confidence scores
