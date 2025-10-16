# HQA-VLAttack

## Quick Start

### 1. Install dependencies
See in `requirements.txt`.

### 2. Prepare datasets and models
Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in `./configs/Retrieval_flickr.yaml, image_root`.  
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL).
Download the required files, [counter-fitted-vectors.txt](https://drive.google.com/file/d/14Non5yIDaHPUk2TPIskfE-IZ-NiuYjXW/view?usp=drive_link) and [mat_sim_0.4.txt](https://drive.google.com/file/d/1Uk_IFWdfTLn3rwAmdbT5iY5luv06LSyl/view?usp=drive_link) (Place these two files in the data folder)

### 3. Attack evaluation

ALBEF_weight and TCL_weight are the paths to the downloaded weight files.

From ALBEF to TCL on the Flickr30k dataset:
```python
python eval_albef2tcl_flickr.py --config ./configs/Retrieval_flickr.yaml --source_model ALBEF  --source_ckpt ALBEF_weight --target_model TCL --target_ckpt TCL_weight --original_rank_index ./std_eval_idx/flickr30k --scales 0.5,0.75,1.25,1.5
```

From ALBEF to CLIP<sub>ViT</sub> on the Flickr30k dataset:
```python
python eval_albef2clip-vit_flickr.py --config ./configs/Retrieval_flickr.yaml --source_model ALBEF  --source_ckpt ALBEF_weight --target_model ViT-B/16  --original_rank_index ./std_eval_idx/flickr30k --scales 0.5,0.75,1.25,1.5
```

From CLIP<sub>ViT</sub> to ALBEF on the Flickr30k dataset:
```python
python eval_clip-vit2albef_flickr.py --config ./configs/Retrieval_flickr.yaml --source_model ViT-B/16  --target_model ALBEF --target_ckpt ALBEF_weight --original_rank_index ./std_eval_idx/flickr30k --scales 0.5,0.75,1.25,1.5
```

From CLIP<sub>ViT</sub> to CLIP<sub>CNN</sub> on the Flickr30k dataset:
```python
python eval_clip-vit2clip-cnn.py --config ./configs/Retrieval_flickr.yaml --source_model ViT-B/16  --target_model RN101  --original_rank_index ./std_eval_idx/flickr30k --scales 0.5,0.75,1.25,1.5
```
