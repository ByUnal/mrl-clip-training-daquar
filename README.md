# MRL-based CLIP Training

A PyTorch implementation of Matryoshka Representation Learning (MRL) applied to CLIP for Visual Question Answering (VQA) tasks.

## Overview

This project extends OpenAI's CLIP model with Matryoshka Representation Learning (MRL) layers to improve performance on Visual Question Answering tasks. The model learns nested representations at different granularity levels, enabling better alignment between images and question contexts.

## Features

- **MRL-CLIP Architecture**: Custom CLIP extension with nested linear layers for Matryoshka representation learning
- **VQA Training**: End-to-end training pipeline for Visual Question Answering
- **DAQUAR Dataset Support**: Data loaders for the DAQUAR (Cornell-DAQUAR) dataset
- **Multi-level Analysis**: Tools to evaluate model performance at different representation granularities
- **DDP Support**: Distributed Data Parallel training for multi-GPU setups

## Project Structure

```
├── main.py                      # Main training pipeline
├── main_ddp.py                  # Distributed training entry point
├── mrl_clip_model.py            # MRL-CLIP model architecture
├── mrl_layer.py                 # Custom MRL linear layer
├── train_vqa_model.py           # VQA training logic
├── evaluate_vqa.py              # Evaluation and visualization utilities
├── evaluate_trained_model.py    # Model evaluation script
├── daquar_loader.py             # DAQUAR dataset loader
├── daquar_loader_v2.py          # Improved dataset loader
├── daquar-dataset/              # Dataset directory
├── artifacts/                   # Pre-trained model checkpoints
└── requirements.txt             # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ByUnal/mrl-clip-training-daquar.git
cd mrl-clip-training-daquar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download CLIP model weights (automatic on first run):
```python
import clip
clip.load("ViT-B/32", device="cpu")
```

## Usage

### Training
```bash
python main.py
```

For distributed training on multiple GPUs:
```bash
torchrun --nproc_per_node=<num_gpus> main_ddp.py
```

### Evaluation
```bash
python evaluate_trained_model.py
```

## Dataset

The project uses the DAQUAR dataset with the following structure:
- `images/`: Image files (PNG format)
- `data_train.csv`: Training split with columns: image_id, question, answer
- `data_eval.csv`: Evaluation split
- `answer_space.txt`: Available answer classes
- `all_qa_pairs.txt`: Complete QA pair list

## Model Architecture

- **Backbone**: OpenAI CLIP ViT-B/32
- **MRL Projection**: Custom layer with nested dimensions for Matryoshka representation learning
- **Loss Function**: Matryoshka contrastive loss across different representation levels

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional for GPU training)
- torchvision, torchaudio
- scikit-learn, pandas, matplotlib, tqdm

See `requirements.txt` for exact versions.

## Citation

If you use this project in your research, please cite the following paper:

```bibtex
@article{unal6076940mrl,
  title={MRL-LLaVA: Efficient Visual Question Answering via Matryoshka Representation Learning Based Adaptive Granularity Selection},
  author={Unal, Muhammed Cihat and Demirel, Berkan and Ikizler-Cinbis, Nazli},
  journal={Available at SSRN 6076940}
}
```

Additionally, please cite the original CLIP paper if you use CLIP components in your work.

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2024 Muhammed Cihat Unal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


