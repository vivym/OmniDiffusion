# OmniDiffusion

## Usage

### 1) Installation

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
poetry install
```

### 2) Training

```bash
python -m omni_diffusion fit --config configs/fine_tune_ray.yaml
```
