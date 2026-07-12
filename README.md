# DPRL [IJCAI 2025]

This is the implementation for the paper: “Disentangled and Personalized Representation Learning for Next Point-of-Interest Recommendation.” 

## Preliminaries

### Requirements
```
pip install -r requirements.txt
```
### Data Preparation
see [REPLAY](https://github.com/UM-Data-Intelligence-Lab/REPLAY).

## Model Training

```
python train.py --dataset checkins-gowalla.txt --hidden-dim 30 --weight_decay 1e-5
python train.py --dataset checkins-4sq.txt --hidden-dim 20 --weight_decay 1e-6
```


## Acknowledgement

The code is implemented based on [Flashback](https://github.com/eXascaleInfolab/Flashback_code).

<!-- ## Citing -->

<!-- If you use DPRL in your research, please cite the following [paper](https://ieeexplore.ieee.org/document/10772008):
```
@article{DBLP:journals/tkde/RaoJSCHYK25,
  author       = {Xuan Rao and
                  Renhe Jiang and
                  Shuo Shang and
                  Lisi Chen and
                  Peng Han and
                  Bin Yao and
                  Panos Kalnis},
  title        = {Next Point-of-Interest Recommendation With Adaptive Graph Contrastive
                  Learning},
  journal      = {{IEEE} Trans. Knowl. Data Eng.},
  pages        = {1366--1379},
  year         = {2025}
}
``` -->
