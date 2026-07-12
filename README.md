# Sem-DPRL

This repository implements a Point-of-Interest (POI) recommendation framework for next-location prediction. The model learns users' mobility patterns from historical check-in trajectories by jointly modeling temporal influence, spatial proximity, user preference, POI transitions, and regional structures.

---

## 1. Dataset

[Dataset Link](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset?resource=download&select=dataset_TSMC2014_NYC.csv)

### 1.1 Foursquare-NYC

Foursquare-NYC is a Foursquare check-in dataset collected from the New York region.

Each check-in record contains:

```
user_id    timestamp    latitude    longitude    poi_id
```

Example:

```
1001    2012-04-05T18:30:00Z    40.7128    -74.0060    12345
```

where:

* `user_id`: user identifier
* `timestamp`: check-in time
* `latitude`: latitude coordinate
* `longitude`: longitude coordinate
* `poi_id`: point-of-interest identifier

### 1.2 Foursquare-TKY

Foursquare-TKY is a Tokyo-based Foursquare check-in dataset.

The data format is consistent with Foursquare-NYC:

```
user_id    timestamp    latitude    longitude    poi_id
```

The dataset is used to evaluate the model's ability to capture user mobility patterns in different urban environments.

---

### 1.3 Dataset Split

For each user trajectory:

* 80% of check-ins are used for training.
* 20% of check-ins are used for testing.

The split preserves chronological order to simulate real-world next-location prediction.

---

## 2. Evaluation Metrics

The model is evaluated using ranking-based metrics.

### POI Prediction

Metrics:

* Recall@1
* Recall@5
* Recall@10
* MAP

---

## 3. Requirements

Install dependencies:

```bash
pip install -r requirement.txt
```

---

## 4. Project Structure

```
.
├── data/
│   ├── checkins_4sq_nyc_time_offset.txt
│   ├── checkins-4sq-nyc.txt
│   └── poi_semantics_4sq_nyc.txt
│
├── dataset.py
├── dataloader.py
├── network.py
├── trainer.py
├── evaluation.py
├── setting.py
├── requirement.txt
└── README.md
```

---

## 5. Running Experiments

### Train on Foursquare-NYC

```bash
python convert.py
```
```bash
python train.py --dataset checkins-4sq-nyc.txt --offset checkins_4sq_nyc_time_offset.txt --semantic-file poi_semantics_4sq_nyc.txt
```

### Train on Foursquare-TKY

```bash
python convert.py
```
```bash
python train.py --dataset checkins-4sq-tky.txt --offset checkins_4sq_tky_time_offset.txt --semantic-file poi_semantics_4sq_tky.txt
```

Training parameters can be modified in:

```
setting.py
```

including:

* hidden dimension  
* learning rate 
* batch size
* number of epochs
* RNN type
* spatial/temporal decay factors

---

## 6. Citation

If you use this repository or datasets in your research, please cite the corresponding work.

```
@article{
  title={Spatial-Temporal POI Recommendation},
  author={},
  journal={},
  year={}
}
```
