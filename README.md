# Towards a Self-Driving Trigger at the LHC: Adaptive Response in Real Time

## Repository Developers
- [Shaghayegh Emami: shghygh@umich.edu]
- [Giovanna Salvi: giovanna.salvi@cern.ch]
- [Zixin Ding: zixin@uchicago.edu]

  
## Datasets

These datasets are derived from the **CMS 2016 Open Data** for Level-1 (L1) hadronic objects (jets).  
Each file contains reconstructed jet features and the **number of primary vertices ($N_{PV}$)** per event, which serve as the core observables for our anomaly detection and control algorithm studies. The gradual decrease of $N_{PV}$ over time reflects the drop in luminosity and pileup as the fill progresses.

Two main dataset categories are included:

- **Base physics samples:** used for training and evaluating Anomaly Detection (AD) models and for benchmarking control algorithms.  
- **Trigger_food datasets:** precomputed control-variable files containing **Anomaly Scores** and **Hadronic Transverse Momentum (HT)** for each event, generated using our trained AD model to accelerate trigger-control experiments.

| File | Description | Usage |
|------|--------------|-------|
| **`MinBias_1.h5`** | Minimum-bias Monte Carlo (MC) Simulated sample with reconstructed jets and NPV. | Only Used for Anomaly Detection **training** (MC Simulated background) for Autoencoder. |
| **`MinBias_2.h5`** | Alternate minimum-bias MC background sample. | Used for **control-algorithm studies** (MC-only). |
| **`TT_1.h5`** | MC signal sample for **Standard Model** hadronic decay of the $\(t\bar{t}\)$ process. | Simulated **Standard Model** signal sample |
| **`HToAATo4B.h5`** | MC signal sample for **Beyond Standard Model** process $\(H \rightarrow AA \rightarrow 4b\)$. | Simulated **Beyond Standard Model** signal sample |
| **`data_Run_2016_283876.h5`** | Real CMS 2016 run with reconstructed jets and NPV. | Used for **training** the AD model with real-data background. |
| **`data_Run_2016_283408_longest.h5`** | Longest CMS 2016 real-data run. | Used for **control-algorithm testing** with real-data background. |
| **`Trigger_food_MC.h5`** | Precomputed **control-variables dataset** (MC): includes anomaly scores, HT, and NPV for each event across multiple MC processes. | Used for fast control-algorithm studies with **MC** background. |
| **`Trigger_food_Data.h5`** | Precomputed **control-variables dataset** (real data): includes anomaly scores, HT, NPV, and matched MC signal + real background (matched by NPV). | Used for fast control-algorithm studies with **real-data** background. |

> **Notes:**  
> • Some datasets are reserved exclusively for control-algorithm benchmarks to avoid overlap with AD model training.  
> • “Trigger_food” files store pre-evaluated anomaly scores and kinematic variables, reducing runtime for repeated experiments.  
> • All datasets originate from **CMS 2016 Open Data**.

> • Trigger_food_MC.h5 combines event information from MinBias_2.h5, TT_1.h5, HToAATo4B.h5.
> • Trigger_food_Data.h5 combines event information from data_Run_2016_283408_longest.h5, TT_1.h5, HToAATo4B.h5.
---

### Dataset Link

All datasets (base samples and precomputed control-variable files) are publicly hosted on **Zenodo**:

➡️ **Zenodo Record:** [https://zenodo.org/records/17399948?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwZmU5ZDg3LTYxMTYtNGE5OC05M2ZlLTQ5ZjdmYjE2NDRkMyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNTQzMjkyYWVlMTQ2ZDE0NmI5MGIyZGFkYzFlN2VkZSJ9.rl-hT8qA2Og1SAncUUlR-98JWpI5FreQ9YOcwsZ5_utfP2Y8mHLYDXxDC5ErF-cxb2AS-6xQjBJx6ynofYVkeQ](https://zenodo.org/records/17399948?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwZmU5ZDg3LTYxMTYtNGE5OC05M2ZlLTQ5ZjdmYjE2NDRkMyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNTQzMjkyYWVlMTQ2ZDE0NmI5MGIyZGFkYzFlN2VkZSJ9.rl-hT8qA2Og1SAncUUlR-98JWpI5FreQ9YOcwsZ5_utfP2Y8mHLYDXxDC5ErF-cxb2AS-6xQjBJx6ynofYVkeQ)

> The record includes `.h5` files for all datasets listed above.  

---

### Example: Load dataset in Python
```python
import h5py
import pandas as pd

with h5py.File("Trigger_food_MC.h5", "r") as f:
    print("Available keys:", list(f.keys()))
    df = pd.DataFrame({
        "HT": f["HT"][:],
        "NPV": f["NPV"][:],
        "score": f["anomaly_score"][:],
        "process": [p.decode() for p in f["process"][:]],
    })
    print(df.head())
```
## Setup
```
# clone
git clone https://github.com/Shaghayegh-E/Adaptive-ParticlePhysics-Triggers.git
cd Adaptive-ParticlePhysics-Triggers

# create env (recommended)
conda create -n jfti python=3.9 -y
conda activate jfti

# install required packages
pip install -r requirements.txt
```

## File Structure
Download data from Zenodo: All `.h5` datasets can be downloaded from the public Zenodo record:
After downloading, place them under the following structure.

```text
Adaptive-ParticlePhysics-Triggers/
├── Data/                     # Place downloaded .h5 datasets here (from Zenodo and should be ignored by Git)
│   ├── MinBias_1.h5
│   ├── MinBias_2.h5
│   ├── TT_1.h5
│   ├── HToAATo4B.h5
│   ├── data_Run_2016_283876.h5
│   ├── data_Run_2016_283408_longest.h5
│   ├── Trigger_food_MC.h5
│   └── Trigger_food_Data.h5
│
├── SampleProcessing/         
│   ├── ae/                   # Autoencoder models & training scripts & building autoencoders for Anomaly Detection Algorithm. Data Samples: Data/MinBias_1 Data/HToAATo4B.h5 Data/TT_1.h5
│   │   ├── data.py
│   │   ├── experiment_testae.py #Only uses Data/MinBias_1.h5 for training.
│   │   ├── losses.py
│   │   ├── models.py
│   │   └── plots.py
│   │
│   ├── derived_info/         # Build Data/MinBias_2.h5, Data/HToAATo4B.h5, Data/TT_1.h5, models/autoencoder_model0_1.keras, models/autoencoder_model0_4.keras -> Data/trigger_food_MC (monte carlo samples)
│   │   ├── build_trigger_food.py
│   │   ├── data_io.py
│   │   ├── preprocess.py
│   │   └── scoring.py
│   ├── models/               #saving trained autoencoders with dimension = 1 or 4
│   │   ├── autoencoder_model0_1.keras #autoencoder with dimension = 1
│   │   ├── autoencoder_model0_4.keras #autoencoder with dimension = 4
│   
├── Control/   #Running single trigger / Local Multi Trigger and Multi Path trigger
│   ├── agents.py
│   ├── mc_localmulti.py
│   ├── mc_multipath.py
│   ├── mc_singletrigger_io.py
│   ├── mc_singletrigger_plots.py
│   ├── mc_singletrigger.py
│   ├── summary.py
│   └── metrics.py
│── RL/ # Running RL algorithms 
│  
├── outputs/                  # Generated plots & results (create your own outputs folder to store the plots)
├── controllers.py                  
├── triggers.py                  
└── README.md
```
## Step 1 Training Autoencoder
### Training Autoencoder with dimension = 1 or 4 
```
python3 -m SampleProcessing.ae.experiment_testae --dims=1
python3 -m SampleProcessing.ae.experiment_testae --dims=4
```
## Step 2 Building Trigger_food_MC
### Building Trigger_food_MC.h5 (building Monte Carlo Samples and store it under Data folder)
```
python3 -m SampleProcessing.derived_info.build_trigger_food
```
## Step 3 Choose different agents for Trigger Control (Control-only framework)
### Multi Trigger Control Framework Case 1/2/3
```
python3 -m Control.mc_multipath --agent v1
python3 -m Control.mc_multipath --agent v2
python3 -m Control.mc_multipath --agent v3
```
### A Simple Local Controller Case 1/2/3
```
python3 -m Control.mc_localmulti --agent v1 \                   
    --path Data/Trigger_food_MC.h5 \
    --outdir outputs/case1

python3 -m Control.mc_localmulti --agent v2 \                   
    --path Data/Trigger_food_MC.h5 \
    --outdir outputs/case2
python3 -m Control.mc_localmulti --agent v3 \                   
    --path Data/Trigger_food_MC.h5 \
    --outdir outputs/case3
```
### Single-path demo (PD controller on HT & AS)
```
python3 -m Control.mc_singletrigger
python3 -m Control.mc_singletrigger_plots 
```
## Step 4 Generate Summary Plots
### Summary of different agents’ performance on simulation samples with autoencoder dimension = 1/4.

```
python3 -m Control.summary --dim=4
python3 -m Control.summary --dim=1
```

