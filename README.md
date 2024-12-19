# ActiveLearning
Official code for "Pool-based Active Learning with Decision Trees: Incorporate the 
Tree Structure to Explore and  Exploit" 
([M. Schöne](https://www.researchgate.net/profile/Marvin-Schoene), 
[B. Jaster](https://www.researchgate.net/profile/Bjarne-Jaster), 
[J. Bültemeier](https://www.researchgate.net/profile/Julian-Bueltemeier-2),  
J. Köster, C. Holst, M. Kohlhase (Accepted for IEEE SSCI'25) \[[pdf]()\]


# Installation
To start working on `guide_active_learning` clone the repository from GitHub and set up
the development environment

```shell
git clone https://github.com/bjaster/GUIDE-AL.git
cd guide_active_learning
python -m pip install --user virtualenv (if not installed)
virtualenv .venv
source .venv/bin/activate (on Linux) or .venv\Scripts\activate (on Windows)
pip install .
```

# Run the Experiment
To reproduce the results of our paper, run the `benchmark_dataset_script.py` file located in the `scripts` folder:
```
python scripts/benchmark_dataset_script.py
```
This also generates plots of the results, but if needed the results of a run are saved under `guide_active_learning/Data/result` for further investigation.

## Settings
To change run settings have a look at the `settings.ini` file located in the same folder.
There you can adapt the following settings:
- `benchmark_name`: name of the run (for saving purposes)

- `num_benchmark`: number of runs per active learner per dataset

- `datasets`: a list of the used datasets

- `initial_datapoints`: how many datapoints should be queried randomly at the beginning of each run

- `max_depth`: Max depth of the GUIDE tree (used for regularization)

- `min_info_gain`: Minimal information gain to allow a split (used for regularization)

- `ensemble_size`: Size of the Random Forest or GUIDE Ensemble used for the active learning process

- `parallel_computation`: How many runs should be carried out in parallel

- `active_learning_steps`: The amount of datapoints queried after the initial datapoints

- `active_learning_methods`: a list of the used active learning methods (`qbc_exex_rf` is our proposed method GUIDE-AL)
