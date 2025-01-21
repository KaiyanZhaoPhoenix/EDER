# Efficient Diversity-based Experience Replay (EDER)

This repository contains the code for the **IJCAI 2025 paper**:  
**"Efficient Diversity-based Experience Replay for Deep Reinforcement Learning"**

---

## Overview

Efficient Diversity-based Experience Replay (EDER) introduces a novel experience replay mechanism to address the limitations of existing methods in **high-dimensional environments** and **sparse reward settings**. By employing **Determinantal Point Processes (DPP)** to model sample diversity, **Cholesky decomposition** to reduce computational complexity, and **rejection sampling** to prioritize diverse samples, EDER enhances the efficiency and performance of reinforcement learning agents in complex scenarios.  



---

## Experimental Results

### Results of Habitat Environment:
![Eder RGB View](./figs/eder_rgb_view.gif)
![Eder Depth View](./figs/eder_depth_view.gif)
![Eder Semantic View](./figs/eder_semantic_view.gif)
![Eder Topdown View](./figs/eder_topdown_view.gif)

#### DDPG
![DDPG RGB View](./figs/ddpg_rgb_view.gif)
![DDPG Depth View](./figs/ddpg_depth_view.gif)
![DDPG Semantic View](./figs/ddpg_semantic_view.gif)
![DDPG Topdown View](./figs/ddpg_topdown_view.gif)

#### DDPG + HER
![Her RGB View](./figs/her_rgb_view.gif)
![Her Depth View](./figs/her_depth_view.gif)
![Her Semantic View](./figs/her_semantic_view.gif)
![Her Topdown View](./figs/her_topdown_view.gif)

#### DDPG + TER
![Ter RGB View](./figs/ter_rgb_view.gif)
![Ter Depth View](./figs/ter_depth_view.gif)
![Ter Semantic View](./figs/ter_semantic_view.gif)
![Ter Topdown View](./figs/ter_topdown_view.gif)

#### DDPG + LoBER
![Lober RGB View](./figs/lober_rgb_view.gif)
![Lober Depth View](./figs/lober_depth_view.gif)
![Lober Semantic View](./figs/lober_semantic_view.gif)
![Lober Topdown View](./figs/lober_topdown_view.gif)

#### DDPG + Relo
![Relo RGB View](./figs/relo_rgb_view.gif)
![Relo Depth View](./figs/relo_depth_view.gif)
![Relo Semantic View](./figs/relo_semantic_view.gif)
![Relo Topdown View](./figs/relo_topdown_view.gif)



---

## Installing Dependencies

### Step 1: Clone the Repository
```bash
git clone --recurse-submodules [EDER]
cd EDER
```

### Step 2: Set Up the Environment
Create and activate a new conda environment:
```bash
conda create -n eder python=3.9
conda activate eder
```

Install the basic requirements:
```bash
pip install -r requirements.txt
```

### Step 3: Install Habitat and Dependencies
Follow the official [Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab) to set up `habitat-lab` and `habitat-sim`.  

Install Habitat dependencies:
```bash
conda install habitat-sim withbullet -c conda-forge -c aihabitat
cd src
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
pip install -e habitat-baselines
```

### Step 4: Download Required Datasets
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
```

---

## Running Experiments

### Step 1: Verify Installation
Run an example script to ensure proper setup:
```bash
python src/habitat-lab/examples/example.py
```

### Step 2: Train with EDER
To train using EDER, navigate to the `habitat-lab` directory and execute:
```bash
cd src/habitat-lab
python eder_train.py
```

### Step 3: Distributed Training (Optional)
For large-scale distributed training across multiple GPUs (e.g., on a Slurm cluster):
```bash
sbatch multi_node_training.sh
```

---

## Citation
If you find this repository helpful in your research, please cite our **IJCAI 2025 paper**:

```
@inproceedings{anonymous2025eder,
  title={Efficient Diversity-based Experience Replay for Deep Reinforcement Learning},
  author={Anonymous},
  booktitle={Proceedings of the 34th International Joint Conference on Artificial Intelligence},
  year={2025}
}
```

---

## Acknowledgements

This repository leverages code and resources from the Habitat ecosystem:
- [Habitat Lab](https://github.com/facebookresearch/habitat-lab)
- [Habitat Sim](https://github.com/facebookresearch/habitat-sim)

We acknowledge and thank the contributors for their efforts.  

--- 
