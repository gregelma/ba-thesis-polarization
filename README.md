# ba-thesis-polarization
**Detecting Multilingual, Multicultural and Multievent Online Polarization**

This repository contains the code, data structure, and experimental results for my Bachelor’s thesis on multilingual polarization detection.  
The work investigates binary polarization detection, polarization type classification, and rhetorical manifestation identification across multiple languages.

---

## Repository Structure

### 01_data
Contains the datasets for all subtasks.

- For **each subtask**, there are two folders:
  - `train/` – **labeled data**, used for training, validation, and testing
  - `dev/` – **unlabeled data**, not used in any experiments, only relevant for predictions

All reported experiments rely exclusively on the labeled data from the `train` folders.

---

### 02_src
Contains the complete source code for the experiments.

- All scripts are designed to be executed **from within the `02_src` directory**
- Modules are executed using Python’s `-m` flag

**Example:**
```bash
cd 02_src
python -m monolingual_transformer.zero_shot_transformer
```
### 03_additional_experiments
Contains exploratory experiments conducted during development. **Not part of the final evaluation.**

---

### 04_results
Contains the outputs of the final experiments reported in the thesis.

- Evaluation results
- Plots and visualizations
- Configuration files and experiment logs

All results in this directory correspond to the experimental setups discussed in the thesis.
