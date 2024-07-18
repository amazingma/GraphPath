# GraphPath: a graph attention model for molecular stratification with interpretability based on the pathway-pathway interaction network
![GraphPath](https://github.com/amazingma/GraphPath/blob/main/Figures/GraphPath.png)
## 📄 Introduction
Achieving accurate and interpretable clinical predictions requires paramount attention to thoroughly characterizing patients at both the molecular and biological pathway levels. In this paper, we present GraphPath, a biological knowledge-driven graph neural network with multi-head self-attention mechanism that implements the pathway-pathway interaction network. We train GraphPath to classify the cancer status of patients with prostate cancer based on their multi-omics profiling. Experiment results show that our method outperforms P-NET and other baseline methods.

## 🚀 Getting Started
### 1. Clone the repo
```
git clone https://github.com/amazingma/GraphPath.git
```
### 2. Create conda environment
```
conda env create --name GraphPath --file=environment.yml
```

## Usage
### 1. Activate the created conda environment
```
source activate GraphPath
```
### 2. Train the model
```
python train.py
```

## References
```
1. Ma T, Wang J. GraphPath: a graph attention model for molecular stratification with interpretability based on the pathway–pathway interaction network[J]. Bioinformatics, 2024, 40(4): btae165.
2. Elmarakeby H A, Hwang J, Arafeh R, et al. Biologically informed deep neural network for prostate cancer discovery[J]. Nature, 2021, 598(7880): 348-352.
```
