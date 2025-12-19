# ML-project-meshcompression
Code repository for research project on machine learning based mesh compression 

# Project setup
After cloning you need to setup the conda environment 
```
conda env create -n meshcompression
```
And need activate it:
```
conda activate meshcompression
```
Install the dependencies: 
**Be aware: very specific versions for torch, torch-geometric, torch-scatter, etc are needed.**
```
pip install -f indiening.txt
```
Compile the faster nndistance 
```
cd src/WrappingNet/nndistance
python build.py install
cd ../../..
```
Install the code as an editable module
```
pip install -e .
```

# Get testing dataset
```
cd src/WrappingNet/datasets
wget https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Manifold40.zip
unzip Manifold40.zip
```


# Creating results
Create csv file from checkpoint  
```
python3 src/WrappingNet/benchmarks/benchmark_autoencoder.py --dataset src/WrappingNet/datasets/Manifold40/raw/ --checkpoint <CHECKPOINT> --latent-dim <LATENT-DIM>
```

Plot hamfer, hausdorff & p2s 
```
python3 src/WrappingNet/results/plot_error.py --folder src/WrappingNet/results/ae_basic_shrec16_test/
```
