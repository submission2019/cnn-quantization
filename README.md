# cnn-quantization

## Dependencies
- python3.x
- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization
- [scikit-learn](https://scikit-learn.org) for kmeans clustering
- [mlflow](https://mlflow.org/) for logging
To install requirements run:
```
pip install torch torchvision bokeh pandas sklearn mlflow
```

## HW requirements
NVIDIA GPU / cuda support

## Data
- To run this code you need validation set from ILSVRC2012 data
- Configure your dataset path by providing --data "PATH_TO_ILSVRC" or copy ILSVRC dir to ~/datasets/ILSVRC2012.
- To get the ILSVRC2012 data, you should register on their site for access: <http://www.image-net.org/>

## Prepare environment
- Clone source code
```
git clone https://github.com/submission2019/cnn-quantization.git
```
- Create virtual environment for python3 and activate:
```
virtualenv --system-site-packages -p python3 venv3
. ./venv3/bin/activate
```
- Install dependencies
```
pip install torch torchvision bokeh pandas sklearn mlflow
```

## Building cuda kernels for GEMMLOWP
To improve performance GEMMLOWP quantization was implemented in cuda and requires to compile kernels.
- build kernels
```
cd kernels
./build_all.sh
```

### Run inference experiments
**Post-training quantization of Res50**<br/><br/>
>*Note that accuracy results could have 0.5% variance due to data shuffling.*

- Experiment W4A4 naive:
```
python inference/inference_sim.py -a resnet50 -b 512 -pcq_w -pcq_a -sh --qtype int4 -qw int4
```
>* Prec@1 62.154 Prec@5 84.252

- Experiment W4A4 + ACIQ + Bit Alloc(A) + Bit Alloc(W) + Bias correction:
```
python inference/inference_sim.py -a resnet50 -b 512 -pcq_w -pcq_a -sh --qtype int4 -qw int4 -c laplace -baa -baw -bcw
```
>* Prec@1 73.330 Prec@5 91.334
<br/>


## AICQ: Analytical Clipping for Integer Quantization

We solve eq. 6 numerically to find optimal clipping value &alpha; for both Laplace and Gaussian prior.<br/>
![eq-6](fig/opt_clipping-eq-6.png)

Numerical solution source code:
[optimal_alpha.ipynb](optimal_alpha.ipynb)
<br/>

## Per-channel bit allocation

Given a quota on the total number of bits allowed to be written to memory, the optimal bit width assignment Mi for channel i is the following.<br/>
![eq-6](fig/bit_alloc-eq-11.png)<br/>

## Bias correction
We observe an inherent bias in the mean and the variance of the weight values following their quantization.<br/>
![bias-err](fig/resnet101_bias_err.png)<br/>
We calculate this bias using equation 12.<br/>
![eq-12](fig/bias-corr-eq-12.png)<br/>
Then, we compensate for the bias for each channel of W as follows:<br/>
![eq-13](fig/bias-corr-eq-13.png)<br/>


## Quantization with optimal clipping
In order to quantize tensor to M bit with optimal clipping we use GEMMLOWP quantization with small modification. We replace dynamic range in scale computation by 2*alpha where alpha is optimal clipping value.

Quantization code can be found here: 
[int_quantizer.py](pytorch_quantizer/quantization/qtypes/int_quantizer.py)
