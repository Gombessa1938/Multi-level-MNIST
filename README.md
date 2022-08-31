# Multi-level-Network

This repo has multiple branch for different experiment research code. 

```llnl```, ```llnl_3``` are design for using llnl PDE dataset <br>
```main``` is for using mnist dataset <br>
```mnist_upsample``` is for using mnist dataset but upscaled the image by 2x to create sudo high resolution image <br>

## LLNL related branch

```llnl``` is an experiment branch, where it only does two level training. It start from the small network and load the weight into the large network. 

```llnl_3``` is designed to train with three network, where it start to train small network first then load the weight to medium size network then lastly load both weight into large size network. 

To run experiment : 
```
python main.py
```
the module will average the accuracy and time of 10 run.

## MNIST related branch

```main``` is more complete branch, which use three network to train, very similar to the ```llnl_3``` branch, which it trains a small network first then load weight to medium network then to large network. 

Run
```
python dataset.py
python main.py
```
to create the dataset needed for the experiment. After running, ```dataset.py``` will create 3 ```dataset.npy``` file in the same folder and you will need to change the ```config.py``` data address before you run ```main.py```.

```mnist_upsample``` branch basically play with the similar experiment ``` mnist_full_img```, however it just upsample the image by 2x to create a sudo higher resolution. To use it, it's the same as above.
```
python dataset.py
```
change .npy file address in ```config.py``` <br>
```
python main.py
```




