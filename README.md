# nn-training
Code for training neural networks

# Usage
1. Git fork the repo
2. Create a python/conda envrionment with python >= 3.9 (other versions might work, but I personally use python 3.9) and activate it
3. Inside the project root directory, pip install the package locally with the command
```
pip install -r requirements.txt -e .
```
4. The file `src/nn_training/config.ini` inside the project directory shows what paths are used by default. The code saves model checkpoints at the path indicated by the `checkpoints` variable. Tensorboard log files are saved at the `tensorboard` variable. Imagenet is assumed to be located at the `imagenet` variable. Change any of these variables as necessary.
5. Now you can run the code for training models. For example, to train a supervised resnet50 with factorization loss with weight 1.0 using distributed data parallel and also log training information to tensorboard, do
```
python main.py disentangle --co --fw_1.0 -m -t
```
You can see all the general training options available in `src/nn_training/main.py` and the options specific to factorization in `src/nn_training/nn_modules/disentangle.py`.
