
## Organization of my project
Each task folder contains these files: output_A, dataset.py, evaluator.py, medmnist.json, models.py, task_A.py. 
In addition, there is also a main.py that can call the function in task A and task B respectively to execute the corresponding task and return the running result.

## The role of each file
1. output_A：This file stores the model checkpoints during the training process, so that the model's state can be easily loaded later for further training or evaluation.
2. dataset.py：This file initializes, splits and preprocess the dataset.
3. evaluator.py: This file defines several functions for evaluating and saving classification results.
4. medmnist.json: This file provides some information of the "PathMNIST" dataset.
5. models.py: This file defines the model structure of ResNet, including two main types of residual blocks: BasicBlock and Bottleneck. 
6. task_A or task_B: This file calls all the files mentioned above to implement training, validating and testing specified dataset, and it outputs performance of the model.
7. main.py: This file calls the function in task A and task B respectively to execute the corresponding task and return the running result.

## The packages required
tqdm
numpy
torch
torchvision
scikit-learn
pandas
Pillow