# PlexusNet: A neural network architectural concept for medical image classification

This package was built on the Keras framework and easily allows using the PlexusNet architecture for research projects.

- To Install (please download it from the file list above): 
```pip3 install git+https://github.com/oeminaga/PlexusNet.git``` <br />
- Please install tensorflow, tensorflow_addons and tensorflow_probability before using this package or use requirements.txt before starting with install. <br />
```
pip install -r requirements.txt
```

- For reproducibility, please add the function<br />
```
import plexusnet
seed_everything()
```

- An example code to use the package: 
```
from plexusnet.architecture import PlexusNet
model=PlexusNet(depth=2, length=3, junction=3, n_class=2).model

model.compile(optimizer="adam", metrics=["acc"], loss="categorical_crossentropy")
model.fit(X,Y)
```
- If you want to load a PlexusNet model for prediction:
```
from plexusnet.architecture import LoadModel
model=LoadModel("your_model.h5")
model.predict(X)
```

- For NAS (only one epoch for each model configuration), example codes are provided:</br>
-- Grid search</br>(https://github.com/oeminaga/PlexusNet/blob/master/Example_Cifar_PleuxsNET-GridSearch.ipynb)</br>
-- Bayesian Optimization</br> (https://github.com/oeminaga/PlexusNet/blob/master/Example_Cifar_PleuxsNET_BayesianOptimization.ipynb)</br>
-- Hyperband</br> (https://github.com/oeminaga/PlexusNet/blob/master/Example_Cifar_PleuxsNET_Hyperband.ipynb)</br>
-- Random search</br> (https://github.com/oeminaga/PlexusNet/blob/master/Example_Cifar_PleuxsNET_RandomSearch.ipynb)</br>

<br> Please exclude h5 in the filename if you want to save the model during training with TF version >2.8 to avoid the model saving errors! <br>
<br> Please exclude the option "save weight only" to avoid errors when you wanted to load the model. Instead, save the whole model during model training.

<br> Bayesian block is now added (bayesian_inception). </br> 
<br><b>Please cite, when you use this package:</b></br>
O. Eminaga, M. Abbas, J. Shen, M. Laurie, J.D. Brooks, J.C. Liao, D.L.
Rubin, PlexusNet: A neural network architectural concept for medical image classification, Computers in
Biology and Medicine (2023),doi: https://doi.org/10.1016/j.compbiomed.2023.106594.
__________
