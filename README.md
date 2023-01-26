# PlexusNet
PlexusNet for medical imaging.

@This package was built on the Keras framework and easily allows using the PlexusNet architecture for research projects.

@To Install: pip3 install git+https://github.com/oeminaga/PlexusNet.git <br />
@Please install tensorflow, tensorflow_addons and tensorflow_probability before using this package. <br />

For reproducibility, please add the function<br />
import plexusnet<br />
seed_everything()<br />

#An example code to use the package: 

from plexusnet.architecture import PlexusNet <br />
model=PlexusNet(depth=2, length=3, junction=3, n_class=2).model <br />

model.compile(optimizer="adam", metrics=["acc"], loss="categorical_crossentropy") <br />
model.fit(X,Y)<br />

#if you want to load a PlexusNet model:

from plexusnet.architecture import LoadModel<br />
model=LoadModel("your_model.h5") #Only TF 2.8. Any version above 2.8, please exclude .h5 <br />
model.predict(X)<br />

<b>Please cite, when you use this package:</b>
O. Eminaga, M. Abbas, J. Shen, M. Laurie, J.D. Brooks, J.C. Liao, D.L.
Rubin, PlexusNet: A neural network architectural concept for medical image classification, Computers in
Biology and Medicine (2023),doi: https://doi.org/10.1016/j.compbiomed.2023.106594.
__________
Please be aware that the architecture and the formula for color intensity normalization are only provided for review and academic research purposes. The use for other purposes not listed here is not allowed without the permission of the author. Replication, modification or reuse of the software or the template are not allowed without the permission of the author.
