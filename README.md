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
model=LoadModel("your_model.h5")<br />
model.predict(X)<br />

Please when you use this package.
__________
Please be aware that the architecture and the formula for color intensity normalization are only provided for review and academic research purposes. The use for other purposes not listed here is not allowed without the permission of the author. Replication, modification or reuse of the software or the template are not allowed without the permission of the author.
