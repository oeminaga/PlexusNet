# PlexusNet
PlexusNet for medical imaging.

@This package was built on the Keras framework and easily allows using the PlexusNet architecture for research projects.

@To Install: pip3 install git+https://github.com/oeminaga/PlexusNet.git <br />
@Please install tensorflow before using this package. <br />

#An example code to use the package: 

from PlexusNet.architecture import PlexusNet <br />
model=PlexusNet(depth=2, length=3, junction=3, n_class=2).model <br />

model.compile(optimizer="adam", metrics=["acc"], loss="categorical_crossentropy") <br />
model.fit(X,Y)<br />

#if you want to load a PlexusNet model:

from PlexusNet.architecture import LoadModel<br />
model=LoadModel("your_model.h5")<br />
model.predict(X)<br />

__________
Please be aware that the architecture and the formula for color intensity normalization are patented and only provided for review and academic research purposes. The use for other purposes not listed here is not allowed without the permission of the author. Replication, modification or reuse of the software or the template are not allowed without the permission of the author.
