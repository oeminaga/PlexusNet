#This package easily allows using PlexusNet architecture in research project. An example code:

from PlexusNet.architecture import PlexusNet
model=PlexusNet(depth=2, length=3, junction=3, n_class=2)
model_.compile(optimizer=optimizer, metrics=["acc"], loss="categorical_crossentropy")
modle.fit(X, Y)

#if you want to load a model:

from PlexusNet.architecture import LoadModel
model=LoadModel("your_model.h5)
model.predict(X)

Please be aware that the architecture and the formula for color intensity normalization are patented and only provided for review and academic research purposes. The use for other purposes not listed here is not allowed without the permission of the author. Replication, modification or reuse of the software or the template are not allowed without the permission of the author.