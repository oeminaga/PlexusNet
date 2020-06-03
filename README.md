# PlexusNet
PlexusNet for medical imaging

# This package easily allows using PlexusNet architecture in research project.

# To Install: pip3 install git+https://github.com/oeminaga/PlexusNet.git

#An example code to use the package: from PlexusNet.architecture import PlexusNet model=PlexusNet(depth=2, length=3, junction=3, n_class=2) model_.compile(optimizer=optimizer, metrics=["acc"], loss="categorical_crossentropy") modle.fit(X,Y)

# if you want to load a model:

from PlexusNet.architecture import LoadModel model=LoadModel("your_model.h5) model.predict(X)

# Please be aware that the architecture and the formula for color intensity normalization are patented and only provided for review and academic research purposes. The use for other purposes not listed here is not allowed without the permission of the author. Replication, modification or reuse of the software or the template are not allowed without the permission of the author.

# Please cite: 
Eminaga O, Abbas M, Kunder C, Loening AM, Shen J, Brooks JD, Langlotz CP, Rubin DL. Plexus Convolutional Neural Network (PlexusNet): A novel neural network architecture for histologic image analysis. arXiv preprint arXiv:1908.09067. 2019 Aug 24.

https://arxiv.org/abs/1908.09067
