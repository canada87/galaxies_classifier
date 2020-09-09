# galaxies_classifier
Taking a large image with galaxies of different kind. The aim of the project is to detect the position of the galaxies in the image and the classify those galaxies in circular or elliptical.

The Transfer learning with Mask-R-CNN has been used to achieve the objective.

A set of fake galaxies are generate and label and fed to the MRCNN as training and test set.

Once the MRCNN is trained is then used to find the galaxies in the real image.

# Files
galaxy_generator.py read the real image and produce a large number of fake images using the real one as blueprint.

MRCNN.py read the fake images to train and then read the real one to predict
