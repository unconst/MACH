# MACH

Training bigger models over more iterations is almost always a better when solving some key problems in Machine Learning, for instance image and speech.

It is almost always the case that we can improve the performance of a Machine Learning system by leveraging more computation because it allows us to training bigger models over more iterations.

However, as the size of our models get larger, we encounter a locking problem caused by the direct interdependence of many components. Specifically to computer the gradient for a weight in the first layer I must first compute the gradient for every computer in between. This is prohibatively slow as our models become incredibly deep, for instance, as we scale them too biological scale.

Our solution is to limit the direct interdependence of all parts of the network by splitting it into smaller standalone components outfitted with their own loss and access to a dataset. Each section is training against its own loss function using the dataset provided. 
