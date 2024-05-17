
# Hair Segmentation
This is an ML model to segment hairs in pictures.

# Replicate
The code used to publish to [Replicate](https://replicate.com/hadilq/hair-segment)
is in the `replicate` directory.

# Training
The Neural network, that is used to crop the part of image that has hair,
is a fine-tuned model based on [Yolo8x-seg](https://github.com/ultralytics/ultralytics).

# Post process
In the post process, it uses [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering),
followed by [BFS](https://en.wikipedia.org/wiki/Breadth-first_search) on similar pixels,
to separate hair pixels.

# Colab
You can play around with the implementation in the [Google Colab](https://colab.research.google.com/drive/1nHIAV7oKPMpWNYgg3wNO9oeKvTfAqbLG?usp=sharing).
Notice, its `.ipynb` file is also saved in this repository if you prefer local Jupyter lab instance.

