# Art Classifier
## Neural Network in Keras that classifies artwork to their artists. 
This project was started as a collaboration between [Michael Samon](https://github.com/samon11) and [Adam Forland](https://github.com/AForland) for the purpose of showcasing machine learning to high school students.
 
### Current Artists List
- Andy Warhol
- Bob Ross
- George Seurat
- Jackson Pollock
- Leonardo Davinci
- Michelangelo
- Picasso 
- Vincent Van Gogh


### Functions to Get Started
First thing's first edit the variable `basedir` to the location of the repo. 


#### visualize_transforms()
`visualize_transforms(artist)` 
- __artist__: string, name of artist to randomly select image to view transforms.


#### view_activations()
`view_activations(layer_index, artist, img_index=None)`
 - __layer_index__: int, index of layer to view activations. 
 - __artist__: string, name of artist to input into activations model.
 - __img_index__: int, use only to input same image with each funciton call.
