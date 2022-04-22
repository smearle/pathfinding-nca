# Future Locations
- Have an option to use skip connection or not. i.e., the initial maze is concatenated with the model's hidden activations after each layer.
- Add MLP 
- Option to treat model as CA (with shared weights) or vanilla (distinct weights at each layer)
  - CA: model is 1 or 2 layers, training loop repeatedly passes input (and maybe computes loss part-way through). (This is what we're doing now).
  - vanilla: model is many distinct layers, training loop passes input only once.
- Apply loss at different intervals.
  - How will this mesh with model variants above?