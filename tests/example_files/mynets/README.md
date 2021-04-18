# NOTE

- The code for these models is copied almost 1:1 form torchvision (version 0.8.2)
- **Why did we copy this code instead of just using the models from the library?**
    - for our experiments the used models are just examples, we want to cover the case of custom models developed using
      pytorch
    - to give us more flexibility throughout the experiments we copied the models
        - this is for example helpful when we have to store the model we are currently using in the DB
        - in this case it would be unrealistic to just load it from the torchvision library
      