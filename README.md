# Dimension Parsing using Transformers
![There should be an image here](src/kannan_shrank.png)
This is a simple experiment to extract the dimensions of components from the part description.
In the context of supply chain and manufacturing, there are often procurement data without sufficient metadata to each of the components.

### Acknowledgment

I would like to acknowledge [@karpathy](https://github.com/karpathy/) for being an amazing human being and providing high quality education about deep learning and LLM's. So most of the Transformer's code is inspired from [this repo](https://github.com/karpathy/build-nanogpt).

For example, properties like dimension, would be available but they will be buried in the part description as follows:

| Description            | Height | Width |
|------------------------|--------|-------|
| Plate - TAPER   59X16  | 59     | 16    |
| Plate - STUD     46X22 | 46     | 22    |
| Bar - 28X64 - STUD     | 28     | 64    |
| Bar - HEX HD 39X66     | 39     | 66    |
| Bar - HEX HD 96X28     | 96     | 28    |
| Plate  Rect   SST 37X72| 37     | 72    |
| Plate - STUD     39X11 | 39     | 11    |

Eventhough this table is oversimplified, there are business use cases where infering the dimension from inconsistent data can be challenging beyond using regular expressions.
This repo utilizes Transformers to use an encoder network to predict the height and width out of the part description. 


Although the data is grossly oversimplified, it captures a simple case almost perfectly and shows the power of Transformers and Language models in general. Implementing this helped me gain better insights about the architecture of transformers and LLM's in general.

### Installing

1. Run ```pip install -r requirements.txt```.
2. Run ```python data_gen.py``` to generate data.
3. For training use ```python train.py```.


### Future steps
- Build a decoder network to generate complex dimensional information.