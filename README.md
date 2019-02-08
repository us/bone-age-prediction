
## File Structure
```
.
+-- flow_dataframe.py	
+-- train.py
+-- rsna-bone-age      <-------------- downloaded dataset
|   +-- boneage-test-dataset.csv
|   +-- boneage-train-dataset.csv
|   +-- boneage-train-dataset
|   |   +-- boneage-train-dataset
|   |   |   +-- 1377.png
|   |   |   +-- 1378.png
|   |   |   +-- ...
|   +-- boneage-test-dataset
|   |   +-- boneage-test-dataset
|   |   |   +-- 4360.png
|   |   |   +-- 4361.png
|   |   |   +-- ...

```

## Dataset
The dataset is release to [RSNA Pediatric Bone Age Machine Learning Challenge](https://pubs.rsna.org/doi/10.1148/radiol.2018180736). 
That is consisting of 14 236 hand radiographs (12 611 training set, 1425 validation set, 200 test set) 

## Model Structure (16Bit)
<img src="https://www.16bit.ai/static/img/blog/rsna/architecture.png" alt="drawing" width="600"/>

## Prerequisites

- Python 3.6+
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Sklearn](https://scikit-learn.org/)

## Usage

1. Clone this repository.

2. Download images of 2017 RSNA Bone Age Challenge Dataset from this [kaggle page](https://kaggle.com/kmader/rsna-bone-age) and decompress them to the directory. Or download with kaggle-api


    ```kaggle datasets download -d kmader/rsna-bone-age```

3. Setting up your own parameters and run

   `python train.py`

## to-do's
- [ ] Adding test.
- [ ] Add model's accurancy table.
- [ ] Rewrite with pytorch.
