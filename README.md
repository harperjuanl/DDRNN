# Dynamic Diffusion Recurrent Neural Networks for Air Quality Prediction

### Requirements
- python 2.7.12
- tensorflow 1.12.0
- numpy 1.14.2
- pandas 0.23.4

### Model architecture
![image](https://github.com/HarperSweet/DDRNN/blob/master/img/model.jpg)

### Data Source
We evaluate our approach on two real-world datasets, KDDCUP 2018 and UrbanAir. You can visit the homepage to get the dataset of KDDCUP 2018 competition and the open data of Urban Computing Lab in JD Group, and should put them into the data/ folder. Here, we use the small dataset(UrbanAir) to represent the effectiveness and reproducibility of our model in the Data/beijing/ folder, which is already processing. 
Run the following commands to generate train/val/test dataset:
'''
python Scripts.gen_tra_data.py
''' 

### Model Training
The model implement mainly lies in "ddrnn_supervisor.py", "ddrnn_model.py", "ddrnn_cell.py" in model/ folder. To train or test our model, please follow the command:
```
python ddrnn_train.py
```
We train the model on a GPU server with GeForce 1080Ti GPU and TensorFlow is considered as the corresponding programming environment. And, more details are being added...
