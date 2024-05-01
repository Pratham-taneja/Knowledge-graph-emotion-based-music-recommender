

# Emotion-Driven Music Recommendations: A Novel Approach Using Multi-Modal Knowledge Graph Convolutional Networks
## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
- Pytorch == 1.1.0
- torch-cluster == 1.4.2
- torch-geometric == 1.2.1
- torch-scatter == 1.2.0
- torch-sparse == 0.4.0
- numpy == 1.16.0


- `aggr_mode` 
  It specifics the type of aggregation layer. Here we provide three options:  
  1. `mean` (by default) implements the mean aggregation in aggregation layer. Usage `--aggr_mode 'mean'`
  2. `max` implements the max aggregation in aggregation layer. Usage `--aggr_mode 'max'`
  3. `add` implements the sum aggregation in aggregation layer. Usage `--aggr_mode 'add'`
  
  
- `concat`:
  It indicates the type of combination layer. Here we provide two options:
  1. `concat`(by default) implements the concatenation combination in combination layer. Usage `--concat 'True'`
  2. `ele` implements the element-wise combination in combination layer. Usage `--concat 'False'`

||#Interactions|#Users|#Items|Visual|Acoustic|Textual|
|:-|:-|:-|:-|:-|:-|:-|

-`train.npy`
   Train file. Each line is a user with her/his positive interactions with items: (userID and micro-video ID)  
-`val.npy`
   Validation file. Each line is a user several positive interactions with items: (userID and micro-video ID)  
-`test.npy`
   Test file. Each line is a user with several positive interactions with items: (userID and micro-video ID)  

### Running the code

$ python main.py --dataset music-d (note: use -h to check optional arguments)


## Usage

# Navigate into the Directory
cd Emotion-Aware-Music-Recommendation-System/

# Run the Application
streamlit run App.py


## Dependencies

* To run this application, a Web-browser should have been installed in the machine. Please allow the application to access the Web-Cam to capture faces.
* Python3 must be installed.
* The following additional packages should also be installed.

# Install Pip - Package installer for Python
sudo apt-get -y install python3-pip

# Install Streamlit
sudo pip3 install streamlit

# Install FER module
sudo pip3 install fer

# Install tensorflow
sudo pip3 install tensorflow

# Install sklearn
sudo pip3 install sklearn

Connect Neural Network Playground with your own system using Tensorflow API and write code in Jupyter Notebook. 
Use this setting to configure results 
Parameter 	Setting 
Training epoch 	 400
Activation  	ReLU
Optimizer 	Adam
Learn rate 	0.03
Regularization 	L1
 
 Parameter 	Setting 
Training epoch 	 600
Activation  	Sigmoid
Optimizer 	Adam
Learn rate 	0.01
Regularization 	L2

