# PCB-Failure-Analyser

We have built a full PCB Failure Analyser that automates the task of detecting and classifying defects in PCB which will reduce human error percentage. The system is able to detect, extract and predict the type of defect in PCB. Comparatively analysed various model architectures on the DeepPCB dataset through transfer learning techniques.


## Requirements

This code requires you have PyTorch and FastAi installed. Please see the requirements.txt file. To ensure you're up to date, run:

pip install -r requirements.txt

## Getting the data
The data set we used was DeepPCB dataset, a dataset that contains 1500 image pairs each comprising a defect-free template image and an aligned checked image with annotations, including locations of six of the most common PCB flaws. 

To download the DeepPCB dataset into the data folder:

git clone https://github.com/tangsanli5201/DeepPCB.git

The folders are available inside the PCBdata folder.

Now you can run the pcb_start.ipynb scripts till the data extraction part mentioned to extract the images and zip it in one folder for the purpose of annotation.

Any open source software can be used for image annotation.

Once the annotations are done, run the remaining blocks in pcb_start file to prepare the CSV's for training.


## Training models

Run the pcb_train_test.ipynb according to the model you want to load, train and test.


## Demo/Using models

Run the pcb_inference file and specify the images path that you want to test.
