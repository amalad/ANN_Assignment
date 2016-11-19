# ANN_Assignment
An implementation of artificial neural networks to perform the following tasks: Face Recognition, Pose Recognition and Sunglasses Recognition

#Preprocessing steps:

For each training and testing list, the corresponding image files were read using a pgm parser (for the P5 format) written in C++ and the pixel values of the images were stored in text files (sorted according to the lists). Each line of a text file contains the pixel values of one image, preceded by a ‘1’. For each of the tasks, namely face recognition, pose recognition and sunglasses recognition, output files were generated for the images in the corresponding lists. Each line of the output text files contains the class of the image in the 1-of-n format. The parser can be found in the file named “parser.cpp”

#Network Structure and Parameters:

Following is a detailed description of the structure of the network used for each task:

1. Face Recognizer: The network used for this task consists of three layers - an input layer, a hidden layer, and an output layer. The input layer has 960 nodes containing pixel values of the image being fed to the network. Both, the hidden layer and output layer consist of 20 nodes each. 
2. Pose Recognizer: The network consists of three layers - an input layer, a hidden layer and an output layer. The input layer has 960 nodes each with a pixel value of the image being fed to the network. The hidden layer consists of 6 nodes and the output layer, of 4. 
3. Sunglasses Recognizer:  The network used for this purpose consists of three layers - an input layer, a hidden layer and an output layer. The input layer has 960 nodes each with a pixel value of the image being fed to the network. The hidden layer consists of 5 nodes and the output layer, of 2. 

#Dataset:

The dataset can be downloaded from this link : http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces_4.tar.Z

The link for the training and testing lists for each task can be found here: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/trainset/

Following are the training and testing lists to be used for each task:

1. Face Recognizer: The training file for this task is “straighteven_train.list” and the testing files are “straighteven_test1.list” and “straighteven_test2.list”.
2. Pose Recognizer: The training file for this task is “all_train.list” and the testing files are “all_test1.list” and “all_test2.list”.
3. Sunglasses Recognizer: The training file for this task is “straightrnd_train.list” and the testing files are “straightrnd_test1.list” and “straightrnd_test2.list”.


#Results:

Following are the maximum accuracies achieved for each task:

1. Face Recognizer: 
2. Pose Recognizer: 
3. Sunglasses Recognizer: 

