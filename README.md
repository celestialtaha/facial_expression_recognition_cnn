# facial_expression_recognition_cnn
this was my Final Project for Introduction to computer vision course.
## we were supposed to write code in order to :
1. Train a cascade detector to Detect faces in images
2. After detecting faces, feed them into Convolutional neural network to classify facial expressions
3. the clssification has to be done in at least 3 of the following categories:
neutral - happy - surprised - sad - disgusted - angry
# Procedure
## 1.Cascade Detection
1.1 for cascade detection of faces there are pre-trained models avialable which are pretty good and you can download and use it.In this case you can skip Cascade detection.

1.2.I downloaded fer2013 data set from kaggle. link : https://www.kaggle.com/deadskull7/fer2013

1.3. put negative and positive data into separate directories
	*The negative and positive samples should be of 
	same size.
  
1.4.list image files in negative directory and write them on a file using below script:
find ./res_negative -iname "*.jpg" > bg.txt
also do this for positive images and list the sample directories into positives.dat

1.5.the face lacations in each image also should be present in the file we created above in this format
image address [tab] number_of_faces(roi) [tab] face_coordinates:
ex:./positive/yaleB30_P00A+070E+45.pgm	1	0 0 168 192
so,I wrote a script to do that.(refer to project files)

1.6.Now we need to create positive sample vectors:
the width and height params should have the same ratio as original images
	opencv_createsamples -info positives.dat -vec p.vec -num 500 -w 21 -h 24
 
 1.7.After All , we are able to train cascade detector using:

opencv_traincascade -data haar2 -vec p.vec -bg bg.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -mode ALL -numPos 500 -numNeg 1000 -w 21 -h 24

you can learn more about training parameters here : http://note.sonots.com/SciSoftware/haartraining/document.html
## 2. Facial expression recognition CNN
for facial expression recognition I used MiniXception architecture which after 100 epochs achieved 65% accuracy.compared to winner of the kaggle challenge in 2013 who achieved 71% its quite good i think.

for more information about MiniXception refer to : https://arxiv.org/pdf/1710.07557.pdf
