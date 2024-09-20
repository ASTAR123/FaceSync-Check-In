# FaceSync-Check-In(Real-time-check-in-system-based-on-face-recognition)
Real-time check-in system based on face recognition
# File description:

Facedata [Folders]----  source of dataset(Data set files cannot be disclosed)

Facedata-2 [Folders]---- Dataset after data enhancement

saved_model [Folder]----  source of loading models

project [ipynb]  ----  Main code for data enhancement, PCA, MLP and real-time recognition programs

attendanc [xlsx]  ----  the attendance assign list form

bg1 [qrc] & bg1_rc [py]  ----  the source of backgound in GUI

FACE [py]  ----  Part of the main project code that will be used for the check-in system is converted into a python file

faceRecg [py]  ----  source code of GUI and please run it as main programme

Data set files can be replaced. Put it in the Facedata folder.

# Run

Start by running the project.ipynb file. Perform tasks such as data processing, data enhancement, and then model training. Then run the faceRecg.py file for real-time check-in detection, replacing the file's model path file with the result you got. If you do not want to train the model, you can directly use the model in saved_model. best_model_final_2024-02-28_15-27-36.keras is the optimal model

# 1.1 Background Introduction
Facial recognition stands as a pivotal biometric technology, exerting
significant influence in contemporary society. With the ongoing advancements
in machine learning, facial recognition systems find extensive applications
across security, identity verification, intelligent surveillance, and various other
domains. This project involves the design and implementation of a facial
recognition system utilizing Principal Component Analysis (PCA) methodology.
Leveraging machine learning techniques such as Multi-Layer Perceptron
(MLP) algorithms, the focus is on creating an advanced facial recognition
system. The primary objective is to automate facial recognition and
classification tasks with high precision and efficiency.
# 1.2 Objectives
Our objective is to design a sign-in system, which can effectively solve
these problems. Through face recognition technology, students can
automatically identify and sign in through the camera without manual
operation, saving time and labor costs. At the same time, the face recognition
system has high accuracy and reliability, effectively preventing fraud and
errors in sign-in data. In addition, the system can record students' sign-in
status in real time and provide real-time data analysis and management
functions to meet the timeliness requirements for sign-in data during the
teaching process.
# 1.3 Project Process
During the development of our system, the entire project includes,
● Data collecting
● Data preprocessing
● Appropriate algorithm
● Coding
● Real-time detection
# 1.4 Key ideas
Applied PCA(Principal Components Analysis)

In face image data, each pixel can be regarded as a feature, and the
dimensionality of the image is usually very high, which leads to
problems of high computational complexity and poor model
generalization ability. PCA maps the original high-dimensional feature
space to a low-dimensional subspace through linear transformation,
thereby achieving dimensionality reduction of the data while retaining
the most important feature information.

# Important
Optimize the PCA process by using SVD to speed up the operational efficiency


![image](https://github.com/user-attachments/assets/83e64ed6-829e-46ad-9b81-6e6a731f695c)

![image](https://github.com/user-attachments/assets/f30de65d-0453-4f62-a4ef-24b4c2d3d7d9)
