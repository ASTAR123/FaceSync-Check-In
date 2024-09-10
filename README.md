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
