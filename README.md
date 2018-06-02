# defect_detection

This project contains code to detect a defefct in a mechanical part

Clone this project 
'''
git clone https://github.com/sahilbadyal/defect_detection.git
'''

Notations = 

OKB -> Ok Back
OKF -> Ok Front
SM  -> Scratch Mark
SD  -> Slot Defect
WR  -> Wrinkling
TH  -> Thinning

##How to run client ?

'''
cd client
'''
Now set up a test file in config class in defect_detection_client.py
png_folder='../All_61326/test_61326/' -> This is the folder where the png/jpegs will reside
test_csv='../test.csv' -> The test File (contains name of files)

Run client

'''
python defect_detection_client.py
'''
