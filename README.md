# wolfcorp
NCSU E 101 FEDD Project (Spring 2025, section 019)

## Windows Tutorial

Ensure Python 3.12.9 is installed and added to path during installation as well as Microsfoft C++ Redistributables. If scripts don't work in Window's command line, install Git and Git Bash.

1. Clone repository
a. `git clone https://github.com/paper5375/wolfcorp.git`
2. Enter `tflite1` folder 
3. Install `virtualenv` and create virtual environment
a. `pip install virtualenv`
b. `python -m venv tflite1-env`
c. `source tflite-env/Scripts/activate`
4. Install packages
a. `pip install tensorflow`
b. `pip install opencv-python`
5. Get TensorFLow model
a. `curl -O https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`
b. `unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model`
6. Start object detection window
a. `python TFLite_detection_webcam.py --modeldir=Sample_TFLite_model`

## Raspberry Pi Tutorial

1. Update essential packages
a. `sudo apt-get update`
b. `sudo apt-get dist-upgrade`
2. Clone repository
a. `git clone https://github.com/paper5375/wolfcorp.git`
3. Create folder
a. `mv wolfcorp tflite1`
b. `cd tflite1`
4. Create virtual environment
a. `sudo pip3 install virtualenv`
b. `python3 -m venv tflite1-env`
c. `source tflite1-env/Scripts/activate`
5. Install required packages
a. `bash get_pi_requirements.sh`
b. `wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`
c. `unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model`
6. Start the model
a. `python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model`
