# wolfcorp
NCSU E 101 FEDD Project (Spring 2025, section 019)

## Windows Tutorial

Ensure Python 3.12.9 is installed and added to path during installation as well as Microsfoft C++ Redistributables. If scripts don't work in Window's command line, install Git and Git Bash.

1. Clone repository
    1. `git clone https://github.com/paper5375/wolfcorp.git`
2. Enter `tflite1` folder 
3. Install `virtualenv` and create virtual environment
    1. `pip install virtualenv`
    2. `python -m venv tflite1-env`
    3. `source tflite-env/Scripts/activate`
4. Install packages
    1. `pip install tensorflow`
    2. `pip install opencv-python`
5. Get TensorFLow model
    1. `curl -O https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`
    2. `unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model`
6. Start object detection window
    1. `python TFLite_detection_webcam.py --modeldir=Sample_TFLite_model`

## Raspberry Pi Tutorial

1. Update essential packages
    1. `sudo apt-get update`
    2. `sudo apt-get dist-upgrade`
2. Clone repository
    1. `git clone https://github.com/paper5375/wolfcorp.git`
3. Create folder
    1. `mv wolfcorp tflite1`
    2. `cd tflite1`
4. Create virtual environment
    1. `sudo pip3 install virtualenv`
    2. `python3 -m venv tflite1-env`
    3. `source tflite1-env/Scripts/activate`
5. Install required packages
    1. `bash get_pi_requirements.sh`
    2. `wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`
    3. `unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model`
6. Start the model
    1. `python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model`
