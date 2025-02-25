# wolfcorp
NCSU E 101 FEDD Project (Spring 2025, section 019)

**Windows Tutorial**
Ensure Python 3.12.9 is installed and added to path during installation as well as Microsfoft C++ Redistributables
git clone https://github.com/paper5375/wolfcorp.git
If Scripts dont work in cmd install git + git bash

Enter tflite1 Folder 

Run: pip install virtualenv
Then Run: python -m venv tflite1-env
Then Run: source tflite-env/Scripts/activate
--------------------------------------------
Now Run: pip install tensorflow
Then Run: pip install opencv-python
--------------------------------------------
Now Run: curl -O https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
Then Run: unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model
--------------------------------------------
To run the object detection window: python TFLite_detection_webcam.py --modeldir=Sample_TFLite_model

**Raspberry Pi Tutorial**
sudo apt-get update
sudo apt-get dist-upgrade

git clone https://github.com/paper5375/wolfcorp.git

mv wolfcorp tflite1
cd tflite1

sudo pip3 install virtualenv
python3 -m venv tflite1-env

source tflite1-env/Scripts/activate

bash get_pi_requirements.sh

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model

python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model
