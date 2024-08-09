## Setting up the procotring

First create a new conda virtual environment - `conda create -n ai-proctor python==3.7.0 -y`

Activate the created virtual environment - `conda activate ai-proctor`

Install the dlib library from conda - `conda install -c conda-froge dlib -y`

Next pip install the requirements
```
pip install --no-cache-dir -r requirements/core.txt
pip install --no-cache-dir -r requirements/dev.txt
pip install --no-cache-dir -r requirements/api.txt
```
Run the following 
`python online_proctoring_system --video_path "video_path" --debug`


For more info refer to the following - https://github.com/AparGarg99/Intelligent-Online-Exam-Proctoring-System
