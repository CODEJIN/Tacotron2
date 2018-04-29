# Tacotron2
## Materials
1. Audio files.
2. A JSON file containing the pronunciation for the audio file. The file name and text are the key and value, respectively.
3. Python 3.x and Tensorflow environment. This code is only tested in Python 3.6 and Tensorflow 1.7.

## How to use
### Generating pattern files.

    python Pattern_Generate.py -r 'recognition.json' -af 'audio_files_path' -pf 'export_pattern_files_path'

If you set 'pattern_Files_Path' of hyper parameter, PF arg is optional.

### Run

    python Tacotron1_5.py
    
## To do
1. Change Griffin-lim to Wavenet.
