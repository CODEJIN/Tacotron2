Current Testing......

# ~~Tacotron2~~ Tacotron1.5 Kor ver.

## Materials
1. Audio files.
2. A JSON file containing the pronunciation for the audio file. The file name and text are the key and value, respectively.
3. Python 3.x and Tensorflow environment. This code is only tested in Python 3.6 and Tensorflow 1.7.

## How to use
### Generating pattern files.

    python Pattern_Generate.py -r 'recognition.json' -af 'audio_files_path' -pf 'export_pattern_files_path'

If you set 'pattern_Files_Path' of hyper parameter, PF arg is optional.

### Set the parameters in Hyper_Parameters.py

### Run

    python Tacotron1_5.py -e 'extract_folder' -m 'mode' -t 'time' -s 'string'
    
options:
-e: extract folder for the graph summary, checkpoint, and result. Required.

-m: mode. 'train' or 'test'. Required.

-t: When mode is train, how often to run tests and save the checkpoint. Default is '1000'

-s: In test, the sentence which will be tested. Default is '올해 겨울은 참 유난히도 길었습니다.'
    
## To do
1. Change Griffin-lim to Wavenet for Tacotron2.
