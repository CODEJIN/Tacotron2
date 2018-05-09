import numpy as np;
import json, re, os, librosa, argparse;
from scipy import signal;
from Audio import *;
from hanja import hangul;
import _pickle as pickle
from Kor_Str_Management import String_to_Token_List;
from Hyper_Parameters import sound_Parameters, pattern_Prameters;

token_Index_Dict = {
    0:"<SOS>",
    1:"<EOS>",
    2:"<Space>",
    3:"ㄱ(Onset)",
    4:"ㄲ(Onset)",
    5:"ㄴ(Onset)",
    6:"ㄷ(Onset)",
    7:"ㄸ(Onset)",
    8:"ㄹ(Onset)",
    9:"ㅁ(Onset)",
    10:"ㅂ(Onset)",
    11:"ㅃ(Onset)",
    12:"ㅅ(Onset)",
    13:"ㅆ(Onset)",
    14:"ㅇ(Onset)",
    15:"ㅈ(Onset)",
    16:"ㅉ(Onset)",
    17:"ㅊ(Onset)",
    18:"ㅋ(Onset)",
    19:"ㅌ(Onset)",
    20:"ㅍ(Onset)",
    21:"ㅎ(Onset)",
    22:"ㅏ(Nucleus)",
    23:"ㅐ(Nucleus)",
    24:"ㅑ(Nucleus)",
    25:"ㅒ(Nucleus)",
    26:"ㅓ(Nucleus)",
    27:"ㅔ(Nucleus)",
    28:"ㅕ(Nucleus)",
    29:"ㅖ(Nucleus)",
    30:"ㅗ(Nucleus)",
    31:"ㅘ(Nucleus)",
    32:"ㅙ(Nucleus)",
    33:"ㅚ(Nucleus)",
    34:"ㅛ(Nucleus)",
    35:"ㅜ(Nucleus)",
    36:"ㅝ(Nucleus)",
    37:"ㅞ(Nucleus)",
    38:"ㅟ(Nucleus)",
    39:"ㅠ(Nucleus)",
    40:"ㅡ(Nucleus)",
    41:"ㅢ(Nucleus)",
    42:"ㅣ(Nucleus)",
    43:" (Coda)",
    44:"ㄱ(Coda)",
    45:"ㄲ(Coda)",
    46:"ㄳ(Coda)",
    47:"ㄴ(Coda)",
    48:"ㄵ(Coda)",
    49:"ㄶ(Coda)",
    50:"ㄷ(Coda)",
    51:"ㄹ(Coda)",
    52:"ㄺ(Coda)",
    53:"ㄻ(Coda)",
    54:"ㄼ(Coda)",
    55:"ㄽ(Coda)",
    56:"ㄾ(Coda)",
    57:"ㄿ(Coda)",
    58:"ㅀ(Coda)",
    59:"ㅁ(Coda)",
    60:"ㅂ(Coda)",
    61:"ㅄ(Coda)",
    62:"ㅅ(Coda)",
    63:"ㅆ(Coda)",
    64:"ㅇ(Coda)",
    65:"ㅈ(Coda)",
    66:"ㅊ(Coda)",
    67:"ㅋ(Coda)",
    68:"ㅌ(Coda)",
    69:"ㅍ(Coda)",
    70:"ㅎ(Coda)",
    71:".(Symbol)",
    72:",(Symbol)",
    73:"?(Symbol)",
    74:"!(Symbol)",
}

def Pattern_Generate(recognition_File_Path, audio_Path, export_Pattern_Path = None):
    if export_Pattern_Path is None:
        export_Pattern_Path = pattern_Prameters.pattern_Files_Path;    
    if not os.path.exists(export_Pattern_Path):
        os.makedirs(export_Pattern_Path);

    load_Recognition_Dict = json.loads(open(recognition_File_Path, "r", encoding = "utf-8-sig").read());

    max_Token_Length = 0;
    max_Spectrogram_Length = 0;
    max_Mel_Spectrogram_Length = 0;
    pattern_File_Name_List = [];
    pattern_Token_Length_List = [];

    for index, (file_Name, recognition_String) in enumerate(load_Recognition_Dict.items()):
        token_Pattern = String_to_Token_List(recognition_String);
        if not token_Pattern:
            print("{}/{}\t{} is removed bacause of an unsupported character".format(index, len(load_Recognition_Dict), file_Name));
            continue;

        pattern_File_Name = file_Name.replace("wav", "pickle");
        pattern_File_Name_List.append(pattern_File_Name);
        pattern_Token_Length_List.append(len(token_Pattern));

        signal = librosa.core.load(os.path.join(audio_Path, file_Name).replace("\\", "/"), sr = sound_Parameters.sample_Rate)[0];
        spectrogram_Pattern = spectrogram(
            y= signal,
            num_freq = pattern_Prameters.spectrogram_Dimension,
            frame_shift_ms = sound_Parameters.frame_Shfit,
            frame_length_ms = sound_Parameters.frame_Length,
            sample_rate = sound_Parameters.sample_Rate,
            );
        mel_Spectrogram_Pattern = melspectrogram(
            y= signal,
            num_freq = pattern_Prameters.spectrogram_Dimension,
            num_mels = pattern_Prameters.mel_Spectrogram_Dimension,
            frame_shift_ms = sound_Parameters.frame_Shfit,
            frame_length_ms = sound_Parameters.frame_Length,
            sample_rate = sound_Parameters.sample_Rate,
            );

        if max_Token_Length < len(token_Pattern):
            max_Token_Length = len(token_Pattern);
        if max_Spectrogram_Length < spectrogram_Pattern.shape[1]:
            max_Spectrogram_Length = spectrogram_Pattern.shape[1];
        if max_Mel_Spectrogram_Length < mel_Spectrogram_Pattern.shape[1]:
            max_Mel_Spectrogram_Length = mel_Spectrogram_Pattern.shape[1];

        save_Dict = {};
        save_Dict["Text"] = recognition_String;
        save_Dict["Token_Pattern"] = token_Pattern;
        save_Dict["Spectrogram_Pattern"] = spectrogram_Pattern;
        save_Dict["Mel_Spectrogram_Pattern"] = mel_Spectrogram_Pattern;
        with open(os.path.join(export_Pattern_Path, pattern_File_Name).replace("\\", "/"), "wb") as f:
            pickle.dump(save_Dict, f, protocol=2);
                    
        print("{}/{}\t{}\t{}\t{}\t{}".format(index, len(load_Recognition_Dict), file_Name, len(token_Pattern), spectrogram_Pattern.shape, mel_Spectrogram_Pattern.shape));

    metadata_Dict = {};
    metadata_Dict["Token_Index_Dict"] = token_Index_Dict;
    metadata_Dict["File_Name_List"] = pattern_File_Name_List;
    metadata_Dict["Token_Length_List"] = pattern_Token_Length_List;
    
    metadata_Dict["Sample_Rate"] = sound_Parameters.sample_Rate;
    metadata_Dict["Frame_Shift"] = sound_Parameters.frame_Shfit;
    metadata_Dict["Frame_Length"] = sound_Parameters.frame_Length;

    metadata_Dict["Mel_Spectrogram_Dimension"] = pattern_Prameters.mel_Spectrogram_Dimension;
    metadata_Dict["Spectrogram_Dimension"] = pattern_Prameters.spectrogram_Dimension;    
    
    metadata_Dict["Max_Token_Length"] = max_Token_Length;
    metadata_Dict["Max_Spectrogram_Length"] = max_Spectrogram_Length;
    metadata_Dict["Max_Mel_Spectrogram_Length"] = max_Mel_Spectrogram_Length;

    with open(os.path.join(export_Pattern_Path, "Pattern_Metadata_Dict.pickle").replace("\\", "/"), "wb") as f:
        pickle.dump(metadata_Dict, f, protocol=2);

if __name__ == '__main__':
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-r", "--recognition", required=True);
    argParser.add_argument("-af", "--audio", required=True);
    argParser.add_argument("-pf", "--export", required=False);
    argument_Dict = vars(argParser.parse_args());
    
    
    Pattern_Generate(
        recognition_File_Path=argument_Dict["recognition"],
        audio_Path=argument_Dict["audio"]
    )
    
    
    #Pattern_Generate(
    #    recognition_File_Path="D:/Tacotron2_Data/Talker1/alignment.json",
    #    audio_Path="D:/Tacotron2_Data/Talker1/Audio"
    #)

