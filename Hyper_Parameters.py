import tensorflow as tf

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

sound_Parameters_Dict = {
    "sample_Rate": 22050,    
    "frame_Shfit": 12.5,  #ms
    "frame_Length": 50,  #ms
    }

pattern_Prameters_Dict = {    
    "pattern_Files_Path": "D:/Tacotron2_Data/Talker1/Pattern",
    "mel_Spectrogram_Dimension": 80,
    "spectrogram_Dimension": 1024,
    "max_Spectrogram_Length": 1500,
    "batch_Size": 32,
    "max_Queue": 20,    
    }

encoder_Prameters_Dict = {
    "number_of_Token": len(token_Index_Dict),
    "token_Embedding_Size": 512,
    "conv_Filter_Count": 512,
    "conv_Kernal_Size": 5,
    "conv_Layer_Count": 3,
    "lstm_Cell_Size": 256, #Each direction
    "zoneout_Rate": 0.1,
    }

attention_Prameters_Dict = {
    "attention_Size": 128,
    }

decoder_Prameters_Dict = {
    "pre_Net_Layer_Size": 256,
    "pre_Net_Layer_Count": 2,
    "pre_Net_Dropout_Rate": 0.5,
    "lstm_Cell_Size": 1024,
    "zoneout_Rate": 0.1,    
    "output_Size_per_Step": 3,    #Tacotron1: 3~5 by the code, Tacotron2: 1
    "post_Net_Conv_Filter_Count": 512,
    "post_Net_Conv_Kernal_Size": 5,
    "post_Net_Conv_Layer_Count": 5,
    }

training_Loss_Parameters_Dict = {
    "priority_Frequencies": None,    #(200, 4000)
    "initial_Learning_Rate": 0.002,
    "decay_Type":"noam",  #"noam", "exponential", "static"
    }

sound_Parameters = tf.contrib.training.HParams(**sound_Parameters_Dict)
pattern_Prameters = tf.contrib.training.HParams(**pattern_Prameters_Dict)
encoder_Prameters = tf.contrib.training.HParams(**encoder_Prameters_Dict)
attention_Prameters = tf.contrib.training.HParams(**attention_Prameters_Dict)
decoder_Prameters = tf.contrib.training.HParams(**decoder_Prameters_Dict)
training_Loss_Parameters = tf.contrib.training.HParams(**training_Loss_Parameters_Dict)