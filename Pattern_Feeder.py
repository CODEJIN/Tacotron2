import tensorflow as tf;
import numpy as np;
from threading import Thread;
from collections import deque;
import time, os;
import _pickle as pickle;
from random import shuffle;
from hanja import hangul;
from Kor_Str_Management import String_to_Token_List;
from Hyper_Parameters import sound_Parameters, pattern_Prameters, encoder_Prameters, attention_Prameters, decoder_Prameters, training_Loss_Parameters;

class Pattern_Feeder:
    def __init__(self, test_Only = True):        
        self.Placeholder_Generate();

        if not test_Only:
            with open(os.path.join(pattern_Prameters.pattern_Files_Path, "Pattern_Metadata_Dict.pickle").replace("\\", "/"), "rb") as f:
                load_Dict = pickle.load(f);        
            self.file_Name_List = load_Dict['File_Name_List'];
            self.token_Length_List = load_Dict['Token_Length_List'];
            
            if sound_Parameters.sample_Rate != load_Dict['Sample_Rate']:
                raise ValueError("The sample rate of assigned parameter({}) and data({}) are different.\nPlease check the inconsistency.".format(sound_Parameters.sample_Rate, load_Dict['Sample_Rate']));
            if sound_Parameters.frame_Shift != load_Dict['Frame_Shift']:
                raise ValueError("The frame shift of assigned parameter({}) and data({}) are different.\nPlease check the inconsistency.".format(sound_Parameters.frame_Shift, load_Dict['Frame_Shift']));
            if sound_Parameters.frame_Length != load_Dict['Frame_Length']:
                raise ValueError("The frame length of assigned parameter({}) and data({}) are different.\nPlease check the inconsistency.".format(sound_Parameters.frame_Length, load_Dict['Frame_Length']));

            if pattern_Prameters.spectrogram_Dimension != load_Dict['Spectrogram_Dimension']:
                raise ValueError("The spectram dimension of assigned parameter({}) and data({}) are different. Please check the inconsistency.".format(pattern_Prameters.spectrogram_Dimension, load_Dict['Spectrogram_Dimension']));            
            if pattern_Prameters.mel_Spectrogram_Dimension != load_Dict['Mel_Spectrogram_Dimension']:
                raise ValueError("The mel-spectram dimension of assigned parameter({}) and data({}) are different. Please check the inconsistency.".format(pattern_Prameters.mel_Spectrogram_Dimension, load_Dict['Mel_Spectrogram_Dimension']));
            if pattern_Prameters.max_Spectrogram_Length < load_Dict['Max_Spectrogram_Length']:
                raise ValueError("The max spectram length of assigned parameter({}) is small than the data({}). Please check the inconsistency.".format(pattern_Prameters.max_Spectrogram_Length, load_Dict['Max_Spectrogram_Length']));

            self.pattern_Queue = deque();

            pattern_Generate_Thread = Thread(target=self.Pattern_Generate);
            pattern_Generate_Thread.daemon = True;
            pattern_Generate_Thread.start();


    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeHolders') as scope:
            self.placeholder_Dict["Is_Training"] = tf.placeholder(tf.bool, name="is_Training_Placeholder");    #boolean
            self.placeholder_Dict["Signal"] = tf.placeholder(tf.float32, shape=(None, None), name="signal_Placeholder");    #[batch_Size, token_Length];
            self.placeholder_Dict["Token"] = tf.placeholder(tf.int32, shape=(None, None), name="token_Placeholder");    #[batch_Size, token_Length];
            self.placeholder_Dict["Token_Length"] = tf.placeholder(tf.int32, shape=(None,), name="token_Length_Placeholder");    #[batch_Size];
            self.placeholder_Dict["Spectrogram"] = tf.placeholder(tf.float32, shape=(None, None, pattern_Prameters.spectrogram_Dimension), name="spectrogram_Placeholder");    #[batch_Size, spectrogram_Length, spectogram_Dimension];
            self.placeholder_Dict["Mel_Spectrogram"] = tf.placeholder(tf.float32, shape=(None, None, pattern_Prameters.mel_Spectrogram_Dimension), name="mel_Spectrogram_Placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict["Mel_Spectrogram_Length"] = tf.placeholder(tf.int32, shape=(None,), name="mel_Spectrogram_Length_Placeholder");    #[batch_Size];

    def Pattern_Generate(self, is_Random = False):
        pattern_Index_List = list(range(len(self.file_Name_List)));
        if not is_Random:
            pattern_Index_List = [x for _, x in sorted(zip(self.token_Length_List, pattern_Index_List))]    #Sequence by length

        while True:
            if is_Random:
                shuffle(pattern_Index_List);    #Randomized order

            pattern_Index_Batch_List = [pattern_Index_List[x:x+pattern_Prameters.batch_Size] for x in range(0, len(pattern_Index_List), pattern_Prameters.batch_Size)]
            shuffle(pattern_Index_Batch_List);

            current_Index = 0;
            while current_Index < len(pattern_Index_Batch_List):
                if len(self.pattern_Queue) >= pattern_Prameters.max_Queue:
                    time.sleep(0.1);
                    continue;
                self.New_Pattern_Append(pattern_Index_Batch_List[current_Index]);
                current_Index += 1;

    def New_Pattern_Append(self, pattern_Index_List):
        pattern_Count = len(pattern_Index_List);

        signal_Pattern_List = [];
        token_Pattern_List = [];
        spectrogram_Pattern_List = [];
        mel_Spectrogram_Pattern_List = [];

        for index in pattern_Index_List:
            file_Name = self.file_Name_List[index];
            with open(os.path.join(pattern_Prameters.pattern_Files_Path, file_Name).replace("\\", "/"), "rb") as f:
                load_Dict = pickle.load(f);
            signal_Pattern_List.append(load_Dict["Signal"]);
            token_Pattern_List.append(load_Dict["Token_Pattern"]);
            spectrogram_Pattern_List.append(np.transpose(load_Dict["Spectrogram_Pattern"]));
            mel_Spectrogram_Pattern_List.append(np.transpose(load_Dict["Mel_Spectrogram_Pattern"]));

        signal_per_Frame = sound_Parameters.sample_Rate // int(1000 / sound_Parameters.frame_Shift);
        max_Signal_Pattern_Length = int(np.ceil(max([x.shape[0] for x in signal_Pattern_List]) / signal_per_Frame) * signal_per_Frame);   #To compare to the output
        max_Token_Pattern_Length = max([len(x) for x in token_Pattern_List]);
        max_Spectrogram_Pattern_Length = int(np.ceil(max([x.shape[0] for x in spectrogram_Pattern_List]) / decoder_Prameters.output_Size_per_Step) * decoder_Prameters.output_Size_per_Step);   #To compare to the output

        signal_Pattern_Array = np.zeros((pattern_Count, max_Signal_Pattern_Length)).astype("float32");
        token_Pattern_Array = np.zeros((pattern_Count, max_Token_Pattern_Length)).astype("int32");
        token_Length_Array = np.zeros((pattern_Count)).astype("int32");
        spectrogram_Pattern_Array = np.zeros((pattern_Count, max_Spectrogram_Pattern_Length, pattern_Prameters.spectrogram_Dimension)).astype("float32");
        mel_Spectrogram_Pattern_Array = np.zeros((pattern_Count, max_Spectrogram_Pattern_Length, pattern_Prameters.mel_Spectrogram_Dimension)).astype("float32");
        mel_Spectrogram_Length_Array = np.zeros((pattern_Count)).astype("int32");

        for index, (signal, token_Pattern, spectrogram_Pattern, mel_Spectrogram_Pattern) in enumerate(zip(signal_Pattern_List, token_Pattern_List, spectrogram_Pattern_List, mel_Spectrogram_Pattern_List)):
            signal_Pattern_Array[index, :signal.shape[0]] = signal;
            token_Pattern_Array[index, :len(token_Pattern)] = token_Pattern;
            token_Length_Array[index] = len(token_Pattern);
            spectrogram_Pattern_Array[index, :spectrogram_Pattern.shape[0], :] = spectrogram_Pattern;
            mel_Spectrogram_Pattern_Array[index, :mel_Spectrogram_Pattern.shape[0], :] = mel_Spectrogram_Pattern;
            mel_Spectrogram_Length_Array[index] = mel_Spectrogram_Pattern.shape[0];

        feed_Dict = {
            self.placeholder_Dict["Is_Training"]: True,
            self.placeholder_Dict["Signal"]: signal_Pattern_Array,
            self.placeholder_Dict["Token"]: token_Pattern_Array,
            self.placeholder_Dict["Token_Length"]: token_Length_Array,
            self.placeholder_Dict["Spectrogram"]: spectrogram_Pattern_Array,
            self.placeholder_Dict["Mel_Spectrogram"]: mel_Spectrogram_Pattern_Array,
            self.placeholder_Dict["Mel_Spectrogram_Length"]: mel_Spectrogram_Length_Array
            }
        self.pattern_Queue.append(feed_Dict);

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Queue.popleft();

    def Get_Test_Pattern(self, test_String):
        if type(test_String) == str:
            test_String_List = [test_String];
        else:
            test_String_List = test_String;

        batch_Size = len(test_String_List);
        token_Pattern_List = [];

        for index, test_String in enumerate(test_String_List):
            token_Pattern_List.append(String_to_Token_List(test_String));

        max_Token_Pattern_Length = max([len(x) for x in token_Pattern_List])

        token_Pattern_Array = np.zeros((batch_Size, max_Token_Pattern_Length));
        token_Length_Array = np.zeros((batch_Size));

        for index, token_Pattern in enumerate(token_Pattern_List):
            token_Pattern_Array[index, :len(token_Pattern)] = token_Pattern;
            token_Length_Array[index] = len(token_Pattern);

        max_Iterations = int(np.ceil(pattern_Prameters.max_Spectrogram_Length / decoder_Prameters.output_Size_per_Step));
        dummy_Spectrogram_Pattern_Array = np.zeros((batch_Size, max_Iterations, pattern_Prameters.spectrogram_Dimension)).astype("float32");
        dummy_Mel_Spectrogram_Pattern_Array = np.zeros((batch_Size, max_Iterations, pattern_Prameters.mel_Spectrogram_Dimension)).astype("float32");
        dummy_Mel_Spectrogram_Length_Array = np.zeros((batch_Size)).astype("int32");

        feed_Dict = {
            self.placeholder_Dict["Is_Training"]: False,
            self.placeholder_Dict["Token"]: token_Pattern_Array,
            self.placeholder_Dict["Token_Length"]: token_Length_Array,
            self.placeholder_Dict["Spectrogram"]: dummy_Spectrogram_Pattern_Array,
            self.placeholder_Dict["Mel_Spectrogram"]: dummy_Mel_Spectrogram_Pattern_Array,
            self.placeholder_Dict["Mel_Spectrogram_Length"]: dummy_Mel_Spectrogram_Length_Array
            }
        
        return feed_Dict;

if __name__ == "__main__":
    new_Pattern_Feeder = Pattern_Feeder(test_Only = False)
    x = new_Pattern_Feeder.Get_Test_Pattern("웃훙")
    print(x)
    while True:
        time.sleep(1);
        print(len(new_Pattern_Feeder.pattern_Queue));
        if len(new_Pattern_Feeder.pattern_Queue) > 0:
            print(new_Pattern_Feeder.Get_Pattern())
            break;