#referring to: https://github.com/Rayhane-mamah/Tacotron-2/tree/master/tacotron

import numpy as np;
import tensorflow as tf;
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauMonotonicAttention, BahdanauAttention, BasicDecoder, dynamic_decode;
from tensorflow.contrib.rnn import BasicRNNCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper;
from ZoneoutLSTMCell import ZoneoutLSTMCell;
from Customized_Modules import *;
from Audio import *;
from Pattern_Feeder import Pattern_Feeder;
from Hyper_Parameters import sound_Parameters, pattern_Parameters, encoder_Parameters, attention_Parameters, decoder_Parameters, training_Loss_Parameters;
import time, os, argparse;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;

class Tacotron_Model:
    def __init__(self, extract_Dir = "Result", test_Only = False):
        self.extract_Dir = extract_Dir;
        if not os.path.exists(self.extract_Dir):
            os.makedirs(self.extract_Dir);

        self.tf_Session = tf.Session();

        self.pattern_Feeder = Pattern_Feeder(test_Only = test_Only)

        self.Tensor_Generate();

        self.tf_Saver = tf.train.Saver(max_to_keep=5);

    def Tensor_Generate(self):
        placeholder_Dict = self.pattern_Feeder.placeholder_Dict;

        with tf.variable_scope('encoder') as scope:
            batch_Size = tf.shape(placeholder_Dict["Token"])[0];

            token_Embedding = tf.get_variable(
                name = "token_Embedding",
                shape = (encoder_Parameters.number_of_Token, encoder_Parameters.token_Embedding_Size),
                dtype = tf.float32,
                initializer = tf.truncated_normal_initializer(stddev=0.5)
            )

            embedded_Input_Pattern = tf.nn.embedding_lookup(token_Embedding, placeholder_Dict["Token"]); #Shape: [batch_Size, token_Length, embedded_Pattern_Size];

            encoder_Activation = Encoder(
                input_Pattern = embedded_Input_Pattern,
                input_Length = placeholder_Dict["Token_Length"],                
                is_Training = placeholder_Dict["Is_Training"],
                scope = "encoder_Module"
                )

        with tf.variable_scope('attention') as scope:
            attention_Mechanism = BahdanauMonotonicAttention(
                num_units = attention_Parameters.attention_Size,
                memory = encoder_Activation,
                normalize=True,
                name="bahdanau_Monotonic_Attention"
            )

        with tf.variable_scope('decoder') as scope:
            linear_Projection_Activation, stop_Token, alignment_Histroy = Decoder(
                batch_Size = batch_Size,
                attention_Mechanism= attention_Mechanism,
                is_Training = placeholder_Dict["Is_Training"],
                target_Pattern = placeholder_Dict["Mel_Spectrogram"],
                scope = "decoder_Module"
                )            
            post_Net_Activation = PostNet(
                input_Pattern = linear_Projection_Activation,
                conv_Filter_Count_and_Kernal_Size_List = [(decoder_Parameters.post_Net_Conv_Filter_Count, decoder_Parameters.post_Net_Conv_Kernal_Size)] * decoder_Parameters.post_Net_Conv_Layer_Count,
                is_Training = placeholder_Dict["Is_Training"],
                scope = "post_Net"
                )

            mel_Spectrogram_Activation = linear_Projection_Activation + post_Net_Activation;

            #Wavenet is here in Tacotron2, but now I use the Tacotron1's method(CBHG).
            post_CBHG_Activation = CBHG(
                input_Pattern = mel_Spectrogram_Activation,
                input_Length = None,
                scope = "post_CBHG",
                is_Training = placeholder_Dict["Is_Training"],
                conv_Bank_Filter_Count = 256,
                conv_Bank_Max_Kernal_Size = 8,
                max_Pooling_Size = 2,
                conv_Projection_Filter_Count_and_Kernal_Size_List = [(256, 3), (80, 3)],
                highway_Layer_Count = 4,
                gru_Cell_Size = 128
            )

            spectrogram_Activation = tf.layers.dense(
                post_CBHG_Activation, 
                pattern_Parameters.spectrogram_Dimension, 
                name = "spectrogram"
                )

        with tf.variable_scope('training_Loss') as scope:
            #Mel-spectrogram loss            
            mel_Loss1 = tf.reduce_mean(tf.pow(placeholder_Dict["Mel_Spectrogram"] - linear_Projection_Activation, 2));
            mel_Loss2 = tf.reduce_mean(tf.pow(placeholder_Dict["Mel_Spectrogram"] - mel_Spectrogram_Activation, 2));
            
            #Stop token loss
            tiled_Range = tf.cast(tf.tile(tf.expand_dims(tf.range(tf.shape(stop_Token)[1]), axis=0), multiples=[batch_Size, 1]), tf.float32);            
            tiled_Spectrogram_Length = tf.cast(tf.tile(tf.expand_dims(placeholder_Dict["Mel_Spectrogram_Length"] - 1, axis=1), multiples=[1, tf.shape(stop_Token)[1]]), tf.float32)
            stop_Target = tf.clip_by_value(tf.sign(tiled_Range - tiled_Spectrogram_Length), clip_value_min = 0, clip_value_max = 1)
            stop_Token_Loss = tf.reduce_mean(tf.pow(stop_Target - stop_Token, 2));
            
            #Spectrogram loss. It is only for Tacotron1 method.
            l1 = tf.abs(placeholder_Dict["Spectrogram"] - spectrogram_Activation);

            if training_Loss_Parameters.priority_Frequencies is None:
                linear_Loss = tf.reduce_mean(l1);
            else:
                lower_Priority_Frequency_Cut, upper_Priority_Frequency_Cut = training_Loss_Parameters.priority_Frequencies;
                lower_Priority_Frequency = int(lower_Priority_Frequency_Cut / (sound_Parameters.sample_Rate * 0.5) * sound_Parameters.spectrogram_Dimension);
                upper_Priority_Frequency = int(upper_Priority_Frequency_Cut / (sound_Parameters.sample_Rate * 0.5) * sound_Parameters.spectrogram_Dimension);
                l1_Priority= l1[:,:,lower_Priority_Frequency:upper_Priority_Frequency];
                linear_Loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1_Priority);

            loss = mel_Loss1 + mel_Loss2 + stop_Token_Loss + linear_Loss;

            #Optimize
            global_Step = tf.Variable(0, name='global_Step', trainable=False);

            if training_Loss_Parameters.decay_Type.lower() == "noam":
                step = tf.cast(global_Step + 1, dtype=tf.float32);
                warmup_Steps = 4000.0;
                learning_Rate = training_Loss_Parameters.initial_Learning_Rate * warmup_Steps ** 0.5 * tf.minimum(step * warmup_Steps**-1.5, step**-0.5)
            elif self.learning_Rate_Decay_Type.lower() == "exponential":
                learning_Rate = training_Loss_Parameters.initial_Learning_Rate * tf.train.exponential_decay(1., global_Step, 3000, 0.95);
            elif self.learning_Rate_Decay_Type.lower() == "static":
                learning_Rate = tf.convert_to_tensor(training_Loss_Parameters.initial_Learning_Rate, dtype=tf.float32);
            else:
                raise Exception("Unsupported learning rate decay type");

            optimizer = tf.train.AdamOptimizer(learning_Rate);
            #The return value of 'optimizer.compute_gradients' is a list of tuples which is (gradient, variable).
            #Using * is making two seprate lists: (gradient1, gradient2, ...), (variable1, variable2)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)

        with tf.variable_scope('test_Inference') as scope:
            inverted_Signal = inv_spectrogram_tensorflow(
                spectrogram=spectrogram_Activation, 
                num_freq=pattern_Parameters.spectrogram_Dimension, 
                frame_shift_ms=sound_Parameters.frame_Shift, 
                frame_length_ms=sound_Parameters.frame_Length, 
                sample_rate=sound_Parameters.sample_Rate
                )
            
            alignment = tf.transpose(alignment_Histroy, [1, 0, 2]);  #Shape: (batch_Size, max_Token, (max_Spectrogram / output_Size_per_Step))            
            transposed_Spectrogram = tf.transpose(spectrogram_Activation, [0, 2, 1]);
            transposed_Mel_Spectrogram = tf.transpose(mel_Spectrogram_Activation, [0, 2, 1]);
                    
        self.training_Tensor_List = [global_Step, learning_Rate, loss, optimize];
        self.test_Tensor_List = [global_Step, learning_Rate, inverted_Signal, alignment, transposed_Spectrogram, transposed_Mel_Spectrogram];
        
        if not os.path.exists(self.extract_Dir + "/Summary"):
            os.makedirs(self.extract_Dir + "/Summary");
        graph_Writer = tf.summary.FileWriter(self.extract_Dir + "/Summary", self.tf_Session.graph);
        graph_Writer.close();
        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):  
        if not os.path.exists(self.extract_Dir + "/Checkpoint"):
            os.makedirs(self.extract_Dir + "/Checkpoint");

        checkpoint_Path = tf.train.latest_checkpoint(self.extract_Dir + "/Checkpoint");
        print("Lastest checkpoint:", checkpoint_Path);

        if checkpoint_Path is None:
            print("There is no checkpoint");
        else:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path);
            print("Checkpoint '", checkpoint_Path, "' is loaded");

    def Train(
        self,
        test_Step = 1000,
        checkpoint_Step = 1000,
        test_String = None
    ):
        if not os.path.exists(self.extract_Dir + "/Checkpoint"):
            os.makedirs(self.extract_Dir + "/Checkpoint");

        #self.Test(test_String = test_String);
        try:
            while True:
                start_Time = time.time();
                global_Step, learning_Rate, training_Loss, _ = self.tf_Session.run(
                    fetches = self.training_Tensor_List,
                    feed_dict = self.pattern_Feeder.Get_Pattern()
                )
                print(
                    "Spent_Time:", np.round(time.time() - start_Time, 3), "\t",
                    "Global_Step:", global_Step, "\t",
                    "Learning_Rate:", learning_Rate, "\t",
                    "Training_Loss:", training_Loss
                )

                if (global_Step % checkpoint_Step) == 0:
                    self.tf_Saver.save(self.tf_Session, self.extract_Dir + '/Checkpoint/tacotron.ckpt', global_step=global_Step)
                    print("Checkpoint saved");

                if (global_Step % test_Step) == 0 and not test_String is None:
                    self.Test(test_String = test_String);
        except KeyboardInterrupt:
            self.tf_Saver.save(self.tf_Session, self.extract_Dir + '/Checkpoint/tacotron.ckpt', global_step=global_Step)
            print("Checkpoint saved");
            self.Test(test_String = test_String);

    def Test(self, test_String):
        if not os.path.exists(self.extract_Dir + "/Test"):
            os.makedirs(self.extract_Dir + "/Test");
        if not os.path.exists(self.extract_Dir + "/Test/WAV"):
            os.makedirs(self.extract_Dir + "/Test/WAV");
        if not os.path.exists(self.extract_Dir + "/Test/Alignment"):
            os.makedirs(self.extract_Dir + "/Test/Alignment");
        if not os.path.exists(self.extract_Dir + "/Test/Spectrogram"):
            os.makedirs(self.extract_Dir + "/Test/Spectrogram");
        if not os.path.exists(self.extract_Dir + "/Test/Mel_Spectrogram"):
            os.makedirs(self.extract_Dir + "/Test/Mel_Spectrogram");
            
        global_Step, learning_Rate, inverted_Signal_Matrix, alignment_Matrix, spectrogram_Matrix, mel_Spectrogram_Matrix = self.tf_Session.run(
            fetches = self.test_Tensor_List,
            feed_dict = self.pattern_Feeder.Get_Test_Pattern(test_String)
        )

        wav_Matrix = inv_preemphasis(inverted_Signal_Matrix);

        for index in range(wav_Matrix.shape[0]):
            save_File_Name = "E%d_T%d_Result" % (global_Step, index);
            wav = wav_Matrix[index];
            alignment = alignment_Matrix[index];
            spectrogram = spectrogram_Matrix[index];
            mel_Spectrogram = mel_Spectrogram_Matrix[index];

            self.Export_Wav(wav = wav, file_Name = self.extract_Dir + "/Test/WAV/" +  save_File_Name + ".wav");
            self.Plot_Alignment(alignment = alignment, file_Name = self.extract_Dir + "/Test/Alignment/" +  save_File_Name + "_Alignment.png");
            self.Plot_Spectrogram(spectrogram = spectrogram, file_Name = self.extract_Dir + "/Test/Spectrogram/" +  save_File_Name + "_Spectrogram.png")
            self.Plot_Spectrogram(spectrogram = mel_Spectrogram, file_Name = self.extract_Dir + "/Test/Mel_Spectrogram/" +  save_File_Name + "_Mel.png")

            print(save_File_Name + " Tested")

    def Export_Wav(self, wav, file_Name):
        if file_Name[-4:] != ".wav":
            file_Name += ".wav";
        librosa.output.write_wav(file_Name, wav, 20000);

    def Plot_Alignment(self, alignment, file_Name):
        fig = plt.figure(figsize=(16, 8));
        plt.imshow(alignment, cmap="viridis", vmin=0.0, vmax=0.3, aspect='auto', origin='lower', interpolation='none');
        plt.xlabel("Encoder attention focus");
        plt.ylabel("Voice Time step");
        plt.colorbar();
        # plt.gca().set_xticks(range(0, self.max_Cycle, int(np.ceil(self.max_Cycle / 5))))
        # plt.gca().set_xticklabels(range(0, int(self.max_Cycle * 50), int(np.ceil(self.max_Cycle / 5) * 50)));
        if file_Name[-4:] != ".png":
            file_Name += ".png";
        plt.savefig(file_Name, bbox_inches='tight');
        plt.close(fig);

    def Plot_Spectrogram(self, spectrogram, file_Name):
        fig = plt.figure(figsize=(16, 8));
        plt.imshow(spectrogram, cmap="viridis", vmin=0.0, vmax=1.0, aspect='auto', origin='lower', interpolation='none');

        if file_Name[-4:] != ".png":
            file_Name += ".png";
        plt.savefig(file_Name, bbox_inches='tight');
        plt.close(fig);


if __name__ == "__main__":
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-e", "--extract", required=True);
    argParser.add_argument("-m", "--mode", required=True);    
    argParser.add_argument("-t", "--time", required=False);
    argParser.set_defaults(time = "1000");
    argParser.add_argument("-s", "--string", required=False);
    argParser.set_defaults(string = "올해 겨울은 참 유난히도 길었습니다.");
    argument_Dict = vars(argParser.parse_args());

    if not argument_Dict["mode"].lower() in ["train", "test"]:
        raise ValueError("Mode should be one of 'train' or 'test'.")
    elif argument_Dict["mode"] == "test":
        test_Only = True;
    else:
        test_Only = False;
    argument_Dict["time"] = int(argument_Dict["time"]);
    
    new_Tacotron_Model = Tacotron_Model(
        extract_Dir = argument_Dict["extract"], 
        test_Only = test_Only
    );
    new_Tacotron_Model.Restore();

    if test_Only:
        new_Tacotron_Model.Test();
    else:
        new_Tacotron_Model.Train(
            test_Step = argument_Dict["time"],
            checkpoint_Step = argument_Dict["time"],
            test_String = argument_Dict["string"]
            )
    
    #new_Tacotron_Model = Tacotron_Model(
    #    extract_Dir = "D:/Tacotron2_Data/Result", 
    #    test_Only = True
    #);
    #new_Tacotron_Model.Test("당신의 생각과는 달리 이천십팔년의 첫 날에 저는 아직 혼자 미국에 있었습니다.")
