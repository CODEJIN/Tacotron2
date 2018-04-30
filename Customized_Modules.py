#Basic input pattern shape: (Batch, Letter_Index);
#Embedded input pattern shape: (Batch, Letter_Index, distributed_Pattern)

import tensorflow as tf;
import numpy as np;
from tensorflow.contrib.rnn import RNNCell, BasicRNNCell, LSTMCell, GRUCell, MultiRNNCell, LSTMStateTuple, OutputProjectionWrapper;
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauMonotonicAttention, BasicDecoder, dynamic_decode, Helper, TrainingHelper;
from ZoneoutLSTMCell import ZoneoutLSTMCell;      
from Hyper_Parameters import pattern_Prameters, encoder_Prameters, decoder_Prameters

def Conv1D(input_Pattern, scope, is_Training, kernel_Size, filter_Count, activation):
    with tf.variable_scope(scope):
        conv1D_Output = tf.layers.conv1d(
            input_Pattern,
            filters=filter_Count,
            kernel_size=kernel_Size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1D_Output, training = is_Training);

def Pre_Net(input_Pattern, layer_Size_List=[256] * 2, dropout_Rate=0.5, is_Training=True, scope="pre_Net"):
    pre_Net_Activation = input_Pattern;
    with tf.variable_scope(scope):
        for index, layer_Size in enumerate(layer_Size_List):
            pre_Net_Activation = tf.layers.dropout(
                tf.layers.dense(
                    pre_Net_Activation,
                    layer_Size,
                    activation = tf.nn.relu,
                    use_bias=True,
                    name = "activation_{}".format(index)
                ),
                rate = dropout_Rate,
                training = is_Training,
                name = "dropout_{}".format(index)
            )

    return pre_Net_Activation;


def Encoder(
    input_Pattern,  #[Batch, Embedding_Dimension]
    input_Length,   #[Batch]
    is_Training = True,
    scope = "encoder_Module"
    ):
    with tf.variable_scope(scope):
        conv_Activation = input_Pattern;
        for index, (filter_Count, kernel_Size) in enumerate([(encoder_Prameters.conv_Filter_Count, encoder_Prameters.conv_Kernal_Size)] * encoder_Prameters.conv_Layer_Count):
            conv_Activation = Conv1D(
                input_Pattern = conv_Activation,
                filter_Count = filter_Count,
                kernel_Size = kernel_Size,
                activation = tf.nn.relu,
                scope = "conv_%d" % index,
                is_Training = is_Training
                )        

        #Bidirectional LSTM
        output_Pattern_List, rnn_State_List = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = ZoneoutLSTMCell(
                encoder_Prameters.lstm_Cell_Size, 
                is_training=is_Training, 
                cell_zoneout_rate=encoder_Prameters.zoneout_Rate, 
                output_zoneout_rate=encoder_Prameters.zoneout_Rate
                ),
            cell_bw = ZoneoutLSTMCell(
                encoder_Prameters.lstm_Cell_Size, 
                is_training=is_Training, 
                cell_zoneout_rate=encoder_Prameters.zoneout_Rate, 
                output_zoneout_rate=encoder_Prameters.zoneout_Rate
                ),
            inputs = conv_Activation,
            sequence_length = input_Length,
            dtype = tf.float32,
            scope = "bi_LSTM"
        )
        biLSTM_Activation = tf.concat(output_Pattern_List, axis = 2)

        return biLSTM_Activation;
    

def Decoder(
    batch_Size,
    attention_Mechanism,    
    is_Training = True,
    target_Pattern = None,
    scope = "decoder_Module"
    ):
    decoder_Pre_Net_Cell = DecoderPrenetWrapper(
        layer_Size_List= [decoder_Prameters.pre_Net_Layer_Size] * decoder_Prameters.pre_Net_Layer_Count,
        dropout_Rate= decoder_Prameters.pre_Net_Dropout_Rate,
        is_Training = is_Training
        )

    attention_Cell = AttentionWrapper(
        cell = decoder_Pre_Net_Cell,
        attention_mechanism = attention_Mechanism,
        alignment_history = True,
        output_attention = False,
        name = "attention"
        )
    concat_Cell = ConcatOutputAndAttentionWrapper(cell = attention_Cell);   #256 + 128 = 384
    decoder_Cell = MultiRNNCell(
        cells = [
            concat_Cell,
            ZoneoutLSTMCell(
                decoder_Prameters.lstm_Cell_Size, 
                is_training=is_Training, 
                cell_zoneout_rate= decoder_Prameters.zoneout_Rate, 
                output_zoneout_rate= decoder_Prameters.zoneout_Rate
                ),
            ZoneoutLSTMCell(
                decoder_Prameters.lstm_Cell_Size, 
                is_training= is_Training, 
                cell_zoneout_rate= decoder_Prameters.zoneout_Rate, 
                output_zoneout_rate= decoder_Prameters.zoneout_Rate
                )
        ])
    projection_Cell = LinearProjectionWrapper(
        cell = decoder_Cell,
        linear_Projection_Size = pattern_Prameters.mel_Spectrogram_Dimension * decoder_Prameters.output_Size_per_Step, 
        stop_Token_Size = 1
        )

    decoder_Initial_State = projection_Cell.zero_state(batch_size=batch_Size, dtype=tf.float32);    
    
    helper = Tacotron2_Helper(
        is_Training = is_Training,
        batch_Size = batch_Size,
        target_Pattern = target_Pattern,
        output_Dimension = pattern_Prameters.mel_Spectrogram_Dimension,
        output_Size_per_Step = decoder_Prameters.output_Size_per_Step,
        linear_Projection_Size = pattern_Prameters.mel_Spectrogram_Dimension * decoder_Prameters.output_Size_per_Step,
        stop_Token_Size = 1
    )

    final_Outputs, final_States, final_Sequence_Lengths = dynamic_decode(
        decoder = BasicDecoder(projection_Cell, helper, decoder_Initial_State),
        maximum_iterations = int(np.ceil(pattern_Prameters.max_Spectrogram_Length / decoder_Prameters.output_Size_per_Step)),
    )
    linear_Projection_Activation, stop_Token = tf.split(
        final_Outputs.rnn_output, 
        num_or_size_splits=[pattern_Prameters.mel_Spectrogram_Dimension * decoder_Prameters.output_Size_per_Step, 1], 
        axis=2
        )
    
    linear_Projection_Activation = tf.reshape(linear_Projection_Activation, [batch_Size, -1, pattern_Prameters.mel_Spectrogram_Dimension]);    
    alignment_Histroy = final_States[0].alignment_history;

    return linear_Projection_Activation, tf.squeeze(stop_Token), alignment_Histroy.stack();


def PostNet(
    input_Pattern,
    conv_Filter_Count_and_Kernal_Size_List = [(512, 5)] * 5,
    is_Training = True,
    scope = "post_Net"
    ):
    with tf.name_scope(name=scope):
        conv_Activation = input_Pattern;
        for index, (filter_Count, kernel_Size) in enumerate(conv_Filter_Count_and_Kernal_Size_List):
            conv_Activation = Conv1D(
                input_Pattern = conv_Activation,
                filter_Count = filter_Count,
                kernel_Size = kernel_Size,
                activation = tf.nn.tanh if index < len(conv_Filter_Count_and_Kernal_Size_List) - 1 else None,
                scope = "conv_%d" % index,
                is_Training = is_Training
                )
        correction_Activation = tf.layers.dense(
            conv_Activation,
            units = input_Pattern.get_shape()[2],
            activation = None,
            use_bias = True,
            name = "correction"
            )
    return correction_Activation;


class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    def __init__(self, layer_Size_List=[256]*2, dropout_Rate=0.5, is_Training=True):
        super(DecoderPrenetWrapper, self).__init__();
        self._layer_Size_List=layer_Size_List;
        self._dropout_Rate=dropout_Rate;
        self._is_Training = is_Training;        

    @property
    def state_size(self):
        return self._layer_Size_List[-1]

    @property
    def output_size(self):
        return self._layer_Size_List[-1]

    def call(self, inputs, state):
        prenet_Out = Pre_Net(
            input_Pattern = inputs,
            layer_Size_List = self._layer_Size_List,
            dropout_Rate = self._dropout_Rate,
            scope = "pre_Net",
            is_Training = self._is_Training
        )

        return prenet_Out, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros(shape=(batch_size, self._layer_Size_List[-1]))


class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.
    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class LinearProjectionWrapper(RNNCell):
    '''Projecting the mel-spectrogram and stop token.'''
    def __init__(self, cell, linear_Projection_Size, stop_Token_Size):
        super(LinearProjectionWrapper, self).__init__()
        self._cell = cell
        self._linear_Projection_Size = linear_Projection_Size;
        self._stop_Token_Size = stop_Token_Size;

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._linear_Projection_Size + self._stop_Token_Size;

    def call(self, inputs, state):
        outputs, res_state = self._cell(inputs, state);
        projection = tf.layers.dense(
            outputs,
            units = self._linear_Projection_Size + self._stop_Token_Size,
            activation = None,
            use_bias = True,
            name = "mel_Projection"
        )
        mel_Projection, stop_Token = tf.split(projection, num_or_size_splits=[self._linear_Projection_Size, self._stop_Token_Size], axis=1);
        stop_Token = tf.nn.sigmoid(stop_Token);

        return tf.concat([mel_Projection, stop_Token], axis=1), res_state;

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


#See L87 of https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/tacotron.py. I need make the Tacotron decoder cell.
#See L72 of https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/Architecture_wrappers.py. I need make the Tacotron decoder cell.
class Tacotron2_Helper(Helper):
    def __init__(self, is_Training, batch_Size, target_Pattern, output_Dimension, output_Size_per_Step, linear_Projection_Size, stop_Token_Size):
        '''
            batch_Size: batch size
            target_Pattern: decoder Mel-Spectrogram Target
            output_Dimension: Mel-Spectrogram dimension (ex. 80)
            output_Size_Per_Step: For last index
        '''
        with tf.name_scope('Tacotron2_Helper'):
            # inputs is [N, T_in], targets is [N, T_out, D]
            self.is_Training = is_Training;
            self.batch_Size = batch_Size;
            self.output_Dimension = output_Dimension;
            self.linear_Projection_Size = linear_Projection_Size;
            self.stop_Token_Size = stop_Token_Size;

            #Only Training use.
            # Feed every r-th target frame as input
            self.target_Pattern_as_Input = target_Pattern[:, output_Size_per_Step-1::output_Size_per_Step, :];
            # Use full length for every target because we don't want to mask the padding frames
            num_Steps = tf.shape(self.target_Pattern_as_Input)[1]
            self.length = tf.tile([num_Steps], [self.batch_Size])    # All batch have same length.

    @property
    def batch_size(self):
        return self.batch_Size;

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([]);  #Ignored

    @property
    def sample_ids_dtype(self):
        return np.int32;

    def initialize(self, name=None):
        return (tf.tile([False], [self.batch_Size]), _go_frames(self.batch_Size, self.output_Dimension));

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self.batch_Size]);  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope('Tacotron2_Helper'):
            def is_Training_True():
                finished = (time + 1 >= self.length)
                
                #Rayhane's code assign the predicted pattern with probability. See the commented code. I ignore that.
                next_Input = self.target_Pattern_as_Input[:, time, :]   #Teacher_forcing                
                #next_inputs = tf.cond(
				#tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
				#lambda: self._targets[:, time, :], #Teacher-forcing: return true frame
				#lambda: outputs[:,-self._output_dim:])

                return (finished, next_Input, state)

            def is_Training_False():
                linear_Projection, stop_Token = tf.split(outputs, num_or_size_splits=[self.linear_Projection_Size, self.stop_Token_Size], axis=1);
                
                # Feed last output frame as next input. outputs is [N, output_dim * r]
                next_Input = linear_Projection[:, -self.output_Dimension:]                
                # When stop_Token is over 0.5, model stop.
                finished = tf.cast(tf.round(tf.squeeze(stop_Token, axis=1)), dtype=tf.bool);

                return (finished, next_Input, state)

            return tf.cond(pred = self.is_Training, true_fn = is_Training_True, false_fn = is_Training_False);

def _go_frames(batch_Size, output_Dimension):
  '''Returns all-zero <GO> frames for a given batch size and output dimension'''
  return tf.tile([[0.0]], [batch_Size, output_Dimension])




#CBHG and Highway-net is for 1.5.
#Wavenet will replace this modules.
def CBHG(
    input_Pattern,
    input_Length,
    scope,
    is_Training,
    conv_Bank_Filter_Count = 128,
    conv_Bank_Max_Kernal_Size = 16,
    max_Pooling_Size = 2,
    conv_Projection_Filter_Count_and_Kernal_Size_List = [(128, 3), (128, 3)],
    highway_Layer_Count = 4,
    gru_Cell_Size = 128,
    ):

    with tf.variable_scope(scope):
        with tf.variable_scope('conv_Bank'):
            #Convolution Bank
            bank_Layer_List = [];
            for kernel_Size in range(1, conv_Bank_Max_Kernal_Size + 1):
                bank_Layer = Conv1D(
                    input_Pattern,
                    filter_Count = conv_Bank_Filter_Count,
                    kernel_Size = kernel_Size,
                    activation = tf.nn.relu,
                    scope = "conv1D_%d" % kernel_Size,
                    is_Training = is_Training
                )
                bank_Layer_List.append(bank_Layer);
            conv_Bank_Activation = tf.concat(bank_Layer_List, axis = -1);

        #Max pooling
        max_Pooling_Activation = tf.layers.max_pooling1d(
            conv_Bank_Activation,
            pool_size = max_Pooling_Size,
            strides = 1,
            padding = 'same'
        )

        #Convolution Projections
        conv_Projection_Activation = max_Pooling_Activation;
        for index, (filter_Count, kernel_Size) in enumerate(conv_Projection_Filter_Count_and_Kernal_Size_List):
            conv_Projection_Activation = Conv1D(
                conv_Projection_Activation,
                filter_Count = filter_Count,
                kernel_Size = kernel_Size,
                activation = tf.nn.relu,
                scope = "projection_%d" % index,
                is_Training = is_Training
            );

        #Residual
        residual_Activation = conv_Projection_Activation + input_Pattern;

        #Cell size correction -> But I am not sure why this code is located before the highway.
        correlected_Residual_Activation = tf.layers.dense(
            residual_Activation,
            units = gru_Cell_Size,
            activation = None,
            use_bias = True,
            name = "size_Correction"
        )

        #Highways
        highway_Activation = correlected_Residual_Activation;
        for index in range(highway_Layer_Count):
            highway_Activation = Highway_Net(input_Pattern = highway_Activation, scope = "highway_%d" % index);

        #Bidirectional GRU
        output_Pattern_List, rnn_State_List = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = GRUCell(gru_Cell_Size),
            cell_bw = GRUCell(gru_Cell_Size),
            inputs = highway_Activation,
            sequence_length = input_Length,
            dtype = tf.float32
        )

        return tf.concat(output_Pattern_List, axis = 2)


def Highway_Net(input_Pattern, scope):
    unit_Size = int(input_Pattern.get_shape()[-1]); #tf.shape cannot be used.

    with tf.variable_scope(scope):
        relu_Activation = tf.layers.dense(
            input_Pattern,
            units = unit_Size,
            activation = tf.nn.relu,
            use_bias=True,
            name = 'highway_Relu'
        )

        sigmoid_Activation = tf.layers.dense(
            input_Pattern,
            units = unit_Size,
            activation = tf.nn.sigmoid,
            use_bias=True,
            bias_initializer=tf.constant_initializer(-1.0),
            name = 'highway_Sigmoid'
        )

        return sigmoid_Activation * relu_Activation + (1.0 - sigmoid_Activation) * input_Pattern;