import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, GRU, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import os
import pickle
import numpy as np
import keras_nlp
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, GRU, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from transformers import PreTrainedTokenizerFast
import time

#mediapi_directory = "/silenus/PROJECTS/pr-serveurgestuel/ouakriya/mediapi-rgb/"
#mediapi_directory = "/bettik/PROJECTS/pr-serveurgestuel/ouakriya/mediapi-rgb/"
mediapi_directory = "/research/crissp/ouakrim/data/mediapi-rgb/"
#mediapi_directory = ""
with open(mediapi_directory+"features/swin_features_stride_1.pkl", "rb") as input_file:
    swin_features = pickle.load(input_file)

with open(mediapi_directory+"subtitles.pkl", "rb") as input_file:
    subtitles = pickle.load(input_file)

def get_swin_features(video_name):
    if video_name not in list(swin_features['video_names']):
        return None
    # get video position in list
    pos = list(swin_features['video_names']).index(video_name)
    features_pos = np.argwhere(swin_features['clip_ix'][0] == pos)
    vid_features = swin_features['preds'][features_pos]
    vid_features = np.reshape(vid_features, (vid_features.shape[0], vid_features.shape[2]))
    return vid_features

def get_file_names(directory_path):
    file_names = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            file_names.append(file)
    return file_names

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
#import evaluate

from transformers import CamembertTokenizer
import numpy as np

#bleu_metric = evaluate.load("bleu")
#tokenizer_name = '/silenus/PROJECTS/pr-serveurgestuel/ouakriya/tokenizers/mediapi-rgb-tokenizer-4k'
#tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base", bos_token="<s>", eos_token="</s>", pad_token="<pad>")

def keras_tokenizer(tensor):
    if tensor.shape.rank > 1:
        my_str = tensor.numpy().tolist()[0][0].decode('UTF-8')
    else:
        my_str = tensor.numpy().tolist()[0].decode('UTF-8')

    new_tensor = tf.convert_to_tensor([tokenizer.tokenize(my_str)])
    #print(new_tensor)
    if tensor.shape.rank > 1:
        new_tensor = tf.convert_to_tensor([[tokenizer.tokenize(my_str)]])
    return new_tensor

def decode_sequence(tokenizer, model, sign_embedding):
    generated_text = [[tokenizer.convert_tokens_to_ids("<s>")]]
    generated_text = tf.keras.utils.pad_sequences(generated_text, padding='post', value=0, maxlen=50)[0]
    last_predicted_token = ""
    j = 1

    while j < 50 and generated_text[j-1] != 6:
        decoding_step(model, sign_embedding, generated_text, j)
        j+=1
    return generated_text

def decoding_step(model, sign_embedding, generated_text, j):
    pred = model([sign_embedding, np.array([generated_text])])[0]
    pred = np.array(pred)
    last_predicted_token = np.argmax(pred[j-1], axis=0)
    generated_text[j] = last_predicted_token

def decode_clean_text(text):
    text_with_pad = tf.where(tf.equal(text, 0), tf.ones_like(text), text)
    decoded_text = tokenizer.decode(text_with_pad)
    decoded_text = decoded_text.replace('<pad>', '')
    decoded_text = decoded_text.replace('<s>', '')
    decoded_text = decoded_text.replace('</s>', '')
    return decoded_text

class BLEU_score(keras.callbacks.Callback):
    def __init__(self, data_loader, tokenizer):
        super().__init__()
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    def on_epoch_end(self, epoch, logs=None):
        bleu = keras_nlp.metrics.Bleu(max_order=4, tokenizer=keras_tokenizer)
        bleu_2 = keras_nlp.metrics.Bleu(max_order=2, tokenizer=keras_tokenizer)
        bleu_1 = keras_nlp.metrics.Bleu(max_order=1, tokenizer=keras_tokenizer)
        t1 = time.time()

        j = 0
        for first_batch in self.data_loader:
            for i in range(0, batch_size):
                gt = first_batch[0][1][i]
                sign_embedding = np.array([first_batch[0][0][i]])
                generated_text = decode_sequence(self.tokenizer, self.model, sign_embedding)

                reference_sentence = decode_clean_text(gt)
                translated_sentence = decode_clean_text(generated_text)

                bleu([reference_sentence], translated_sentence)
                bleu_1([reference_sentence], translated_sentence)
                bleu_2([reference_sentence], translated_sentence)
            j = j+1
            if j == 15:
                break
        logs['bleu_4'] = bleu.result()
        logs['bleu_2'] = bleu_2.result()
        logs['bleu_1'] = bleu_1.result()
        logs['time'] = time.time()-t1




VOCAB_SIZE = 35000
LR = 1e-3
MAX_SEQUENCE_LENGTH = 50
MAX_SWIN_LENGTH = 700
EMBED_DIM = 768
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 1
FFN_DIM = 64
OPTIMIZER_NAME = 'RMSProp'
DROPOUT = 0.5
LR_REDUCE_PATIENCE = 7
EARLY_STOPPING_PATIENCE = 15
checkpoint_path = "checkpoints/transformer_model_exp_33_pre_pad.tf"

print("LR =", LR)
print("VOCAB_SIZE =", VOCAB_SIZE)
print("MAX_SEQUENCE_LENGTH =", MAX_SEQUENCE_LENGTH)
print("MAX_SWIN_LENGTH =", MAX_SWIN_LENGTH)
print("EMBED_DIM =", EMBED_DIM)
print("NUM_HEADS =", NUM_HEADS)
print("NUM_ENCODER_LAYERS =", NUM_ENCODER_LAYERS)
print("NUM_DECODER_LAYERS =", NUM_DECODER_LAYERS)
print("LR_REDUCE_PATIENCE =", LR_REDUCE_PATIENCE)
print("EARLY_STOPPING_PATIENCE =", EARLY_STOPPING_PATIENCE)
print("OPTIMIZER_NAME =", OPTIMIZER_NAME)
print("FFN_DIM =", FFN_DIM)
print("DROPOUT =", DROPOUT)
print("checkpoint_path =", checkpoint_path)


# Custom data loader using TensorFlow Sequence API
class VideoTextDataLoader(tf.keras.utils.Sequence):
    def __init__(self, video_names, tokenizer, batch_size):
        self.video_names = video_names
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.video_names))

    def __len__(self):
        return len(self.video_names) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_swin_features = [get_swin_features(self.video_names[i].replace('.mp4', '')) for i in batch_indexes]
        batch_swin_features = pad_sequences(batch_swin_features, maxlen=MAX_SWIN_LENGTH, dtype='float32', padding='pre', truncating='post')

        batch_text_captions = [subtitles[self.video_names[i].replace('.mp4', '')]['text'] for i in batch_indexes]

        # Tokenize text captions
        #sequences = self.tokenizer(batch_text_captions, return_tensors='tf', padding=True, truncation=True)
        #sequences = pad_sequences(sequences, maxlen=50, dtype='float32', padding='post', truncating='post')
        #batch_sequences = sequences['input_ids'].numpy()
        sequences = self.tokenizer(batch_text_captions, return_tensors='tf', padding=True, truncation=True)
        batch_sequences = sequences['input_ids'].numpy()

        sequences_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=MAX_SEQUENCE_LENGTH + 1,
            pad_value=1, # id of <pad> for CamemBERT
        )
        batch_sequences = sequences_packer(batch_sequences)
        batch_sequences =  tf.where(tf.equal(batch_sequences, 1), tf.zeros_like(batch_sequences), batch_sequences)  # Replace padding with 0 to be able to mask during training



        return [batch_swin_features, batch_sequences[:, :-1]], batch_sequences[:, 1:]

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)





# Fake data
x = np.random.rand(32, 250, 768)
y = np.random.randint(0, 35001, size=(32, 50, 1))

batch_size = 32
train_video_names = get_file_names(mediapi_directory+'video_crops_train/')
train_data_loader = VideoTextDataLoader(train_video_names, tokenizer, batch_size)
val_video_names = get_file_names(mediapi_directory+'video_crops_val/')
val_data_loader = VideoTextDataLoader(val_video_names, tokenizer, batch_size)

test_video_names = get_file_names(mediapi_directory+'video_crops_test/')
test_data_loader = VideoTextDataLoader(test_video_names, tokenizer, batch_size)




# Define the model architecture
# Encoder
swin_input_shape = (MAX_SWIN_LENGTH, 768)
swin_input_layer = Input(shape=swin_input_shape)
positional_encoding = keras_nlp.layers.SinePositionEncoding()(swin_input_layer)
sign_stream = swin_input_layer + positional_encoding
for i in range(0, NUM_ENCODER_LAYERS):
    sign_stream = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=EMBED_DIM, num_heads=NUM_HEADS)(sign_stream)

# Decoder
text_input_shape = (50)
text_input_layer = Input(shape=text_input_shape)
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=50,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(text_input_layer)
for i in range(0, NUM_DECODER_LAYERS):
    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=FFN_DIM, num_heads=NUM_HEADS)(x, sign_stream)
x = Dropout(DROPOUT)(x)
decoder_outputs = Dense(35000, activation="softmax")(x)
model = Model(inputs=[swin_input_layer,text_input_layer], outputs=decoder_outputs)



# Compile the model
#optimizer = Adam(learning_rate=1e-3)
#model.compile(optimizer=optimizer, loss=CTCLoss)
if OPTIMIZER_NAME == 'RMSProp':
    optimizer = RMSprop(learning_rate=LR)
elif OPTIMIZER_NAME == 'Adam':
    optimizer = Adam(learning_rate=LR)

model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Define callbacks for checkpoints, early stopping, and learning rate reduction

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)
BLEU_callback = BLEU_score(val_data_loader, tokenizer)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    mode='min'
)
# Previously petience 10

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=LR_REDUCE_PATIENCE,
    verbose=1,
    mode='min',
    min_lr=1e-6
)
# Previously petience 5

# Train the model using data loader
history = model.fit(train_data_loader, validation_data=val_data_loader,
                    batch_size=batch_size,
                    epochs=100,
                    callbacks=[early_stop_callback, reduce_lr_callback, checkpoint_callback])




# Load the weights
model.load_weights(checkpoint_path)


# Prediction



rouge_1 = keras_nlp.metrics.RougeN(order=1)
rouge_2 = keras_nlp.metrics.RougeN(order=2)



bleu = keras_nlp.metrics.Bleu(max_order=4, tokenizer=keras_tokenizer)
bleu_1 = keras_nlp.metrics.Bleu(max_order=1, tokenizer=keras_tokenizer)
bleu_2 = keras_nlp.metrics.Bleu(max_order=2, tokenizer=keras_tokenizer)
bleu_3 = keras_nlp.metrics.Bleu(max_order=3, tokenizer=keras_tokenizer)
sacrebleu_bleu = keras_nlp.metrics.Bleu(max_order=4)
sacrebleu_bleu_1= keras_nlp.metrics.Bleu(max_order=1)
sacrebleu_bleu_2= keras_nlp.metrics.Bleu(max_order=2)
sacrebleu_bleu_3= keras_nlp.metrics.Bleu(max_order=3)

for first_batch in test_data_loader:
    for i in range(0, batch_size):
        gt = first_batch[0][1][i]
        sign_embedding = np.array([first_batch[0][0][i]])
        generated_text = decode_sequence(tokenizer, model, sign_embedding)

        reference_sentence = decode_clean_text(gt)
        translated_sentence = decode_clean_text(generated_text)
        print("-----")
        print("Refe", reference_sentence)
        print("Pred", translated_sentence)

        rouge_1(reference_sentence, translated_sentence)
        rouge_2(reference_sentence, translated_sentence)

        #print(tokenizer.tokenize(reference_sentence))
        #print(tokenizer.tokenize(translated_sentence))
        local_bleu = keras_nlp.metrics.Bleu(max_order=4, tokenizer=keras_tokenizer)

        #print(sentence_bleu([tokenizer.tokenize(reference_sentence)], tokenizer.tokenize(translated_sentence)))
        #print(bleu([reference_sentence], translated_sentence))
        print("CamemBERT BLEU-4", float(local_bleu([reference_sentence], translated_sentence)))
        bleu([reference_sentence], translated_sentence)
        bleu_1([reference_sentence], translated_sentence)
        bleu_2([reference_sentence], translated_sentence)
        bleu_3([reference_sentence], translated_sentence)
        sacrebleu_bleu([reference_sentence], translated_sentence)
        sacrebleu_bleu_1([reference_sentence], translated_sentence)
        sacrebleu_bleu_2([reference_sentence], translated_sentence)
        sacrebleu_bleu_3([reference_sentence], translated_sentence)

    print("ROUGE-1 Score: ", rouge_1.result())
    print("ROUGE-2 Score: ", rouge_2.result())
    print("CamemBERT BLEU Score: ", bleu.result())
    print("CamemBERT BLEU-1 Score: ", bleu_1.result())
    print("CamemBERT BLEU-2 Score: ", bleu_2.result())
    print("CamemBERT BLEU-3 Score: ", bleu_3.result())
    print("SacreBLEU BLEU Score: ", sacrebleu_bleu.result())
    print("SacreBLEU BLEU-1 Score: ", sacrebleu_bleu_1.result())
    print("SacreBLEU BLEU-2 Score: ", sacrebleu_bleu_2.result())
    print("SacreBLEU BLEU-3 Score: ", sacrebleu_bleu_3.result())
