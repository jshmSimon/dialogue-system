from flask import Flask, request, session
from hred_trainer import HREDChatbot
import tensorflow as tf
import nltk
from data_utils import read_vocab
import numpy as np
from config import HConfig

app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


hredchatbot = HREDChatbot()
hredckpt = tf.train.get_checkpoint_state(hredchatbot.config.checkpoint_dir)
if hredckpt and tf.train.checkpoint_exists(hredckpt.model_checkpoint_path):
    print(" # Restoring model parameters from %s." % hredckpt.model_checkpoint_path)
    hredchatbot.model.saver.restore(hredchatbot.sess, hredckpt.model_checkpoint_path)


def process_message(message):
    message = nltk.word_tokenize(message)
    word_to_id, id_to_word = read_vocab()
    unk_id = word_to_id["<unk>"]
    message_id_list = [word_to_id.get(word, unk_id) for word in message]
    message_id_list = message_id_list + [0] * HConfig.max_length
    message_id_list = [1] + message_id_list[:HConfig.max_length]
    return message_id_list


def multipul_chat(message_id_lists):
    print('messages: {}'.format(message_id_lists))
    enc_inp = np.array(message_id_lists)
    dec_inp = enc_inp
    dec_tar = enc_inp
    test_outputs = hredchatbot.model.infer_session(hredchatbot.sess, enc_inp, dec_inp, dec_tar)
    infer_sample_id = test_outputs["infer_sample_id"]
    gener = infer_sample_id[1]
    gener_list = [hredchatbot.id_to_word.get(idx, "<unk>") for idx in gener[0]]
    response = gener_list[:gener_list.index('<eos>')]
    response = " ".join(response)
    return response


@app.route('/chatbot/')
def chat():
    message = request.args.get('message')
    message_id_list = process_message(message)
    if 'pre_message_id_lists' in session:
        pre_message_id_lists = eval(session['pre_message_id_lists'])
        message_id_lists = pre_message_id_lists + [[message_id_list]]
        response = multipul_chat(message_id_lists)
        session['pre_message_id_lists'] = str(message_id_lists)
    else:
        pre_message_id_lists = [[message_id_list]]
        message_id_lists = pre_message_id_lists + [[message_id_list]]
        response = multipul_chat(message_id_lists)
        session['pre_message_id_lists'] = str(pre_message_id_lists)
    return response

@app.route('/chatbot/end/')
def chat_end():
    if 'pre_message_id_lists' in session:
        session.pop('pre_message_id_lists')
    return 'This conversation is end.t'


if __name__ == '__main__':
    app.run(host='0.0.0.0')