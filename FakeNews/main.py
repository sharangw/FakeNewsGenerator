# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from flask import render_template, request, Flask
import tensorflow as tf
import numpy as np

app = Flask(__name__)

## Build Model

vocab_size = 152
embedding_dim = 256
rnn_units = 1024
BATCH_SIZE = 128


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
	tf.keras.layers.GRU(rnn_units,
						  return_sequences=True,
						  stateful=True,
						  recurrent_initializer='glorot_uniform'),
	tf.keras.layers.GRU(rnn_units,
						  return_sequences=True,
						  stateful=True,
						  recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


def createFakeNewsModel():

	model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

	print(model.summary())

	model.load_weights('tf-train-weights/ckpt_61')

	model.build(tf.TensorShape([1, None]))

	return model

def getDict():

	with open('bigstring.txt', 'r') as file:
		text = file.read()

	print(len(text))

	vocab = sorted(set(text))
	print('{} unique characters'.format(len(vocab)))

	# Creating a mapping from unique characters to indices
	char2idx = {u: i for i, u in enumerate(vocab)}

	idx2char = np.array(vocab)

	return char2idx, idx2char


def generateNews(headline, vocab_dict, idx2char):

	print("first line: ", headline)

	model = createFakeNewsModel()

	# news = generate_text(vocab_dict, idx2char, model, start_string=u"The news coming from Washington was a shock to everyone across the nation.")
	news = generate_text(vocab_dict, idx2char, model,
						 start_string=headline)

	return news

def generate_text(char2idx, idx2char, model, start_string):

  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 750

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in (range(num_generate)):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

##################

@app.route('/', methods=['GET', 'POST'])
def getHeadline():

	if request.method == "POST":
		headline = request.form.get("headline")
		print("headline: ", headline)
		news = headline + " and everyone died"

		createFakeNewsModel()

		vocab_dict, idx2char = getDict()

		news = generateNews(headline, vocab_dict, idx2char)
		print("news: ", news)

		return render_template("home.html", news=news)

	return render_template("home.html")

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080, debug=True)