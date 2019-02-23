# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import os

import modeling
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string(
    "bert_model_dir", None,
    "The directory containing the pre-trained BERT model. "
    "This includes the model architecture, vocabulary, and parameter checkpoint.")

layers_DEFAULT = "-1,-2,-3,-4"
flags.DEFINE_string("layers", layers_DEFAULT, "")

max_seq_length_DEFAULT = 128
flags.DEFINE_integer(
    "max_seq_length", max_seq_length_DEFAULT,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

group_count_DEFAULT = 1
flags.DEFINE_integer(
    "group_count", group_count_DEFAULT,
    "Number of consecutive lines to be grouped as rows of one tensor.")

do_lower_case_DEFAULT = True
flags.DEFINE_bool(
    "do_lower_case", do_lower_case_DEFAULT,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

batch_size_DEFAULT = 32
flags.DEFINE_integer("batch_size", batch_size_DEFAULT, "Batch size for predictions.")

use_tpu_DEFAULT = False
flags.DEFINE_bool("use_tpu", use_tpu_DEFAULT, "Whether to use TPU or GPU/CPU.")

master_DEFAULT = None
flags.DEFINE_string("master", master_DEFAULT,
                    "If using a TPU, the address of the master.")

num_tpu_cores_DEFAULT = 8
flags.DEFINE_integer(
    "num_tpu_cores", num_tpu_cores_DEFAULT,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

use_one_hot_embeddings_DEFAULT = False
flags.DEFINE_bool(
    "use_one_hot_embeddings", use_one_hot_embeddings_DEFAULT,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_vector_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
    return _bytes_feature(value.encode('utf-8'))

def extract(bert_model_dir, input_file, output_file, do_lower_case=do_lower_case_DEFAULT, layer_flags=layers_DEFAULT, group_count=group_count_DEFAULT, max_seq_length=max_seq_length_DEFAULT, batch_size=batch_size_DEFAULT, master=master_DEFAULT, num_tpu_cores=num_tpu_cores_DEFAULT, use_tpu=use_tpu_DEFAULT, use_one_hot_embeddings=use_one_hot_embeddings_DEFAULT):
  
  bert_config_file = os.path.join(bert_model_dir, 'bert_config.json')
  vocab_file = os.path.join(bert_model_dir, 'vocab.txt')
  init_checkpoint = os.path.join(bert_model_dir, 'bert_model.ckpt')
  
  layer_indexes = [int(x) for x in layer_flags.split(",")]
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)
  tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=num_tpu_cores,
          per_host_input_for_training=is_per_host))

  examples = read_examples(input_file)

  features = convert_examples_to_features(examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=batch_size)

  input_fn = input_fn_builder(features=features, seq_length=max_seq_length)
  group_number = 1
  vector_list = []
  vector_count = 0

  with tf.python_io.TFRecordWriter(output_file) as writer:
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]
      
      # First token should be '[CLS]' and contains the sentence embeddings
      token = feature.tokens[0]
      if token != '[CLS]':
        print('Unexpected token')
      
      vector = None
      
      for (j, layer_index) in enumerate(layer_indexes):
        layer_output = result["layer_output_0"]
        layer_features = [float(x) for x in layer_output[0:1].flat]
        
        if vector == None:
          vector = layer_features
        else:
          vector = vector + layer_features
          
      vector_list = vector_list + vector
      vector_count += 1
      
      if vector_count == group_count:
        if group_number % 10 == 0:
          print('Processed group:', group_number)
        
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'vector_length': _int64_feature(len(vector)),
              'vector_count': _int64_feature(group_count),
              'vector_list': _float_vector_feature(vector_list)
              }))
        writer.write(example.SerializeToString())
        group_number += 1
        vector_list = []
        vector_count = 0


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  extract_features(bert_model_dir=FLAGS.bert_model_dir, input_file=FLAGS.input_file, output_file=FLAGS.output_file, do_lower_case=FLAGS.do_lower_case, layer_flags=FLAGS.layers, group_count=FLAGS.group_count, max_seq_length=FLAGS.max_seq_length, batch_size=FLAGS.batch_size, master=FLAGS.master, num_tpu_cores=FLAGS.num_tpu_cores, use_tpu=FLAGS.use_tpu, use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_model_dir")
  flags.mark_flag_as_required("output_file")
  tf.app.run()