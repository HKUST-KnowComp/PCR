from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf

import util
import coref_ops
import conll
import metrics
import optimization
from bert import tokenization
from bert import modeling
from pytorch_to_tf import load_from_pytorch_checkpoint
import tqdm


class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.max_segment_len = config['max_segment_len']
    self.max_span_width = config["max_span_width"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    self.subtoken_maps = {}
    self.gold = {}
    self.eval_data = None # Load eval data lazily.
    self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
    self.tokenizer = tokenization.FullTokenizer(
                vocab_file=config['vocab_file'], do_lower_case=False)
    ### loading the frequenct spans
    self.freq_spans = json.load(open("./data/freq_spans.json"))
    

    input_props = []
    input_props.append((tf.int32, [None, None])) # input_ids.
    input_props.append((tf.int32, [None, None])) # input_mask
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None, None])) # Speaker IDs.
    input_props.append((tf.int32, [])) # Genre.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.
    input_props.append((tf.int32, [None])) # Sentence Map

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    # bert stuff
    tvars = tf.trainable_variables()
    # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
    # Get the assignment map from the tensorflow checkpoint. Depending on the extension, use TF/Pytorch to load weights.
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config['tf_checkpoint'])
    init_from_checkpoint = tf.train.init_from_checkpoint if config['init_checkpoint'].endswith('ckpt') else load_from_pytorch_checkpoint
    init_from_checkpoint(config['init_checkpoint'], assignment_map)
    print("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      # init_string)
      print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    num_train_steps = int(
                    self.config['num_docs'] * self.config['num_epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    self.global_step = tf.train.get_or_create_global_step()
    self.train_op = optimization.create_custom_optimizer(tvars,
                      self.loss, self.config['bert_learning_rate'], self.config['task_learning_rate'],
                      num_train_steps, num_warmup_steps, False, self.global_step, freeze=-1,
                      task_opt=self.config['task_optimizer'], eps=config['adam_eps'])

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        if self.config['single_example']:
          for example in train_examples:
            try:
              tensorized_example = self.tensorize_example(example, is_training=True)
              feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
              session.run(self.enqueue_op, feed_dict=feed_dict)
            except:
              a = "do nothing"
        else:
          examples = []
          for example in train_examples:
            tensorized = self.tensorize_example(example, is_training=True)
            if type(tensorized) is not list:
              tensorized = [tensorized]
            examples += tensorized
          random.shuffle(examples)
          print('num examples', len(examples))
          for example in examples:
            feed_dict = dict(zip(self.queue_input_tensors, example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() ]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)


  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

  def get_speaker_dict(self, speakers):
    speaker_dict = {'UNK': 0, '[SPL]': 1}
    for s in speakers:
      if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
        speaker_dict[s] = len(speaker_dict)
    return speaker_dict


  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = example["speakers"]
    # assert num_words == len(speakers), (num_words, len(speakers))
    speaker_dict = self.get_speaker_dict(util.flatten(speakers))
    sentence_map = example['sentence_map']


    max_sentence_length = self.max_segment_len
    text_len = np.array([len(s) for s in sentences])

    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
      sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
      sent_input_mask = [1] * len(sent_input_ids)
      sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
      while len(sent_input_ids) < max_sentence_length:
          sent_input_ids.append(0)
          sent_input_mask.append(0)
          sent_speaker_ids.append(0)
      input_ids.append(sent_input_ids)
      speaker_ids.append(sent_speaker_ids)
      input_mask.append(sent_input_mask)
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    speaker_ids = np.array(speaker_ids)
    assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))


    doc_key = example["doc_key"]
    self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
    self.gold[doc_key] = example["clusters"]
    genre = self.genres.get(doc_key[:2], 0)

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
    example_tensors = (input_ids, input_mask,  text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      if self.config['single_example']:
        return self.truncate_example(*example_tensors)
      else:
        offsets = range(self.config['max_training_sentences'], len(sentences), self.config['max_training_sentences'])
        tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
        return tensor_list
    else:
      return example_tensors

  def truncate_example(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, sentence_offset=None):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = input_ids.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    sentence_map = sentence_map[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return input_ids, input_mask, text_len, speaker_ids, genre, is_training,  gold_starts, gold_ends, cluster_ids, sentence_map

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]
    if self.config['use_prior']:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, c]
      distance_scores = util.projection(tf.nn.dropout(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), self.dropout), 1, initializer=tf.truncated_normal_initializer(stddev=0.02)) #[10, 1]
      antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1), antecedent_distance_buckets) # [k, c]
      fast_antecedent_scores += antecedent_distance_scores

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets


  def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
    model = modeling.BertModel(
      config=self.bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=False,
      scope='bert')
    all_encoder_layers = model.get_all_encoder_layers()
    mention_doc = model.get_sequence_output()

    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

    num_sentences = tf.shape(mention_doc)[0]
    max_sentence_length = tf.shape(mention_doc)[1]
    mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)
    num_words = util.shape(mention_doc, 0)
    antecedent_doc = mention_doc


    flattened_sentence_indices = sentence_map
    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]

    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

    candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts, candidate_ends) # [num_candidates, emb]
    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    tf.print(candidate_cluster_ids)
    ##################################################################################################
    ################### BELOW IS THE CODE THAT CHANGES THE MENTION INTO [GOLD MENTION] ###############
    ##################################################################################################
    if self.config["gold_mention"] == "yes":
    # Another method to use the gold candidates, changing the score with the right label (not zero) to maximum
      max_cand_score = tf.math.reduce_max(candidate_mention_scores)
      max_score_mask = tf.fill(tf.shape(candidate_mention_scores), max_cand_score)
      gold_mention_mask = tf.cast(candidate_cluster_ids,dtype=tf.bool)

      one_score_mask = tf.fill(tf.shape(candidate_mention_scores), 1.0)
      zero_score_mask = tf.fill(tf.shape(candidate_mention_scores), 0.0)

      # convert to max when the label is not zero(not in any cluster.)
      # candidate_mention_scores = tf.where(gold_mention_mask, max_score_mask, candidate_mention_scores)
      candidate_mention_scores = tf.where(gold_mention_mask, one_score_mask, zero_score_mask)

    
    ##################################################################################################
    ##################################################################################################

    # beam size
    k = tf.minimum(3900, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
    c = tf.minimum(self.config["max_top_antecedents"], k)
    # pull from beam
    top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               num_words,
                                               True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]

    # change the candidates to the gold candidate [GOLD MENTION]
    # here we might have some dimension issue
    # top_span_starts = gold_starts
    # top_span_ends = gold_ends

    # k = tf.minimum(k, top_span_starts.shape[0])
    # c = tf.minimum(self.config["max_top_antecedents"], k)

    top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), genre) # [emb]
    if self.config['use_metadata']:
      speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
      top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]i
    else:
        top_span_speaker_ids = None


    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
    num_segs, seg_len = util.shape(input_ids, 0), util.shape(input_ids, 1)
    word_segments = tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len])
    flat_word_segments = tf.boolean_mask(tf.reshape(word_segments, [-1]), tf.reshape(input_mask, [-1]))
    mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1) # [k, 1]
    antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents)) #[k, c]
    segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0, self.config['max_training_sentences'] - 1) if self.config['use_segment_distance'] else None #[k, c]
    if self.config['fine_grained']:
      for i in range(self.config["coref_depth"]):
        with tf.variable_scope("coref_layer", reuse=(i > 0)):
          top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
          top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance) # [k, c]
          top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
          top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
          attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
          with tf.variable_scope("f"):
            f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
            top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]
    else:
        top_antecedent_scores = top_fast_antecedent_scores

    top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]

    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
    top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
    same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
    non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
    pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
    top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
    loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
    loss = tf.reduce_sum(loss) # []

    return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss


  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
      head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [K, T]
      span_emb_list.append(head_attn_reps)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]


  def get_mention_scores(self, span_emb, span_starts, span_ends):
      with tf.variable_scope("mention_scores"):
        span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]
      if self.config['use_prior']:
        span_width_emb = tf.get_variable("span_width_prior_embeddings", [self.config["max_span_width"], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)) # [W, emb]
        span_width_index = span_ends - span_starts # [NC]
        with tf.variable_scope("width_scores"):
          width_scores =  util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [W, 1]
        width_scores = tf.gather(width_scores, span_width_index)
        span_scores += width_scores
      return span_scores


  def get_width_scores(self, doc, starts, ends):
    distance = ends - starts
    span_start_emb = tf.gather(doc, starts)
    hidden = util.shape(doc, 1)
    with tf.variable_scope('span_width'):
      span_width_emb = tf.gather(tf.get_variable("start_width_embeddings", [self.config["max_span_width"], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02)), distance) # [W, emb]
    scores = tf.reduce_sum(span_start_emb * span_width_emb, axis=1)
    return scores


  def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
      num_words = util.shape(encoded_doc, 0) # T
      num_c = util.shape(span_starts, 0) # NC
      doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1]) # [K, T]
      mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1), doc_range <= tf.expand_dims(span_ends, 1)) #[K, T]
      with tf.variable_scope("mention_word_attn"):
        word_attn = tf.squeeze(util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)
      mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
      return mention_word_attn


  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance=None):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), tf.to_int32(same_speaker)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)
    if segment_distance is not None:
      with tf.variable_scope('segment_distance', reuse=tf.AUTO_REUSE):
        segment_distance_emb = tf.gather(tf.get_variable("segment_distance_embeddings", [self.config['max_training_sentences'], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), segment_distance) # [k, emb]
      feature_emb_list.append(segment_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
    return slow_antecedent_scores # [k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index, (i, predicted_index)
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, global_step=None, official_stdout=False, keys=None, eval_mode=False):
    self.load_eval_data()

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    losses = []
    doc_keys = []
    num_evaluated= 0


    ##################################################################################################
    ################### WE FURTHER DETECT THE RESULTS SEPERATEDLY FOR P-P; NP-NP, P-NP ###############
    ##################################################################################################
    coref_predictions_pp = {}
    coref_predictions_pnp = {}
    coref_predictions_npnp = {}

    # span type
    coref_evaluator_pp = PairEvaluator()
    coref_evaluator_pnp = PairEvaluator()
    coref_evaluator_npnp = PairEvaluator()
    coref_evaluator_all = PairEvaluator()

    num_coref_pp = 0
    num_coref_pnp = 0
    num_coref_npnp = 0
    num_coref_all = 0

    # span freq
    coref_evaluator_freq = PairEvaluator()
    coref_evaluator_rare = PairEvaluator()
    
    num_coref_freq = 0
    num_coref_rare = 0

    # pron type
    coref_evaluators_type = dict()
    coref_evaluators_type["demo"], coref_evaluators_type["pos"], coref_evaluators_type["third"] = PairEvaluator(), PairEvaluator(), PairEvaluator()
    nums_coref_type = dict()
    nums_coref_type["demo"], nums_coref_type["pos"], nums_coref_type["third"] = 0, 0, 0

    count = 0 

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      try:
        # count += 1
        # if count == 10:
        #   break
        _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
        feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
        # if tensorized_example[0].shape[0] <= 9:
        if keys is not None and example['doc_key'] not in keys:
          # print('Skipping...', example['doc_key'], tensorized_example[0].shape)
          continue
        doc_keys.append(example['doc_key'])
        loss, (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores) = session.run([self.loss, self.predictions], feed_dict=feed_dict)
        # losses.append(session.run(self.loss, feed_dict=feed_dict))
        losses.append(loss)
        predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)

        coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)

        if example_num % 10 == 0:
          print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        #####################################################################################
        # Evaluate on three different settings: NP-NP, NP-P, P-P by using a different cluster
        #####################################################################################

        # Span Type
        flatten_sentences = util.flatten(example["sentences"])
        gold_pp_pairs, gold_pnp_pairs, gold_npnp_pairs, num_pp_pairs, num_pnp_pairs, num_npnp_pairs, num_relation = self.cluster_to_pairs(example["clusters"], flatten_sentences)
        # predicted_clusters = coref_predictions[example["doc_key"]]
        pred_pp_pairs, pred_pnp_pairs, pred_npnp_pairs, _, _, _, _ = self.cluster_to_pairs(coref_predictions[example["doc_key"]], flatten_sentences)

        # Span Frequency
        gold_freq_pnp_pairs, gold_rare_pnp_pairs, num_freq_pairs, num_rare_pairs = self.cluster_to_pair_frequent(example["clusters"], flatten_sentences)
        pred_freq_pnp_pairs, pred_rare_pnp_pairs, _, _ = self.cluster_to_pair_frequent(coref_predictions[example["doc_key"]], flatten_sentences)

        # pronoun type demo, pos, third
        gold_type_pairs, gold_type_nums = self.cluster_to_pair_detailed_pronoun(example["clusters"], flatten_sentences)
        pred_type_pairs, pred_type_nums = self.cluster_to_pair_detailed_pronoun(coref_predictions[example["doc_key"]], flatten_sentences)
        
        for pron_type in ["demo", "pos", "third"]:
          coref_evaluators_type[pron_type].update(gold_type_pairs[pron_type], pred_type_pairs[pron_type])
          nums_coref_type[pron_type] += gold_type_nums[pron_type]

        all_gold = gold_pp_pairs + gold_pnp_pairs + gold_npnp_pairs
        all_pred = pred_pp_pairs + pred_pnp_pairs + pred_npnp_pairs

        coref_evaluator_pp.update(pred_pp_pairs, gold_pp_pairs)
        coref_evaluator_pnp.update(pred_pnp_pairs, gold_pnp_pairs)
        coref_evaluator_npnp.update(pred_npnp_pairs, gold_npnp_pairs)
        coref_evaluator_all.update(all_pred, all_gold)

        coref_evaluator_freq.update(pred_freq_pnp_pairs, gold_freq_pnp_pairs)
        coref_evaluator_rare.update(pred_rare_pnp_pairs, gold_rare_pnp_pairs)

        num_coref_pp += num_pp_pairs
        num_coref_pnp += num_pnp_pairs
        num_coref_npnp += num_npnp_pairs
        num_coref_all = num_coref_all + num_pp_pairs + num_pnp_pairs + num_npnp_pairs
        num_coref_freq += num_freq_pairs
        num_coref_rare += num_rare_pairs
      except:
        a = "do nothing"

    summary_dict = {}

    self.print_prf(coref_evaluator_pp, summary_dict, doc_keys, "PP", num_coref_pp)
    self.print_prf(coref_evaluator_pnp, summary_dict, doc_keys, "PNP", num_coref_pnp)
    self.print_prf(coref_evaluator_npnp, summary_dict, doc_keys, "NPNP", num_coref_npnp)

    self.print_prf(coref_evaluator_freq, summary_dict, doc_keys, "FREQ", num_coref_freq)
    self.print_prf(coref_evaluator_rare, summary_dict, doc_keys, "RARE", num_coref_rare)

    for pron_type in ["demo", "pos", "third"]:
      self.print_prf(coref_evaluators_type[pron_type], summary_dict, doc_keys, pron_type, nums_coref_type[pron_type])
    
    self.print_prf(coref_evaluator_all, summary_dict, doc_keys, "ALL_PAIRS", num_coref_all)

    #######################################################################################

    # summary_dict = {}

    print("The evaluatoin results for all clusters")
    print("The number of pairs is "+ str(num_coref_all))
    
    p,r,f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
    summary_dict["Average precision (py)"] = p
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    print("Average recall (py): {:.2f}%".format(r * 100))

    if eval_mode:
      conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, self.subtoken_maps, official_stdout)
      average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
      summary_dict["Average F1 (conll)"] = average_f1
      print("Average F1 (conll): {:.2f}%".format(average_f1))


    return util.make_summary(summary_dict), f

  def cluster_to_pairs(self, clusters, flatten_sentences):
    # distribute clusters into p-p; p-np; np-np
    pp_pairs = []
    pnp_pairs = []
    npnp_pairs = []

    num_pp_pairs = 0
    num_pnp_pairs = 0
    num_npnp_pairs = 0

    for cluster in clusters:
      # print(cluster)
      for i in range(len(cluster)):
        for j in range(len(cluster)):
          if i < j:
            const1 = list(cluster[i])
            const2 = list(cluster[j])
            # print(const1)
            # print(const2)
            const1_tokens = flatten_sentences[const1[0]:const1[1]+1]
            const2_tokens = flatten_sentences[const2[0]:const2[1]+1]

            if self.check_if_pronoun(const1_tokens):
              if self.check_if_pronoun(const2_tokens):
                # print("pp")
                pp_pairs.append([const1,const2])
                num_pp_pairs += 1
              else:
                # print("pnp")
                pnp_pairs.append([const1,const2])
                num_pnp_pairs += 1
            else:
              if self.check_if_pronoun(const2_tokens):
                # print("pnp")
                pnp_pairs.append([const1,const2])
                num_pnp_pairs += 1
              else:
                # print("npnp")
                npnp_pairs.append([const1,const2])
                num_npnp_pairs += 1


    # for cluster in clusters:
    #   p_list = []
    #   np_list = []
      
    #   for const in cluster:
    #     const_tokens = flatten_sentences[const[0]:const[1]+1]
    #     if self.check_if_pronoun(const_tokens):
    #       # is a pronoun
    #       p_list.append(list(const))
    #     else:
    #       np_list.append(list(const))

    #   for p in p_list:
    #     for np in np_list:
    #       pnp_pairs.append([p,np])
    #       num_pnp_pairs += 1

    #   for i in range(len(p_list)-1):
    #     for j in range(i+1,len(p_list)):
    #       p1 = p_list[i]
    #       p2 = p_list[j]
    #       pp_pairs.append([p1,p2])
    #       num_pp_pairs += 1

    #   for i in range(len(np_list)-1):
    #     for j in range(i+1,len(np_list)):
    #       np1 = np_list[i]
    #       np2 = np_list[j]
    #       npnp_pairs.append([np1,np2])
    #       num_npnp_pairs += 1

    num_relation = num_pp_pairs + num_pnp_pairs + num_npnp_pairs

    return pp_pairs, pnp_pairs, npnp_pairs, num_pp_pairs, num_pnp_pairs, num_npnp_pairs, num_relation
      
  
  def cluster_to_pair_frequent(self, clusters, flatten_sentences):
    # we only care about the NP-P pairs
    freq_pnp_pairs = []
    rare_pnp_pairs = []

    num_freq_pairs = 0
    num_rare_pairs = 0

    for cluster in clusters:
      # print(cluster)
      for i in range(len(cluster)):
        for j in range(len(cluster)):
          if i < j:
            const1 = list(cluster[i])
            const2 = list(cluster[j])
            # print(const1)
            # print(const2)
            const1_tokens = flatten_sentences[const1[0]:const1[1]+1]
            const2_tokens = flatten_sentences[const2[0]:const2[1]+1]

            if self.check_if_pronoun(const1_tokens):
              if self.check_if_pronoun(const2_tokens):
                # print("pp")
                continue
              else:
                # print("pnp"), const1 is a pron
                if self.check_if_frequent(const2_tokens):
                  freq_pnp_pairs.append([const1,const2])
                  num_freq_pairs += 1
                else:
                  rare_pnp_pairs.append([const1,const2])
                  num_rare_pairs += 1
            else:
              if self.check_if_pronoun(const2_tokens):
                # print("pnp"), const2 is a pronoun
                if self.check_if_frequent(const1_tokens):
                  freq_pnp_pairs.append([const1,const2])
                  num_freq_pairs += 1
                else:
                  rare_pnp_pairs.append([const1,const2])
                  num_rare_pairs += 1
              else:
                # print("npnp")
                continue
    
    return freq_pnp_pairs, rare_pnp_pairs, num_freq_pairs, num_rare_pairs


  def cluster_to_pair_detailed_pronoun(self, clusters, flatten_sentences):
    # we only care about the NP-P pairs
    demonstrative_pronouns = ['this', 'these', 'that', 'those', 'This', 'These', 'That', 'Those']
    possessive_pronouns = ['his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']
    third_personal_pronouns = ['he', 'him', 'she', 'her', 'they', 'them', "it", 'He', 'Him', 'She', 'Her', 'They', 'Them', "It"]
    pron = dict()
    pron["demo"] = demonstrative_pronouns
    pron["pos"] = possessive_pronouns
    pron["third"] = third_personal_pronouns
    
    pairs = dict()
    pairs["demo"] = []
    pairs["pos"] = []
    pairs["third"] = []

    # num_freq_pairs = 0
    # num_rare_pairs = 0
    nums = dict()
    nums["demo"], nums["pos"], nums["third"] = 0, 0, 0

    for cluster in clusters:
      # print(cluster)
      for i in range(len(cluster)):
        for j in range(len(cluster)):
          if i < j:
            const1 = list(cluster[i])
            const2 = list(cluster[j])
            # print(const1)
            # print(const2)
            const1_tokens = flatten_sentences[const1[0]:const1[1]+1]
            const2_tokens = flatten_sentences[const2[0]:const2[1]+1]

            if self.check_if_pronoun(const1_tokens):
              if self.check_if_pronoun(const2_tokens):
                # print("pp")
                continue
              else:
                # print("pnp"), const1 is a pron
                for pron_type in ["demo", "pos", "third"]:
                  if const1_tokens[0] in pron[pron_type] and len(const1_tokens) == 1:
                    pairs[pron_type].append([const1,const2])
                    nums[pron_type] += 1
                    # print(const1_tokens)
                    # break
            else:
              if self.check_if_pronoun(const2_tokens):
                # print("pnp"), const2 is a pronoun
                for pron_type in ["demo", "pos", "third"]:
                  if const2_tokens[0] in pron[pron_type] and len(const2_tokens) == 1:
                    pairs[pron_type].append([const1,const2])
                    nums[pron_type] += 1
                    # print(const2_tokens)
                    # break
              else:
                # print("npnp")
                continue
    
    return pairs, nums

  def check_if_frequent(self,tokens):
    if tokens in self.freq_spans:
      return True
    else: 
      return False


  def check_if_pronoun(self,tokens):
    # a function supporting the NP-P detection
    # personal
    pronouns = ["i","me","my","mine","myself","you","your","yours","yourself","he","him","his","himself"]
    pronouns.extend(["she","her","hers","herself","it","its","itself","we","us","our","ours","ourselves"])
    pronouns.extend(["youselves","they","them","their","theirs","themselves"])
    
    # relative
    pronouns.extend(["this","that","these","those"])

    # interrogative
    pronouns.extend(["which","who","whom","whose","whichever","whoever","whomever"])

    # indefinite
    pronouns.extend(["anybody","anyone","anything","each","either","everybody","everyone","everything"])
    pronouns.extend(["neither","nobody","none","nothing","one","somebody","someone","something"])

    pronouns.extend(["both","few","many","serveral","all","any","most","some"])


    ### capital
    pronouns.extend(["I","Me","My","Mine","Myself","You","Your","Yours","Yourself","He","Him","His","Himself"])
    pronouns.extend(["She","Her","Hers","Herself","It","Its","Itself","We","Us","Our","Ours","Ourselves"])
    pronouns.extend(["Youselves","They","Them","Their","Theirs","Themselves"])
    
    # relative
    pronouns.extend(["This","That","These","Those"])

    # interrogative
    pronouns.extend(["Which","Who","Whom","Whose","Whichever","Whoever","Whomever"])

    # indefinite
    pronouns.extend(["Anybody","Anyone","Anything","Each","Either","Everybody","Everyone","Everything"])
    pronouns.extend(["Neither","Nobody","None","Nothing","One","Somebody","Someone","Something"])

    pronouns.extend(["Both","Few","Many","Serveral","All","Any","Most","Some"])
    # pronoun_ids = self.tokenizer.convert_tokens_to_ids(pronouns)

    if len(tokens) == 1 and tokens[0] in pronouns:# tokens[0].lower() in pronouns:
      # print("Yes")
      return True
    else:
      return False

  def print_prf(self, coref_evaluator, summary_dict, doc_keys, name, num_pairs):
      print("The evaluatoin results for "+name)
      p_num, p_den, r_num, r_den = coref_evaluator.get_counts()
      print("The number of pairs under this setting is "+ str(num_pairs))
      print("The number of pairs under this setting is "+ str(r_den))
      p,r,f = coref_evaluator.get_prf()
      summary_dict["Average F1 (py_"+name+")"] = f
      print(name+": Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
      summary_dict["Average precision (py_"+name+")"] = p
      print(name+": Average precision (py): {:.2f}%".format(p * 100))
      summary_dict["Average recall (py_"+name+")"] = r
      print(name+": Average recall (py): {:.2f}%".format(r * 100))
      print("-----------------------------------------------------------")


class PairEvaluator(object):
    def __init__(self):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0

    def update(self, predicted_pairs, gold_pairs):
      pn,pd,rn,rd = 0,len(predicted_pairs),0,len(gold_pairs)
      # pair = [const,const]
      for p_pair in predicted_pairs:
        for g_pair in gold_pairs:
          if (p_pair[0] in g_pair) and (p_pair[1] in g_pair):
            pn += 1
            rn += 1

      self.p_num += pn
      self.p_den += pd
      self.r_num += rn
      self.r_den += rd

    def get_f1(self):
        return metrics.f1(self.p_num, self.p_den, self.r_num, self.r_den)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den
