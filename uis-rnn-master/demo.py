# run: python3 demo.py --train_iteration=1000 -l=0.001
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np
import torch
import uisrnn


SAVED_MODEL_NAME = 'saved_model.uisrnn'

train_data = np.load('./data/toy_training_data.npz', allow_pickle=True)
test_data = np.load('./data/toy_testing_data.npz', allow_pickle=True)
#2-dim numpy array of type float (Dim1- length of sequence; Dim2- Feature size of observation = d-vector)
train_sequence = train_data['train_sequence']

  #List with same length as train_sequences;
  # Each element of train_cluster_ids is a 1-dim list or numpy array of strings,
  # containing the ground truth labels for the corresponding sequence in train_sequences
train_cluster_id = train_data['train_cluster_id']
test_sequences = test_data['test_sequences'].tolist()
test_cluster_ids = test_data['test_cluster_ids'].tolist()

print(train_sequence.shape)
#print(train_sequence[1])
# def diarization_experiment(model_args, training_args, inference_args):
#   """Experiment pipeline.
#
#   Load data --> train model --> test model --> output result
#
#   Args:
#     model_args: model configurations
#     training_args: training configurations
#     inference_args: inference configurations
#   """
#
#   predicted_cluster_ids = []
#   test_record = []
#
#   train_data = np.load('./data/toy_training_data.npz', allow_pickle=True)
#   test_data = np.load('./data/toy_testing_data.npz', allow_pickle=True)
#
#   #2-dim numpy array of type float (Dim1- length of sequence; Dim2- Feature size of observation = d-vector)
#   train_sequence = train_data['train_sequence']
#
#   #List with same length as train_sequences;
#   # Each element of train_cluster_ids is a 1-dim list or numpy array of strings,
#   # containing the ground truth labels for the corresponding sequence in train_sequences
#   train_cluster_id = train_data['train_cluster_id']
#   test_sequences = test_data['test_sequences'].tolist()
#   test_cluster_ids = test_data['test_cluster_ids'].tolist()
#
#   model = uisrnn.UISRNN(model_args)
#
#   # Training.
#   # If we have saved a mode previously, we can also skip training by
#   # calling：
#   # model.load(SAVED_MODEL_NAME)
#   model.fit(train_sequence, train_cluster_id, training_args)
#
#   model.save(SAVED_MODEL_NAME)
#
#   # Testing.
#   # You can also try uisrnn.parallel_predict to speed up with GPU.
#   # But that is a beta feature which is not thoroughly tested, so
#   # proceed with caution.
#   for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
#     predicted_cluster_id = model.predict(test_sequence, inference_args)
#     predicted_cluster_ids.append(predicted_cluster_id)
#     accuracy = uisrnn.compute_sequence_match_accuracy(
#         test_cluster_id, predicted_cluster_id)
#     test_record.append((accuracy, len(test_cluster_id)))
#     print('Ground truth labels:')
#     print(test_cluster_id)
#     print('Predicted labels:')
#     print(predicted_cluster_id)
#     print('-' * 80)
#
#   output_string = uisrnn.output_result(model_args, training_args, test_record)
#
#   print('Finished diarization experiment')
#   print(output_string)
#
# def main():
#   """The main function."""
#   model_args, training_args, inference_args = uisrnn.parse_arguments()
#   diarization_experiment(model_args, training_args, inference_args)
#
# if __name__ == '__main__':
#   main()
