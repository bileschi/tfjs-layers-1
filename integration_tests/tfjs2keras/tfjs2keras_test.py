# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess
import tempfile

import keras
import tensorflow as tf


def _call_command(command):
  print('Calling in %s: %s' % (os.getcwd(), ' '.join(command)))
  subprocess.check_call(command)


class Tfjs2KerasExportTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    print('Preparing TensorFlow.js...')
    cls._tmp_dir = tempfile.mkdtemp()
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(cwd, '..', '..'))
    _call_command(['yarn'])
    _call_command(['yarn', 'build'])
    _call_command(['yarn', 'link'])

    os.chdir(cwd)
    _call_command(['yarn'])
    _call_command(['yarn', 'link', '@tensorflow/tfjs-layers'])
    _call_command(['yarn', 'build'])
    _call_command(['node', './dist/tfjs_save.js', cls._tmp_dir])

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls._tmp_dir)

  def _loadAndTestModel(self, json_filename):
    """Load a Keras Model from artifacts generated by tensorflow.js.

    Currently this just loads the topology.
    TODO(cais): Once the IOHandler is ready for node.js filesystem, also load
      the weight values and compare predict() results.

    Args:
      json_path: Path to the model JSON file.
    """
    json_path = os.path.join(self._tmp_dir, json_filename)
    with tf.Graph().as_default(), tf.Session(), open(json_path, 'rt') as f:
      print('Loading %s' % json_path)
      model_json = f.read()
      keras.models.model_from_json(model_json)

  def testMLP(self):
    self._loadAndTestModel('mlp.json')

  def testCNN(self):
    self._loadAndTestModel('cnn.json')

  def testDepthwiseCNN(self):
    self._loadAndTestModel('depthwise_cnn.json')

  def testSimpleRNN(self):
    self._loadAndTestModel('simple_rnn.json')

  def testGRU(self):
    self._loadAndTestModel('gru.json')

  def testBidirectionalLSTM(self):
    self._loadAndTestModel('bidirectional_lstm.json')

  def testTimeDistributedLSTM(self):
    self._loadAndTestModel('time_distributed_lstm.json')

  def testOneDimensional(self):
    self._loadAndTestModel('one_dimensional.json')

  def testFunctionalMerge(self):
    self._loadAndTestModel('functional_merge.json')


if __name__ == '__main__':
  tf.test.main()

