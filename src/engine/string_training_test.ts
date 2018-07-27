// Show off VocabLayer when  you get to this point.


import {Tensor, tensor2d, test_util} from '@tensorflow/tfjs-core';
import {expectValuesInRange} from '@tensorflow/tfjs-core/dist/test_util';

import * as tfl from '../index';
import {initializers} from '../index';
import {getInitializer} from '../initializers';
import {VocabLayerOptimizer} from '../preprocess-layers/preprocess_core';
import {describeMathCPU, describeMathCPUAndGPU} from '../utils/test_utils';
import {expectTensorsClose} from '../utils/test_utils';

describeMathCPUAndGPU('String preproc : Model.predict', () => {
  it('basic model usage: Sequential predict', () => {
    // Define the vocabulary initializer
    const vocabInitializer = initializers.knownVocab(
        {strings: ['hello', 'world', 'こんにちは', '世界']});
    // Define a Sequential model with just a vocab layer
    const knownVocabSize = 4;
    const hashVocabSize = 1;
    const vocabModel = tfl.sequential({
      layers: [tfl.layers.vocab({
        name: 'myVocabLayer',
        knownVocabSize,
        hashVocabSize,
        vocabInitializer,
        inputShape: [2]  // two words per example
      })]
    });
    // Matches known words.
    const x = tfl.preprocessing.stringTensor2d(
        [['world', 'hello'], ['世界', 'こんにちは']], [2, 2]);
    const y = vocabModel.predict(x) as Tensor;
    const yExpected = tensor2d([[1, 0], [3, 2]], [2, 2], 'int32');
    expectTensorsClose(y, yExpected);
    // Handles unknown words.
    const xOutOfVocab = tfl.preprocessing.stringTensor2d(
        [['these', 'words'], ['are', 'out'], ['of', 'vocabulary']], [3, 2]);
    const yOutOfVocab = vocabModel.predict(xOutOfVocab) as Tensor;
    // Out-of-vocab words should hash to buckets after the knownVocab
    expectValuesInRange(
        yOutOfVocab, knownVocabSize, knownVocabSize + hashVocabSize);
  });

  it('basic model usage: Functional predict', () => {
    // Define the vocabulary initializer
    const vocabInitializer = initializers.knownVocab(
        {strings: ['hello', 'world', 'こんにちは', '世界']});
    // Define a functional-style model with just a vocab layer
    const knownVocabSize = 4;
    const hashVocabSize = 1;
    const input = tfl.input({shape: [2], dtype: 'string'});
    const vocabLayer = tfl.layers.vocab({
      name: 'myVocabLayer',
      knownVocabSize,
      hashVocabSize,
      vocabInitializer,
      inputShape: [2]  // two words per example
    });
    const outputSymbolic = vocabLayer.apply(input) as tfl.SymbolicTensor;
    const vocabModel = tfl.model({inputs: input, outputs: outputSymbolic});
    // Matches known words.
    const x = tfl.preprocessing.stringTensor2d(
        [['world', 'hello'], ['世界', 'こんにちは']], [2, 2]);
    const y = vocabModel.predict(x) as Tensor;
    const yExpected = tensor2d([[1, 0], [3, 2]], [2, 2], 'int32');
    expectTensorsClose(y, yExpected);
    // Handles unknown words.
    const xOutOfVocab = tfl.preprocessing.stringTensor2d(
        [['these', 'words'], ['are', 'out'], ['of', 'vocabulary']], [3, 2]);
    const yOutOfVocab = vocabModel.predict(xOutOfVocab) as Tensor;
    // Out-of-vocab words should hash to buckets after the knownVocab
    expectValuesInRange(
        yOutOfVocab, knownVocabSize, knownVocabSize + hashVocabSize);
  });
});

describeMathCPU('String Preproc Model.fit', () => {
  it('Fit a model with just a vocab layer.', async done => {
    // Define a Sequential model with just one layer:  Vocabulary.
    const vocabModel = tfl.sequential({
      layers: [tfl.layers.vocab({
        knownVocabSize: 4,
        hashVocabSize: 1,
        // TODO(bileschi): Use tfl.preprocessing.vocabLayerOptimizer instead
        // of direct access.
        optimizer: new VocabLayerOptimizer(),
        inputShape: [2]  // two words per example
      })]
    });
    // Compile the model.
    // TODO(bileschi): It should be possible to compile with null / null
    // here.
    vocabModel.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    const trainInputs = tfl.preprocessing.stringTensor2d(
        [['a', 'a'], ['b', 'b'], ['c', 'c'], ['d', 'd']]);
    // Fit the model to a tensor of strings.
    await vocabModel.fit(trainInputs, null, {batchSize: 1, epochs: 1})
        .then(history => {
          const testInputs = tfl.preprocessing.stringTensor2d(
              [['a', 'b'], ['c', 'd'], ['hello', 'world']]);
          const testOutputs = vocabModel.predict(testInputs);
          test_util.expectArraysClose(
              testOutputs as Tensor,
              tensor2d([[0, 1], [2, 3], [4, 4]], [3, 2], 'int32'));
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Fit vocab layer overrides initializer',
      async done => {
        // Define a Sequential model with just one layer:  Vocabulary.
        const vocabModel = tfl.sequential({
          layers: [tfl.layers.vocab({
            name: 'myVocabLayer',
            knownVocabSize: 4,
            hashVocabSize: 1,
            vocabInitializer: getInitializer({
              className: 'KnownVocab',
              config: {strings: ['hello', 'world', 'こんにちは', '世界']}
            }),
            // TODO(bileschi): Use tfl.preprocessing.vocabLayerOptimizer instead
            // of direct access.
            optimizer: new VocabLayerOptimizer(),
            inputShape: [2]  // two words per example
          })]
        });
        // Compile the model.
        // TODO(bileschi): It should be possible to compile with null / null
        // here.
        vocabModel.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
        const trainInputs = tfl.preprocessing.stringTensor2d(
            [['a', 'a'], ['a', 'a'], ['a', 'a'], ['a', 'a']]);
        // Fit the model to a tensor of strings.
        await vocabModel.fit(trainInputs, null, {batchSize: 1, epochs: 1})
            .then(history => {
              const testInputs = tfl.preprocessing.stringTensor2d(
                  [['a', 'a'], ['a', 'a'], ['hello', 'world']]);
              const testOutputs = vocabModel.predict(testInputs);
              test_util.expectArraysClose(
                  testOutputs as Tensor,
                  tensor2d([[0, 0], [0, 0], [4, 4]], [3, 2], 'int32'));
              done();
            })
            .catch(err => {
              done.fail(err.stack);
            });
      });
});
