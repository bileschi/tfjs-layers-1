// Show off VocabLayer when  you get to this point.


// tslint:disable-next-line:max-line-length
import {randomNormal, Tensor, tensor2d, test_util, ones, zeros} from '@tensorflow/tfjs-core';
import {expectValuesInRange} from '@tensorflow/tfjs-core/dist/test_util';

import * as tfl from '../index';
import {initializers} from '../index';
import {getInitializer} from '../initializers';
// tslint:disable-next-line:max-line-length
import {UnitVarianceOptimizer, VocabLayerOptimizer, ZeroMeanOptimizer, ZeroMean} from '../preprocess-layers/preprocess_core';
import {describeMathCPU, describeMathCPUAndGPU} from '../utils/test_utils';
import {expectTensorsClose} from '../utils/test_utils';
import {Sequential} from '../models';

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
  fit('Fit a model with just a vocab layer.', async done => {
    // Define a Sequential model with just one layer:  Vocabulary.
    const vocabModel = tfl.sequential({
      layers: [tfl.layers.vocab({
        knownVocabSize: 4,
        hashVocabSize: 1,
        // TODO(bileschi): Use tfl.preprocessing.vocabLayerOptimizer instead
        // of direct access.
        optimizer: new tfl.preprocessing.VocabLayerOptimizer(),
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
    await vocabModel.fit(trainInputs, null, {batchSize: 1, epochs: 1});
    const testInputs = tfl.preprocessing.stringTensor2d(
        [['a', 'b'], ['c', 'd'], ['hello', 'world']]);
    const testOutputs = vocabModel.predict(testInputs);
    test_util.expectArraysClose(
        testOutputs as Tensor,
        tensor2d([[0, 1], [2, 3], [4, 4]], [3, 2], 'int32'));
    done();
  });

  it('Fit vocab layer overrides initializer', async done => {
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
    await vocabModel.fit(trainInputs, null, {batchSize: 1, epochs: 1});
    const testInputs = tfl.preprocessing.stringTensor2d(
        [['a', 'a'], ['a', 'a'], ['hello', 'world']]);
    const testOutputs = vocabModel.predict(testInputs);
    test_util.expectArraysClose(
        testOutputs as Tensor,
        tensor2d([[0, 0], [0, 0], [4, 4]], [3, 2], 'int32'));
    done();
  });
});


describeMathCPU('Preprocess with zeroMean & unitVariance', () => {
  it('Fit a model with two preprocessing layers.', async done => {
    // Define a Sequential model with zeroMean & unitVariance
    const normalizationModel = tfl.sequential({
      layers: [
        tfl.layers.zeroMean({
          optimizer: new ZeroMeanOptimizer(),
          inputShape: [3]  // 3 numbers per example
        }),
        tfl.layers.unitVariance({optimizer: new UnitVarianceOptimizer()})
      ]
    });
    // Compile the model.
    // TODO(bileschi): It should be possible to compile with null / null
    // here.
    normalizationModel.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // 1000 random samples with mean 1234 and stddev 13.
    const mean = 1234;
    const std = 13;
    const trainInputs = randomNormal([100, 3], mean, std);
    // Fit the model to the provided samples.
    await normalizationModel.fit(trainInputs, null, {batchSize: 10, epochs: 1});
    const testInputs = tensor2d([[mean - std, mean, mean + std]]);
    const testOutputs = normalizationModel.predict(testInputs);
    // Expect an accurate prediction of the stddev (within 1%)
    test_util.expectArraysClose(
      testOutputs as Tensor, tensor2d([[-1, 0, 1]], [1, 3], 'float32'), 0.5);
    done();
  });

  it('Fit a zero mean on image shaped data.', async done => {
    // Define a Sequential model with zeroMean
    const normalizationModel = tfl.sequential({
      layers: [
        tfl.layers.zeroMean({
          optimizer: new ZeroMeanOptimizer(),
          inputShape: [224, 224, 3]  // 3 numbers per example
        })
      ]
    });
    // Compile the model.
    // TODO(bileschi): It should be possible to compile with null / null
    // here.
    normalizationModel.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // 1000 random samples with mean 1234 and stddev 13.
    const trainInputs = ones([4, 224, 224, 3]);
    // Fit the model to the provided samples.
    await normalizationModel.fit(trainInputs, null, {batchSize: 10, epochs: 1});
    const testInputs = ones([1, 224, 224, 3]);
    const testOutputs = normalizationModel.predict(testInputs);
    // Expect an accurate prediction of the stddev (within 1%)
    test_util.expectArraysClose(
      testOutputs as Tensor, zeros([1, 224, 224, 3]), 0.);
    done();
  });

  it('Serialize and deserialize', () => {
    const normalizationModel = tfl.sequential({
      layers: [
        tfl.layers.zeroMean({
          optimizer: new ZeroMeanOptimizer(),
          inputShape: [3]  // 3 numbers per example
        }),
        tfl.layers.unitVariance({optimizer: new UnitVarianceOptimizer()})
      ]
    });
    const config = normalizationModel.getConfig();
    const modelPrime = Sequential.fromConfig(Sequential, config) as Sequential;
    expect(modelPrime.getConfig()).toEqual(normalizationModel.getConfig());
    expect(modelPrime.layers[0] instanceof ZeroMean).toBe(true);
  });
});

