/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * TensorFlow.js Layers: Basic Layers.
 */

// tslint:disable:max-line-length
import {add, cast, div, oneHot, scalar, serialization, sub, Tensor, tensor, tidy, Variable, variable, zerosLike} from '@tensorflow/tfjs-core';
import {ConfigDict, Serializable} from '@tensorflow/tfjs-core/dist/serialization';

import {Layer, LayerConfig} from '../engine/topology';
import {ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier} from '../initializers';
import {Kwargs, Shape} from '../types';
import * as type_utils from '../utils/types_utils';
import {LayerVariable} from '../variables';

import {StringTensor} from './string_tensor';

export interface OneHotLayerConfig extends LayerConfig {
  /** Positive integer, dimensionality of the output space. */
  units: number;
}

/**
 * Preprocessing layers are distinct from Layers in that they are not
 * fit via back-propagation. Individual preprocessing layers may implement their
 * own `layer.fitUnsupervised()` method which will be called during
 * `Model.fit()`.
 */
export abstract class PreprocessingLayer extends Layer {
  // If set, this optimizer will be used to update the preprocessing layer
  // during `layer.fit` and `model.fitUnsupervised()`.
  protected optimizer: {} = null;

  // Assumes if an optimizer is set, then the preprocessing layer is trainable.
  public isTrainable(): boolean {
    return ((this.optimizer !== undefined) && (this.optimizer !== null));
  }

  // Call to update internal representation to more closely match x.
  public fitUnsupervised(x: Tensor|StringTensor|
                         Array<Tensor|StringTensor>): void {}
}

/**
 * Requires input of shape [batch] or [batch, 1].  Produces output of shape
 * [batch, units]
 */
export class OneHot extends PreprocessingLayer {
  static className = 'OneHot';
  readonly units: number;

  constructor(config: OneHotLayerConfig) {
    super(config);
    this.units = config.units;

    this.inputSpec = [{minNDim: 1}];
  }

  public build(inputShape: Shape|Shape[]): void {
    this.inputSpec = [{minNDim: 1}];
    this.built = true;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = type_utils.getExactlyOneShape(inputShape) as Shape;
    const outputShape = [inputShape[0], this.units];
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);

      const input = type_utils.getExactlyOneTensor(inputs);
      if ((input.rank !== 1) && (input.rank !== 2)) {
        throw new ValueError(
            `OneHot expects input of either rank-1, or rank-2` +
            ` but got input tensor with shape ${input.shape}`);
      }
      if ((input.rank === 2) && (input.shape[1] !== 1)) {
        throw new ValueError(
            `OneHot expects rank-2 inputs to have shape ` +
            ` [?, 1] but got input tensor with shape ${input.shape}`);
      }
      const output = oneHot(input.as1D(), this.units);
      // TODO(bileschi) remove type fix once oneHot is consistent.
      // https://github.com/tensorflow/tfjs/issues/435
      return output.asType('float32');
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(OneHot);

export interface ZeroMeanConfig extends LayerConfig {
  optimizer?: ZeroMeanOptimizer;
}

export class ZeroMeanOptimizer extends Serializable {
  static className = 'ZeroMeanOptimizer';
  public count: Variable;
  public meanEstimate: Variable;

  constructor() {
    super();
    this.count = variable(scalar(0));
    this.meanEstimate = null;  // Set on first update to get the right shape.
  }

  public update(x: Tensor, layerMean: LayerVariable) {
    if (this.meanEstimate == null) {
      this.meanEstimate =
          variable(x.mean(0, true), false, 'optimizerMean', 'float32');
    }
    tidy(() => {
      // count += num_samples.
      this.count.assign(add(this.count, x.shape[0]));
      // delta = x - mean.
      // mean = mean + (delta / count).
      this.meanEstimate.assign(
          add(this.meanEstimate,
              div(sub(x.mean(0, true), this.meanEstimate), this.count)));
      layerMean.write(this.meanEstimate);
    });
  }

  getConfig(): ConfigDict {
    return {};
  }

  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    return new cls();
  }
}


export class ZeroMean extends PreprocessingLayer {
  static className = 'ZeroMean';
  // Estimate of sample mean.
  private meanEstimate: LayerVariable = null;
  private meanInitializer: Initializer;
  protected optimizer: ZeroMeanOptimizer;

  constructor(config: ZeroMeanConfig) {
    super(config);
    this.meanInitializer = getInitializer('zeros');
    // Like Model, optimizer may be undefined here if it was not provided via
    // config.
    this.optimizer = config.optimizer;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = type_utils.getExactlyOneShape(inputShape);
    const meanShape = inputShape.slice();
    meanShape[0] = 1;  // Reduce over batch dimension.
    if (this.meanEstimate == null) {
      // Initial estimate of the mean is zero
      this.meanEstimate = this.addWeight(
          'mean', meanShape, 'float32', this.meanInitializer, null, true, null);
    }
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    let inputTensor: Tensor;
    if (Array.isArray(inputs)) {
      if (inputs.length !== 1) {
        throw new ValueError(
            `ZeroMean layer expected one tensor input; got ${inputs.length}`);
      }
      inputTensor = cast(inputs[0], 'float32');
    } else {
      inputTensor = cast(inputs, 'float32');
    }
    return tidy(() => {
      return inputTensor.sub(this.meanEstimate.read());
    });
  }

  // TODO(bileschi): This should probably return a `History` or some other
  // way of keeping track of what happens.
  public fitUnsupervised(x: Tensor): void {
    if (this.optimizer) {
      if (!(this.built)) {
        this.build(x.shape);
      }
      this.optimizer.update(x, this.meanEstimate);
    } else {
      throw new ValueError(
          '.fit() called on `ZeroMean` layer with no optimizer.' +
          '  ZeroMean must be configured with an optimizer to be fittable');
    }
  }
}
serialization.SerializationMap.register(ZeroMean);

export interface UnitVarianceConfig extends LayerConfig {
  optimizer?: UnitVarianceOptimizer;
}

// Algorithm from
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
export class UnitVarianceOptimizer extends Serializable {
  static className = 'UnitVarianceOptimizer';
  public count: Variable;
  public meanEstimate: Variable;
  public m2: Variable;

  constructor() {
    super();
    this.count = variable(scalar(0));
    // Set these upon the first update to get the right shape.
    this.meanEstimate = null;
    // Intermediate storage for calculating variance.
    this.m2 = null;
  }

  public update(x: Tensor, layerVariance: LayerVariable) {
    if (this.meanEstimate == null) {
      this.meanEstimate =
          variable(x.mean(0, true), false, 'unitVarMean', 'float32');
      this.m2 =
          variable(zerosLike(this.meanEstimate), false, 'unitVarM2', 'float32');
    }
    tidy(() => {
      // count += num_samples.
      this.count.assign(add(this.count, x.shape[0]));
      // delta = x - mean.
      const delta = sub(x, this.meanEstimate);
      // mean = mean + (delta / count).
      this.meanEstimate.assign(
          add(this.meanEstimate, div(delta, this.count).mean(0, true)));
      // delta2 = x - mean.
      const delta2 = sub(x, this.meanEstimate);
      this.m2.assign(this.m2.add(delta.mul(delta2)).sum(0, true));
      // variance = m2 / count
      const variance = this.m2.div(this.count);
      const varianceWithNoZero =
          variance.add(cast(variance.equal(scalar(0.0)), variance.dtype));
      layerVariance.write(varianceWithNoZero);
    });
  }

  getConfig(): ConfigDict {
    return {};
  }

  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    return new cls();
  }
}

export class UnitVariance extends PreprocessingLayer {
  static className = 'UnitVariance';
  // Estimate of sample variance.
  private varianceEstimate: LayerVariable = null;
  private varianceInitializer: Initializer;
  protected optimizer: UnitVarianceOptimizer;

  constructor(config: UnitVarianceConfig) {
    super(config);
    this.varianceInitializer = getInitializer('ones');
    // Like Model, optimizer may be undefined here if it was not provided via
    // config.
    this.optimizer = config.optimizer;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = type_utils.getExactlyOneShape(inputShape);
    const varianceShape = inputShape.slice();
    varianceShape[0] = 1;  // Reduce over batch dimension.
    if (this.varianceEstimate == null) {
      // Initial estimate of the variance is one.
      this.varianceEstimate = this.addWeight(
          'variance', varianceShape, 'float32', this.varianceInitializer, null,
          true, null);
    }
    this.built = true;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    let inputTensor: Tensor;
    if (Array.isArray(inputs)) {
      if (inputs.length !== 1) {
        throw new ValueError(
            `UnitVariance layer expected one tensor input; got ${
                inputs.length}`);
      }
      inputTensor = cast(inputs[0], 'float32');
    } else {
      inputTensor = cast(inputs, 'float32');
    }
    return tidy(() => {
      return inputTensor.div(this.varianceEstimate.read().sqrt());
    });
  }

  // TODO(bileschi): This should probably return a `History` or some other
  // way of keeping track of what happens.
  public fitUnsupervised(x: Tensor): void {
    if (this.optimizer) {
      if (!(this.built)) {
        this.build(x.shape);
      }
      this.optimizer.update(x, this.varianceEstimate);
    } else {
      throw new ValueError(
          '.fit() called on `UnitVariance` layer with no optimizer.' +
          '  UnitVariance must be configured with an optimizer to be fittable');
    }
  }
}
serialization.SerializationMap.register(ZeroMean);


// `VocabLayerOptimizer` optimizes a `VocabularyLayer`.  It is implemented
// external to the layer itself, mirroring how, e.g., `Dense` layers are
// optimized via back-propagation via optimzers external to the layer itself.
//
// This class uses a very simple counting implementation, keeping track of every
// unique string token it has seen, and how many times it has seen it.
export class VocabLayerOptimizer extends Serializable {
  static className = 'VocabLayerOptimizer';
  public wordCount: Map<string, number>;

  constructor() {
    super();
    this.wordCount = new Map<string, number>();
  }

  getConfig(): ConfigDict {
    return {};
  }

  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    return new cls();
  }


  // Modifies this optimizer's counts of each unique string.
  public updateCounts(words: StringTensor) {
    for (const word of words.stringValues) {
      // If string is a key in wordCount, update it.
      if (this.wordCount.has(word)) {
        this.wordCount.set(word, this.wordCount.get(word) + 1);
      } else {
        this.wordCount.set(word, 1);
      }
    }
  }

  // Sort by greater count first, alphabetically second.
  protected _compareCountWords(a: [number, string], b: [number, string]):
      number {
    if (a[0] === b[0]) {
      // If the counts are the same, a should come first if it's alphabetically
      // first.
      return +(a[1] > b[1]);
    } else {
      // Otherwise a is should come first if its count is larger.
      return +(a[0] < b[0]);
    }
  }

  // Modifies provided vocab to replace low-count words with higher-count words.
  public updateVocab(vocab: Map<string, number>, knownVocabSize: number) {
    // TODO(bileschi): There is room for optimization here by, e.g., only adding
    // and removing values from the map and never moving them.
    vocab.clear();
    // 1. Convert this.wordCount into (count, word) pairs.
    const countWordPairs: Array<[number, string]> = [];
    this.wordCount.forEach((valUnused, key) => {
      countWordPairs.push([this.wordCount.get(key), key]);
    });
    // 2. sort countWordPairs by descending count.
    countWordPairs.sort(this._compareCountWords);
    // 3. Insert the top knownVocabSize words into vocab
    let numInserted = 0;
    for (const countAndWord of countWordPairs) {
      vocab.set(countAndWord[1], numInserted);
      numInserted++;
      if (numInserted >= knownVocabSize) {
        break;
      }
    }
  }
}
serialization.SerializationMap.register(VocabLayerOptimizer);

export interface VocabLayerConfig extends LayerConfig {
  hashVocabSize?: number;
  knownVocabSize: number;
  vocabInitializer?: InitializerIdentifier|Initializer;
  optimizer?: VocabLayerOptimizer;
}

// TODO(bileschi): Replace with the hash op used in c++ / py tensorflow here:
// core/lib/hash/hash.h
export function vocabHash64(s: string) {
  let hash = 0xDECAFCAFFE, i, chr;
  if (s.length === 0) return hash;
  for (i = 0; i < s.length; i++) {
    chr = s.charCodeAt(i);
    hash = ((hash << 5) - hash) + chr;
    hash |= 0;  // Convert to 32bit integer
  }
  return Math.abs(hash);
}


// A `VocabLayer` is a fittable map from strings to integers in the range
// [0, buckets), where buckets is hashVocabSize + knownVocabSize.
export class VocabLayer extends PreprocessingLayer {
  static className = 'VocabularyLayer';
  readonly hashVocabSize: number;
  readonly knownVocabSize: number;
  private vocabInitializer: Initializer;
  protected optimizer: VocabLayerOptimizer;

  readonly DEFAULT_VOCAB_INITIALIZER: InitializerIdentifier = 'rainbowVocab';


  // Map of words in the known vocabulary.  Key is words in the known
  // vocabulary.  Value is the intenger associated with that word.
  private knownVocab: Map<string, number>;

  constructor(config: VocabLayerConfig) {
    super(config);
    this.dtype = 'string';
    this.knownVocabSize = config.knownVocabSize;
    this.hashVocabSize = config.hashVocabSize | 0;
    this.vocabInitializer = getInitializer(
        config.vocabInitializer || this.DEFAULT_VOCAB_INITIALIZER);
    // Like Model, optimizer may be undefined here if it was not provided via
    // config.
    this.optimizer = config.optimizer;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = type_utils.getExactlyOneShape(inputShape);
    if (this.knownVocab == null && this.knownVocabSize &&
        this.knownVocabSize > 0) {
      const vocabTensor = this.vocabInitializer.apply(
                              [this.knownVocabSize], 'string') as StringTensor;
      this.knownVocab = new Map<string, number>();
      for (let i = 0; i < vocabTensor.size; i++) {
        this.knownVocab.set(vocabTensor.get(i), i);
      }
    }
    this.built = true;
  }

  // TODO(bileschi): This should probably return a `History` or some other way
  // of keeping track of what happens
  public fitUnsupervised(x: StringTensor): void {
    if (this.optimizer) {
      if (!(this.built)) {
        this.build(x.shape);
      }
      this.optimizer.updateCounts(x);
      this.optimizer.updateVocab(this.knownVocab, this.knownVocabSize);
    } else {
      throw new ValueError(
          '.fit() called on VocabLayer with no optimizer.' +
          '  VocabLayer must be configured with an optimizer to be fittable');
    }
  }

  strToIntFn(key: string): number {
    // Index each word into the known Vocab.

    // If hashVocabSize is greater than one, for each word that was *not*
    // found in the known vocabulary, hash the word into hashVocabSize
    // buckets and return that.
    if (this.knownVocab.has(key)) {
      return this.knownVocab.get(key);
    } else {
      if (this.hashVocabSize <= 0) {
        throw new ValueError('Key not in vocab.  Configure hashVocabSize > 0.');
      }
      if (this.hashVocabSize === 1) {
        // Out-of-vocabulary buckets begin after known vocabulary buckets.
        return this.knownVocabSize;
      }
      // hashVocabSize > 1;
      // Out-of-vocabulary buckets begin after known vocabulary buckets.
      return this.hashBucketFn(key, this.hashVocabSize) + this.knownVocabSize;
    }
  }

  // TODO(bileschi): Clone hash functions from tensorflow string ops.
  // .../tensorflow/python/ops/lookup_ops.py#L841
  hashBucketFn(s: string, numBuckets: number) {
    return vocabHash64(s) % numBuckets;
  }

  call(inputs: StringTensor|StringTensor[], kwargs: Kwargs): Tensor|Tensor[] {
    let stringTensor: StringTensor;
    if (Array.isArray(inputs)) {
      if (inputs.length !== 1) {
        throw new ValueError(
            `Vocab layer expected one tensor input; got ${inputs.length}`);
      }
      stringTensor = inputs[0];
    } else {
      stringTensor = inputs as StringTensor;
    }
    return tidy(() => {
      const intValues: number[] = [];
      stringTensor.stringValues.forEach(s => {
        intValues.push(this.strToIntFn(s));
      });
      return tensor(intValues, stringTensor.shape, 'int32');
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      hashVocabSize: this.hashVocabSize,
      knownVocabSize: this.knownVocabSize,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  // TODO(bileschi):  I added this for testing.  Do we want something like this?
  public setVocab(newVocab: Map<string, number>) {
    this.knownVocab = newVocab;
  }

  // TODO(bileschi):  I added this for testing.  Do we want something like this?
  public getVocab(): Map<string, number> {
    return this.knownVocab;
  }
}
serialization.SerializationMap.register(VocabLayer);
