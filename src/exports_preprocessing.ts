/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import {Rank, ShapeMap} from '@tensorflow/tfjs-core';

// tslint:disable-next-line:max-line-length
import {UnitVariance, UnitVarianceOptimizer, VocabLayer, VocabLayerOptimizer, ZeroMean, ZeroMeanOptimizer} from './preprocess-layers/preprocess_core';
// tslint:disable-next-line:max-line-length
import {PreprocessingExports, StringArray, StringTensor, StringTensor1D, StringTensor2D, StringTensor3D, StringTensor4D, StringTensor5D, StringTensor6D} from './preprocess-layers/string_tensor';

export {
  UnitVariance,
  UnitVarianceOptimizer,
  VocabLayer,
  VocabLayerOptimizer,
  ZeroMean,
  ZeroMeanOptimizer
};

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor<R extends Rank>(
    values: string|StringArray, shape?: ShapeMap[R]): StringTensor<R> {
  return PreprocessingExports.stringTensor(values, shape);
}

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor1d(values: string[]): StringTensor1D {
  return PreprocessingExports.stringTensor1d(values);
}

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor2d(
    values: string[]|string[][], shape?: [number, number]): StringTensor2D {
  return PreprocessingExports.stringTensor2d(values);
}

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor3d(
    values: string[]|string[][][],
    shape?: [number, number, number]): StringTensor3D {
  return PreprocessingExports.stringTensor3d(values);
}

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor4d(
    values: string[]|string[][][][],
    shape?: [number, number, number, number]): StringTensor4D {
  return PreprocessingExports.stringTensor4d(values);
}

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor5d(
    values: string[]|string[][][][][],
    shape?: [number, number, number, number, number]): StringTensor5D {
  return PreprocessingExports.stringTensor5d(values);
}

/**
 * @doc {
 *   heading: 'Tensors',
 *   subheading: 'Creation'
 * }
 */
export function stringTensor6d(
    values: string[]|string[][][][][][],
    shape?: [number, number, number, number, number, number]): StringTensor6D {
  return PreprocessingExports.stringTensor6d(values);
}
