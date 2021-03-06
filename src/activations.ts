/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// Layer activation functions
import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {getScalar} from './backend/state';
import {deserializeKerasObject} from './utils/generic_utils';

/**
 * Base class for Activations.
 *
 * Special note: due to cross-language compatibility reasons, the
 * static readonly className field in this family of classes must be set to
 * the initialLowerCamelCase name of the activation.
 */
export abstract class Activation extends serialization.Serializable {
  abstract apply(tensor: Tensor, axis?: number): Tensor;
  getConfig(): serialization.ConfigDict {
    return {};
  }
}

/** @docinline */
export type ActivationIdentifier = 'elu'|'hardSigmoid'|'linear'|'relu'|'relu6'|
    'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|string;

/**
 * Exponential linear unit (ELU).
 * Reference: https://arxiv.org/abs/1511.07289
 */
export class Elu extends Activation {
  static readonly className = 'elu';
  /**
   * Calculate the activation function.
   *
   * @param x: Input.
   * @param alpha: Scaling factor the negative section.
   * @return Output of the ELU activation.
   */
  apply(x: Tensor, alpha = 1): Tensor {
    return K.elu(x, alpha);
  }
}
serialization.SerializationMap.register(Elu);

/**
 * Scaled Exponential Linear Unit. (Klambauer et al., 2017).
 * Reference: Self-Normalizing Neural Networks, https://arxiv.org/abs/1706.02515
 * Notes:
 *   - To be used together with the initialization "lecunNormal".
 *   - To be used together with the dropout variant "AlphaDropout".
 */
export class Selu extends Activation {
  static readonly className = 'selu';
  apply(x: Tensor): Tensor {
    return tfc.selu(x);
  }
}
serialization.SerializationMap.register(Selu);

/**
 *  Rectified linear unit
 */
export class Relu extends Activation {
  static readonly className = 'relu';
  apply(x: Tensor): Tensor {
    return tfc.relu(x);
  }
}
serialization.SerializationMap.register(Relu);

/**
 * Rectified linear unit activation maxing out at 6.0.
 */
export class Relu6 extends Activation {
  static readonly className = 'relu6';
  apply(x: Tensor): Tensor {
    return tidy(() => tfc.minimum(getScalar(6.0), tfc.relu(x)));
  }
}
serialization.SerializationMap.register(Relu6);

//* Linear activation (no-op) */
export class Linear extends Activation {
  static readonly className = 'linear';
  apply(x: Tensor): Tensor {
    return x;
  }
}
serialization.SerializationMap.register(Linear);

/**
 * Sigmoid activation function.
 */
export class Sigmoid extends Activation {
  static readonly className = 'sigmoid';
  apply(x: Tensor): Tensor {
    return tfc.sigmoid(x);
  }
}
serialization.SerializationMap.register(Sigmoid);

/**
 * Segment-wise linear approximation of sigmoid.
 */
export class HardSigmoid extends Activation {
  static readonly className = 'hardSigmoid';
  apply(x: Tensor): Tensor {
    return K.hardSigmoid(x);
  }
}
serialization.SerializationMap.register(HardSigmoid);

/**
 * Softplus activation function.
 */
export class Softplus extends Activation {
  static readonly className = 'softplus';
  apply(x: Tensor): Tensor {
    return tfc.softplus(x);
  }
}
serialization.SerializationMap.register(Softplus);

/**
 * Softsign activation function.
 */
export class Softsign extends Activation {
  static readonly className = 'softsign';
  apply(x: Tensor): Tensor {
    return K.softsign(x);
  }
}
serialization.SerializationMap.register(Softsign);

/**
 * Hyperbolic tangent function.
 */
export class Tanh extends Activation {
  static readonly className = 'tanh';
  apply(x: Tensor): Tensor {
    return tfc.tanh(x);
  }
}
serialization.SerializationMap.register(Tanh);

/**
 * Softmax activation function
 */
export class Softmax extends Activation {
  static readonly className = 'softmax';
  /**
   * Calculate the activation function.
   *
   * @param x Tensor.
   * @param axis Integer, axis along which the softmax normalization is applied.
   * Invalid if < 2, as softmax across 1 (the batch dimension) is assumed to be
   * an error.
   *
   * @returns a Tensor of the same shape as x
   *
   * @throws ValueError: In case `dim(x) < 2`.
   */
  apply(x: Tensor, axis: number = (-1)): Tensor {
    return tfc.softmax(x, axis);
  }
}
serialization.SerializationMap.register(Softmax);

export function serializeActivation(activation: Activation): string {
  return activation.getClassName();
}

export function deserializeActivation(
    config: serialization.ConfigDict,
    customObjects: serialization.ConfigDict = {}): Activation {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'activation');
}

export function getActivation(identifier: ActivationIdentifier|
                              serialization.ConfigDict|Activation): Activation {
  if (identifier == null) {
    const config = {className: 'linear', config: {}};
    return deserializeActivation(config);
  }
  if (typeof identifier === 'string') {
    const config = {className: identifier, config: {}};
    return deserializeActivation(config);
  } else if (identifier instanceof Activation) {
    return identifier;
  } else {
    return deserializeActivation(identifier);
  }
}
