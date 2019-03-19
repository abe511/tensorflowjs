const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);

    // appending the features tensor to a column of ones
    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1);

    // use default object or override it with options if provided
    // this.options = {learningRate: 0.1, iterations: 1000, ...options};

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.weights = tf.zeros([2, 1]);
  }

  gradientDescent() {
    // solve: features * ((features * weights) - labels)

    // (features * weights). return [many, 1] shape tensor
    const currentGuesses = this.features.matMul(this.weights);

    // return [many, 1] shape tensor
    const differences = currentGuesses.sub(this.labels); // - labels

    // return [2, 1] shape tensor
    const slopes = this.features
      // change the shape of features tensor from [many, 2] to [2, many]
      .transpose()
      // features * differences (matrix multiplication)
      .matMul(differences)
      // divide by n (n = rows of the original features tensor)
      .div(this.features.shape[0]);

    // get the new weights. [2, 1] shape
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  // no tensors
  // gradientDescent() {
  //   // solve (m*x+b) part of the equation
  //   const currentGuessesForMPG = this.features.map((row) => {
  //     return this.m * row[0] + this.b;
  //   });

  //   // calculate the slope for b
  //   // sum the results of the (guess - actual) part
  //   // multiply by 2 and divide by records quantity
  //   const bSlope =
  //     (_.sum(
  //       currentGuessesForMPG.map((guess, index) => {
  //         return guess - this.labels[index][0];
  //       })
  //     ) *
  //       2) /
  //     this.labels.length;

  //   // calculate the slope for m
  //   // sum the results of the (guess - actual) part and multiply by -x (feature)
  //   // multiply by 2 and divide by quantity of records
  //   const mSlope =
  //     (_.sum(
  //       currentGuessesForMPG.map((guess, index) => {
  //         return -1 * this.features[index][0] * (this.labels[index][0] - guess);
  //       })
  //     ) *
  //       2) /
  //     this.labels.length;

  //   this.b = this.b - bSlope * this.options.learningRate;
  //   this.m = this.m - mSlope * this.options.learningRate;
  // }

  // run for all iterations
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
