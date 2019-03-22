const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseRecords = [];

    // use default object or override it with options if provided
    // this.options = {learningRate: 0.1, iterations: 1000, ...options};  // ES6
    this.options = Object.assign(
      { learningRate: 0.0001, iterations: 1000 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent() {
    // solve: (features * ((features * weights) - labels)) / n

    // (features * weights). return [many, 1] shape tensor
    const currentGuesses = this.features.matMul(this.weights);

    // return [many, 1] shape tensor
    const differences = currentGuesses.sub(this.labels); // - labels

    // return [2, 1] shape tensor
    const slopes = this.features
      // change the shape of features tensor from [many, 2] to [2, many]
      .transpose()
      // features * differences. return [2, 1] shape tensor
      .matMul(differences)
      // divide by n (n = rows of the original features tensor)
      .div(this.features.shape[0]);

    // modify the weights. [2, 1] shape
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  // no tensors version
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
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  // check accuracy using R2 = 1 - (SSres / SStot)
  // SS - sum of squares (residuals and total)
  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    // return [many, 1] shape tensor
    const predictions = testFeatures.matMul(this.weights);

    // SStot = sum of all (Label - Mean) squared
    const tot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .get();

    // SSres = sum of all (Label - Predicted) squared
    const res = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .get();

    return 1 - res / tot;
  }

  // prepare features tensor. standardize and add a column
  processFeatures(features) {
    features = tf.tensor(features);
    // for test set use the existing mean and variance
    // for training set get the mean and variance of features and standardize
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // append features to a column of ones. return [many, 2] shape tensor
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  // store all the MSE records. newest first
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseRecords.unshift(mse);
  }

  // use only if there is more than two MSE records stored
  updateLearningRate() {
    if (this.mseRecords.length < 2) {
      return;
    }
    // decrease learning rate if MSE increased. otherwise add 5% to learning rate
    if (this.mseRecords[0] > this.mseRecords[1]) {
      this.options.learningRate *= 0.5;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;
