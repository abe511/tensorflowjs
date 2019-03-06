require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

// load data, shuffle, split and extract features and labels for test and training
let { features, labels, testFeatures, testLabels } = loadCSV(
  'kc_house_data.csv',
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
  }
);

function knn(features, labels, point, k) {
  const { mean, variance } = tf.moments(features);

  // standardizing the test sample
  const scaledPoint = point.sub(mean).div(variance.pow(0.5));

  return (
    // find the difference between feature and point
    features
      .sub(mean)
      .div(variance.pow(0.5)) // standardizing the features
      .sub(scaledPoint)
      .pow(2)
      .sum(1)
      .pow(0.5) // get the distance between two points
      .expandDims(1)
      .concat(labels, 1) // join features and labels tensor arrays
      .unstack() // convert to an array of tensor
      // sort the array of tensors by distance (lowest first)
      .sort((a, b) => {
        return a.get(0) > b.get(0) ? 1 : -1;
      })
      // extract closest k tensors, sum the labels and find the average
      .slice(0, k)
      .reduce((sum, tensor) => {
        return sum + tensor.get(1);
      }, 0) / k
  );
}

// convert to tensors
features = tf.tensor(features);
labels = tf.tensor(labels);

// run error percentage analysis for all test samples
testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0];
  console.log('error', err * 100);
});
