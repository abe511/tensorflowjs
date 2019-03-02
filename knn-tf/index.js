require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV(
  'kc_house_data.csv',
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
  }
);

function knn(features, labels, point, k) {
  return (
    // finding the difference between feature and point
    features
      .sub(point)
      .pow(2)
      .sum(1)
      .pow(0.5) // getting the distance between two points
      .expandDims(1)
      .concat(labels, 1) // joining features and labels tensor arrays
      .unstack() // converting to an array of tensor
      // sorting the array of tensors by distance (lowest first)
      .sort((a, b) => {
        return a.get(0) > b.get(0) ? 1 : -1;
      })
      // extracting closest k tensors, summing the labels and finding the average
      .slice(0, k)
      .reduce((sum, tensor) => {
        return sum + tensor.get(1);
      }, 0) / k
  );
}

features = tf.tensor(features);
labels = tf.tensor(labels);
testFeatures = tf.tensor(testFeatures[0]);

const result = knn(features, labels, testFeatures, 10);

console.log('guess', result, testLabels[0][0]);
