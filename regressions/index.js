require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: [
    'horsepower',
    'displacement',
    'weight',
    'cylinders',
    'modelyear'
  ],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);
console.log('r2 is', r2);

// console.log('b:', regression.weights.get(0, 0));
// console.log('m:', regression.weights.get(1, 0));
