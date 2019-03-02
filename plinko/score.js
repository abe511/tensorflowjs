const outputs = [];
const point = 300;

// getting the data
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

// run knn analysis using lodash
function runAnalysis() {
  const testSetSize = 100;
  const k = 15;

  // create a range of indices for all the features
  _.range(0, 3).forEach((feature) => {
    // shorten outputs arrays to just two elements like [feature, label]
    const data = _.map(outputs, (row) => [row[feature], _.last(row)]);
    // extracting test and training set from the data array split by the testSetSize
    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
    const accuracy = _.chain(testSet)
      // filter out the ones that are equal to the test bucket label (last element)
      // with arrays in the training set as input data
      // and drop point from test set as control point
      .filter((testSample) => {
        return (
          knn(trainingSet, _.initial(testSample), k) === _.last(testSample)
        );
      })
      .size()
      // and find the percentage of correct predictions
      .divide(testSetSize)
      .value();
    console.log('for feature ', feature, 'accuracy is', accuracy);
  });
}

// data array and the control point as inputs
function knn(data, point, k) {
  return (
    _.chain(data)
      // calculate the difference between all the factors in the array A and B
      // and return it in a new array with the bucket label as a second argument
      // point has all the data except bucket label
      .map((arr) => {
        return [distance(_.initial(arr), point), _.last(arr)];
      })
      // sort by the distance (closest first)
      .sortBy((arr) => arr[0])
      // separate the first k arrays
      .slice(0, k)
      // return an object with bucket label as a key and occurences as a value
      .countBy((arr) => arr[1])
      // convert back to array [bucket label, occurence]
      .toPairs()
      // sort by occurences (the last occured most often)
      .sortBy((arr) => arr[1])
      // extract the last array
      .last()
      // extract the first element (bucket label)
      .first()
      // convert the string to integer
      .parseInt()
      // return the value
      .value()
  );
}

function distance(pointA, pointB) {
  return (
    _.chain(pointA)
      // interweave two arrays like [[a1, b1], [a2, b2]]
      .zip(pointB)
      // extract point A and point B from pairs and get their difference squared
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      // get the square root of sum of all squares
      .value() ** 0.5
  );
}

// shuffle the data array
// split from 0 to testCount and from testCount to the end
function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);
  return [testSet, trainingSet];
}

// normalize the feature values
function minMax(data, featureCount) {
  // duplicate the data array
  const dataClone = _.cloneDeep(data);
  // loop featureCount times (all except bucket label)
  for (let i = 0; i < featureCount; i++) {
    // create an array for each feature
    const column = dataClone.map((row) => row[i]);
    // extract min and max of each feature
    const min = _.min(column);
    const max = _.max(column);
    // loop through all the rows in the data array
    for (let j = 0; j < dataClone.length; j++) {
      // replace all the values with normalized ones
      // j as a row, i as a feature (column)
      dataClone[j][i] = (dataClone[j][i] - min) / (max - min);
    }
  }
  return dataClone;
}
