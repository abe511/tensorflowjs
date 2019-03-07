const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

//  extracts only the necessary data using the indexes of column names
function extractColumns(data, columnNames) {
  const headers = _.first(data);

  const indexes = _.map(columnNames, (column) => {
    return headers.indexOf(column);
  });

  const extracted = _.map(data, (row) => {
    return _.pullAt(row, indexes);
  });

  return extracted;
}

function loadCSV(
  filename,
  {
    dataColumns = [],
    labelColumns = [],
    shuffle = true,
    splitTest = false, // boolean or integer ( if true, the data is divided equally)
    shuffleString = 'defaultString', // change the string to reshuffle
    converters = {} // custom converter function for the column values
  }
) {
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });
  let labels = [];

  // get array of arrays
  data = data.split('\n').map((row) => {
    return row.split(',');
  });

  // removing trailing commas
  data = data.map((row) => {
    return _.dropRightWhile(row, (value) => {
      return value === '';
    });
  });

  // get the headers row
  const headers = _.first(data);

  // parse data using a custom converter or turn into numbers
  data = data.map((row, index) => {
    if (index === 0) {
      return row;
    }
    return row.map((el, index) => {
      if (converters[headers[index]]) {
        // check for a method named like a column
        const converted = converters[headers[index]](el); // call it with an element as arg
        return _.isNaN(converted) ? el : converted;
      }
      const result = parseFloat(el);
      return _.isNaN(result) ? el : result;
    });
  });

  // extracting features and labels
  data = extractColumns(data, dataColumns);
  labels = extractColumns(data, labelColumns);

  data.shift();
  labels.shift();

  // the same shuffle string is used to get matching results
  if (shuffle) {
    data = shuffleSeed.shuffle(data, shuffleString);
    labels = shuffleSeed.shuffle(labels, shuffleString);
  }

  // split the features and labels in half or using the splitTest number
  if (splitTest) {
    const testSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length * 0.5);

    return {
      features: data.slice(testSize),
      labels: labels.slice(testSize),
      testFeatures: data.slice(0, testSize),
      testLabels: labels.slice(0, testSize)
    };
  } else {
    return { features: data, labels }; // return data without splitting
  }
}

// test call

// const { features, labels, testFeatures, testLabels } = loadCSV('./data.csv', {
//   dataColumns: ['id', 'height'],
//   labelColumns: ['passed'],
//   splitTest: 1,
//   converters: {
//     passed: (value) => {
//       return value === 'TRUE' ? 1 : 0;
//     }
//   }
// });

// console.log('features', features);
// console.log('labels', labels);
// console.log('testFeatures', testFeatures);
// console.log('testLabels', testLabels);
