const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

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
    shuffleString = 'defaultString',
    converters = {}
  }
) {
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });
  let labels = [];

  //   get array of arrays
  data = data.split('\n').map((row) => {
    return row.split(',');
  });

  //   removing trailing commas
  data = data.map((row) => {
    return _.dropRightWhile(row, (value) => {
      return value === '';
    });
  });

  //   get the headers row
  const headers = _.first(data);

  data = data.map((row, index) => {
    if (index === 0) {
      return row;
    }
    return row.map((el, index) => {
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](el);
        return _.isNaN(converted) ? el : converted;
      }
      const result = parseFloat(el);
      return _.isNaN(result) ? el : result;
    });
  });

  labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);

  labels.shift();
  data.shift();

  if (shuffle) {
    labels = shuffleSeed.shuffle(labels, shuffleString);
    data = shuffleSeed.shuffle(data, shuffleString);
  }

  console.log(labels);
  console.log(data);
}

loadCSV('./data.csv', {
  dataColumns: ['id', 'height'],
  labelColumns: ['passed'],
  converters: {
    passed: (value) => {
      return value === 'TRUE' ? 1 : 0;
    }
  }
});
