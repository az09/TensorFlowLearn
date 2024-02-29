console.log('Намчнем помолямсь TensorFlow');
import {AntData} from './data_my.js';

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(1);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 64x48 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([64, 48, 1]);
    });
    
    const canvas = document.getElementById('canvas');
    //canvas.width = 64;
    //canvas.height = 48;
    //canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    //surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function createDataset(dataPath) {
  const imageDataset = await tf.data
    .generator(async function* () {
      const classes = fs.readdirSync(dataPath);

      for (const className of classes) {
        const classPath = path.join(dataPath, className);

        for (const imageFileName of fs.readdirSync(classPath)) {
          const imagePath = path.join(classPath, imageFileName);
          const imageData = fs.readFileSync(imagePath);
          const imageTensor = tf.node.decodeImage(imageData);
          const normalizedImage = tf.image.resizeBilinear(imageTensor, [targetSize, targetSize]).div(255.0);

          // Определите ваш метка класса здесь
          const label = className === 'класс_вашего_изображения' ? 1 : 0;
//Замените 'класс_вашего_изображения' на имя класса, к которому принадлежит ваше изображение.
//Также уточните свою логику меток классов в соответствии с вашими потребностями.

          yield {
            xs: normalizedImage,
            ys: tf.oneHot(label, classes.length),
          };
        }
      }
    })
    .shuffle(1000) // Перемешивание данных
    .batch(32); // Укажите размер батча

  return imageDataset;
}

// Теперь вы можете использовать датасет для обучения вашей нейронной сети

async function run() {
// Использование функции для создания датасета
  const datasetPath = './data_my/';
  const dataset = await createDataset(datasetPath);
  //const data = new AntData();
  //await data.load();
  await showExamples(dataset);
  return;
  const model = getModel();
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
    
  await train(model, data);
  await showAccuracy(model, data);
  await showConfusion(model, data);
}

document.addEventListener('DOMContentLoaded', run); //пробуем запускать автоматически

function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 640;
  const IMAGE_HEIGHT = 480;
  const IMAGE_CHANNELS = 3;  
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 14 output units, one for each
  // output class (i.e. 01, 02, 03, 04, 05 и т.д.).
  const NUM_OUTPUT_CLASSES = 14;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  const BATCH_SIZE = 128;
  const TRAIN_DATA_SIZE = 4061;
  const TEST_DATA_SIZE = 421;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 640, 480, 3]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 640, 480, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 17,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

const classNames = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 640;
  const IMAGE_HEIGHT = 480;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}
