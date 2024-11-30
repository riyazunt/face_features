const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

let model = undefined;
let children = [];

// Check if webcam access is supported
function getUserMediaSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}

// Load the TensorFlow Lite model
async function loadModel() {
  try {
    const modelUrl = 'detect.tflite'; // Replace with actual path to your TFLite model
    model = await tflite.loadTFLiteModel(modelUrl);
    console.log('Model loaded successfully!');
    demosSection.classList.remove('invisible');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

// Enable webcam and start the prediction loop
function enableCam(event) {
  event.target.classList.add('removed');
  const constraints = { video: true };
  
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
  });
}

// Preprocess the webcam frame for TensorFlow Lite model input
function preprocessImage(frame) {
  let imgTensor = tf.browser.fromPixels(frame);
  imgTensor = tf.image.resizeBilinear(imgTensor, [224, 224]); // Resize to model input size (320x320)
  const processedImg = tf.sub(tf.div(tf.expandDims(imgTensor), 127.5), 1); // Normalize image
  return processedImg;
}

// function preprocessImage(frame) {
//   let imgTensor = tf.browser.fromPixels(frame);

//   // Get image dimensions
//   const [height, width, channels] = imgTensor.shape;

//   // Determine new size with aspect ratio preserved
//   let newHeight = 320;
//   let newWidth = Math.round((width / height) * newHeight);

//   // If the width is larger than 320, resize to fit width
//   if (newWidth > 320) {
//     newWidth = 320;
//     newHeight = Math.round((height / width) * newWidth);
//   }

//   // Resize image to new dimensions
//   imgTensor = tf.image.resizeBilinear(imgTensor, [newHeight, newWidth]);

//   // Pad the image to 320x320 if needed
//   const padHeight = 320 - newHeight;
//   const padWidth = 320 - newWidth;

//   // Manually pad image tensor
//   imgTensor = tf.pad(
//     imgTensor,
//     [[Math.floor(padHeight / 2), Math.ceil(padHeight / 2)],  // Pad top and bottom
//      [Math.floor(padWidth / 2), Math.ceil(padWidth / 2)],   // Pad left and right
//      [0, 0]]);  // No padding for the color channels

//   // Normalize image
//   const processedImg = tf.sub(tf.div(tf.expandDims(imgTensor), 127.5), 1);

//   return processedImg;
// }



const classMapping = {
  0: "ForeheadRect",
  1: "Forehead",
  2: "Right_Cheek",
  3: "Left_Cheek"
};

// Make predictions and draw bounding boxes
async function predictWebcam() {
  if (!model) {
    return;
  }

  // Get the current video width and height (this is dynamic)
  const videoWidth = video.clientWidth; // Get width of video element on the screen
  const videoHeight = video.clientHeight; // Get height of video element on the screen

  // Preprocess the image for the model
  const inputTensor = preprocessImage(video);
  
  try {
    const outputTensor = await model.predict(inputTensor);

    // Clear previous bounding boxes
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    children = [];

    // Get the model's output tensors
    const boxes = outputTensor['StatefulPartitionedCall:3'].dataSync();
    const classes = outputTensor['StatefulPartitionedCall:2'].dataSync();
    const scores = outputTensor['StatefulPartitionedCall:1'].dataSync();

    // Loop through predictions and draw boxes
    for (let i = 0; i < boxes.length / 4; i++) {
      const ymin = boxes[i * 4] * videoHeight;
      const xmin = boxes[i * 4 + 1] * videoWidth;
      const ymax = boxes[i * 4 + 2] * videoHeight;
      const xmax = boxes[i * 4 + 3] * videoWidth;

      const classId = classes[i];
      const score = scores[i];

      // Only draw boxes with high confidence (greater than 0.7)
      if (score > 0.7) {
        const bbox = document.createElement('div');
        bbox.classList.add('highlighter');
        bbox.style.position = 'absolute';
        bbox.style.left = `${xmin}px`;
        bbox.style.top = `${ymin}px`;
        bbox.style.width = `${xmax - xmin}px`;
        bbox.style.height = `${ymax - ymin}px`;

        // Create and position the label
        const label = document.createElement('p');
        const className = classMapping[classId];
        label.innerText = `${className}, ${score.toFixed(2)}`;
        label.style.position = 'absolute';
        label.style.left = `${xmin}px`;
        label.style.top = `${ymin - 28}px`;

        // Style the label
        label.style.fontSize = '8px';

        // Add bounding box and label to liveView
        liveView.appendChild(bbox);
        liveView.appendChild(label);
        children.push(bbox);
        children.push(label);
      }
    }

    // Request next frame for continuous prediction
    window.requestAnimationFrame(predictWebcam);
  } catch (error) {
    console.error('Error during inference:', error);
  }
}


// Load the model and start webcam stream
loadModel();