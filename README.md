# TensorFlowLearn
создадим модель TensorFlow.js для распознавания рукописных цифр с помощью сверточной нейронной сети

1. Столкнулся с CORS при тестировании в браузере. Т.к. использую VS Code, то воспользовался расширением Live Server. "Hello TensorFlow" в консоль получил.
2. Картинки загрузились и отобразились в окне браузера. При этом в консоле получено сообщение:
```
Canvas2D: Multiple readback operations using getImageData are faster with the willReadFrequently attribute set to true. See: https://html.spec.whatwg.org/multipage/canvas.html#concept-canvas-will-read-frequently
img.onload @ data.js:68
load (async)
(anonymous) @ data.js:49
load @ data.js:47
run @ script.js:35
```