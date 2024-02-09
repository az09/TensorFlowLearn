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
3. Самая мякотка: нужно правильно задать описание входных данных и те фукции активации, по которым будет производиться обучение каждого слоя.
Обратил тут внимание на используемую переменную `tf`. Пока не понял где она объявляется (tf.min.js?).

Ссылки для изучения:
* https://setosa.io/ev/image-kernels/ (свёрточные алгоритмы для изображений)
* https://cs231n.github.io/convolutional-networks/ (большая статья на английском с картинками)