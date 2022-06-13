# Mastering TensorFlow 1.x
This is the code repository for [Mastering TensorFlow 1.x](https://www.packtpub.com/big-data-and-business-intelligence/mastering-tensorflow-1x?utm_source=github&utm_medium=repository&utm_campaign=9781788292061), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
TensorFlow is the most popular numerical computation library built from the ground up for distributed, cloud, and mobile environments. TensorFlow represents the data as tensors and the computation as graphs.

This book is a comprehensive guide that lets you explore the advanced features of TensorFlow 1.x. Gain insight into TensorFlow Core, Keras, TF Estimators, TFLearn, TF Slim, Pretty Tensor, and Sonnet. Leverage the power of TensorFlow and Keras to build deep learning models, using concepts such as transfer learning, generative adversarial networks, and deep reinforcement learning. Throughout the book, you will obtain hands-on experience with varied datasets, such as MNIST, CIFAR-10, PTB, text8, and COCO-Images.

You will learn the advanced features of TensorFlow1.x, such as distributed TensorFlow with TF Clusters, deploy production models with TensorFlow Serving, and build and deploy TensorFlow models for mobile and embedded devices on Android and iOS platforms. You will see how to call TensorFlow and Keras API within the R statistical software, and learn the required techniques for debugging when the TensorFlow API-based code does not work as expected.

The book helps you obtain in-depth knowledge of TensorFlow, making you the go-to person for solving artificial intelligence problems. By the end of this guide, you will have mastered the offerings of TensorFlow and Keras, and gained the skills you need to build smarter, faster, and efficient machine learning and deep learning systems.

## Instructions and Navigation

The code relies on datasetslib, that is added as a submodule and available in datasetslib folder. To checkout the code with the submoudles use the --recurse-submoudles switch as follows:

```
git clone --recurse-submodules git@github.com:armando-fandango/Mastering-TensorFlow.git
```

All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
from datasetslib.ptb import PTBSimple
ptb = PTBSimple()
ptb.load_data()
print('Train :',ptb.part['train'][0:5])
print('Test: ',ptb.part['test'][0:5])
print('Valid: ',ptb.part['valid'][0:5])
print('Vocabulary Length = ',ptb.vocab_len)
```

1. We assume that you are familiar with coding in Python and the basics of
TensorFlow and Keras.
2. If you haven't done already, then install Jupyter Notebooks, TensorFlow, and
Keras.
3. Download the code bundle for this book that contains the Python, R, and
notebook code files.
4. Practice with the code as you read along the text and try exploring by modifying
the provided sample code.
5. To practice the Android chapter, you will need Android Studio and an Andrioid
device.
6. To practice the iOS chapter, you will need an Apple computer with Xcode and an
Apple device.
7. To practice the TensorFlow chapter, you will need Docker and Kubernetes
installed. Instruction for installing Kubernetes and Docker on Ubuntu are
provided in the book.

## Related Products
* [Deep Learning with TensorFlow - Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781788831109)

* [Reinforcement Learning with Tensorflow](https://www.packtpub.com/big-data-and-business-intelligence/reinforcement-learning-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781788835725)

* [TensorFlow 1.x Deep Learning Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-1x-deep-learning-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788293594)
