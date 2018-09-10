import os
import skimage
from skimage import data, transform
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import random

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith('.ppm')]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = os.getcwd()
train_data_directory = os.path.join(ROOT_PATH, 'Training')
test_data_directory = os.path.join(ROOT_PATH, 'Testing')
images_pickle_directory = os.path.join(ROOT_PATH, 'TSTrainingImages.pickle')
labels_pickle_directory = os.path.join(ROOT_PATH, 'TSTrainingLabels.pickle')

try:
    with open(images_pickle_directory, 'rb') as read:
        images = pickle.load(read)
    with open(labels_pickle_directory, 'rb') as read:
        labels = pickle.load(read)
except FileNotFoundError:
    print('File Not Found')
    images, labels = load_data(train_data_directory)
    images = np.array(images)
    labels = np.array(labels)
    with open('TSTrainingImages.pickle', 'wb') as write:
        pickle.dump(images, write)
    with open('TSTrainingLabels.pickle', 'wb') as write:
        pickle.dump(labels, write)
                                       
'''
print(images.ndim)      # 1
print(images.size)      # 4575
print(labels.ndim)      # 1
print(labels.size)      # 4575
print(len(set(labels))) # 62

# Plot the histogram of the data samples present
plt.hist(labels, 62)
plt.show()

# Plot a few random images
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.imshow(images[random.randint(1, labels.size)])
plt.show()

# Plot an image for each type of symbol
unique_labels = set(labels)
i = 1
for label in unique_labels:
    image = images[list(labels).index(label)]
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.title('Label {} ({})'.format(label, list(labels).count(label)))
    i += 1
    plt.imshow(image)
plt.show()
'''

images28 = np.array([transform.resize(image, (28, 28)) for image in images])    # Resize the training images
'''
print(images28.ndim)    # 4
print(images28.size)    # 10760400
print(images28.shape)   # (4575, 28, 28, 3)
'''

images28 = rgb2gray(images28)   # Gray-Scale Conversion
'''
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.imshow(images28[random.randint(1, labels.size)], cmap='gray')

plt.show()
'''

''' Creating the network model '''

# Initialise the placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

''' Running the Neural Network '''
tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(201):
    print('EPOCH', i)
    _, cur_loss, accuracy_val = sess.run([train_op, loss, accuracy], feed_dict={x: images28, y:labels})
    if i%10 == 0:
        print('Loss: ', cur_loss)
        print('Accuracy: ', accuracy_val)
    print('DONE WITH EPOCH')

''' Evaluating the neural network '''

# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)
print('Accuracy: {:.3f}'.format(accuracy))

# Visualising few test samples
sample_indexes = random.sample(range(len(test_images28)), 10)
sample_images = [test_images28[i] for i in sample_indexes]
sample_labels = [test_labels[i] for i in sample_indexes]

predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

print(sample_labels)
print(predicted)

fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, i+1)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, 'Truth: {}\nPrediction: {}'.format(truth, prediction), fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap='gray')
plt.show()

sess.close()
