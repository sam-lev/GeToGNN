from __future__ import print_function


import tensorflow as tf
import numpy as np

class MLP:
    def __init__(self, learning_rate, epochs, batch_size,features=None, labels=None,
                 train_features = None, train_labels = None, test_features = None, test_labels = None,
                 dim_hidden_1=None, dim_hidden_2 = None, dim_hidden_3 = None, dim_input = None, n_classes = 2,
                 feature_map = False):

        # Parameters
        self.learning_rate = learning_rate #= 0.001
        self.training_epochs = epochs #= 15
        self.batch_size = batch_size #= 100
        self.display_step = epochs//2

        if feature_map is not None and feature_map:
            train_set = feature_map['train']
            self.train_features = np.array(train_set[1])
            self.train_pos_labels   = np.array(train_set[0])
            self.train_neg_labels = 1 - self.train_pos_labels
            self.train_labels = np.array([[n , p] for n, p in zip(self.train_neg_labels,
                                                                  self.train_pos_labels)])


            test_set = feature_map['test']
            self.test_features = np.array(test_set[1])
            self.test_pos_labels = np.array(test_set[0])
            self.test_neg_labels = 1 - self.test_pos_labels
            self.test_labels = np.array([[n, p] for n, p in zip(self.test_neg_labels,
                                                                 self.test_pos_labels)])


        else:
            self.train_features = np.array(train_features)
            self.train_pos_labels = train_labels
            #if len(train_labels[0]) == 1:
            self.train_neg_labels = 1 - np.array(self.train_pos_labels)
            self.train_labels = np.array(train_labels) #np.array([[n, p] for n, p in zip(self.train_neg_labels,
            #                                                         self.train_pos_labels)])
            self.test_features = test_features
            self.test_pos_labels = test_labels
            #if len(test_labels[0]) == 1:
            self.test_neg_labels = 1 - np.array(self.test_pos_labels)
            self.test_labels = np.array(test_labels)#np.array([[n, p] for n, p in zip(self.test_neg_labels,
            #                                                        self.test_pos_labels)])

        self.features = features #np.array(features)
        self.labels = np.array(labels)

        self.num_examples = self.train_labels.shape[0]
        self.total_batch = self.num_examples // batch_size

        dim_input = int(self.train_features[0].shape[0])
        # Network Parameters
        self.dim_hidden_1 = dim_hidden_1 #= 256 # 1st layer number of neurons
        self.dim_hidden_= dim_hidden_2 #= 256 # 2nd layer number of neurons
        self.dim_input = dim_input #= 784 # MNIST data input (img shape: 28*28)
        self.n_classes = n_classes #= 10 # MNIST total classes (0-9 digits)

        # tf Graph input
        self.X = tf.placeholder("float", [None, dim_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        #dim_test_in = int(self.features[0].shape[0])
        #self.Z = tf.placeholder("float", [None, dim_test_in])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([dim_input, dim_hidden_1])),
            'h2': tf.Variable(tf.random_normal([dim_hidden_1, dim_hidden_2])),
            'h3': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_3])),
            'h4': tf.Variable(tf.random_normal([dim_hidden_3, dim_hidden_2])),
            'h5': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_1])),
            'out': tf.Variable(tf.random_normal([dim_hidden_1, n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([dim_hidden_1])),
            'b2': tf.Variable(tf.random_normal([dim_hidden_2])),
            'b3': tf.Variable(tf.random_normal([dim_hidden_3])),
            'b4': tf.Variable(tf.random_normal([dim_hidden_2])),
            'b5': tf.Variable(tf.random_normal([dim_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        self.build()




    # Create model
    def multilayer_perceptron(self, x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # l3
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3'])
        layer_4 = tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4'])
        layer_5 = tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_5, self.weights['out']) + self.biases['out']
        return out_layer

    def next_batch(self, num, data, labels):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def model(self):

        # Construct model
        self.logits = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()

    def build(self):
        self.model()

    def train(self):

        with tf.Session() as sess:
            sess.run(self.init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                # Loop over all batches
                for i in range(self.total_batch):
                    batch_x, batch_y = self.next_batch(self.batch_size, self.train_features, self.train_labels)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.train_op, self.loss_op], feed_dict={self.X: batch_x,
                                                                    self.Y: batch_y})
                    # Compute average loss
                    avg_cost += c / self.total_batch
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            ac = accuracy.eval({self.X: self.test_features, self.Y: self.test_labels})
            print("Accuracy:", ac)
            pred_test =pred.eval({self.X: self.test_features, self.Y: self.test_labels})
            pred_train = pred.eval({self.X: self.train_features, self.Y: self.train_labels})
            preds = pred.eval({self.X: self.features, self.Y: self.labels})
            #pred_train = []
            #self.train_labels = []
            self.accuracy=ac
            #preds = pred_test + pred_train
            #labels = self.test_labels + self.train_labels
            return preds, self.labels, ac #[pred_test, pred_train] , [self.test_labels , self.train_labels] , ac