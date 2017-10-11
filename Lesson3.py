import tensorflow as tf
import numpy as np
import random

def main():

    label_index = {
        'D1': 0,
        'D2': 1,
        'D3': 2,
        'D4': 3
    }

    data = []
    labels = []

    with open("Soybean.txt", "r") as input_file:
        for line in input_file:
            if(len(line.strip()) == 0):
                continue

            full_data_line = line.strip().split(",")

            data_line = full_data_line[0:-1]
            label_line = full_data_line[-1]

            data_line = list(map(float, data_line))

            if label_line in label_index:
                label_line = label_index[label_line]
            else:
                print("Bad data", line)
                continue

            data.append(data_line)
            labels.append(label_line)



    print("data", len(data))
    print("labels", len(labels))

    dataset = list(zip(data, labels))
    random.shuffle(dataset)
    test_length = int(len(dataset) * 0.67)

    print("test_length", test_length)
    train_dataset = dataset[:test_length]
    test_dataset = dataset[test_length:]

    x_size = 35

    # Symbols
    inputs = tf.placeholder("float", shape=[None, x_size])
    labels = tf.placeholder("int32", shape=[None])

    weights1 = tf.get_variable("weight1", shape=[35,100], initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("bias1", shape=[100], initializer=tf.constant_initializer(value=0.0))

    layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)

    weights2 = tf.get_variable("weight2", shape=[100, 100], initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable("bias2", shape=[100], initializer=tf.constant_initializer(value=0.0))

    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

    weights3 = tf.get_variable("weight3", shape=[100, 4], initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable("bias3", shape=[4], initializer=tf.constant_initializer(value=0.0))

    outputs = tf.matmul(layer2, weights3) + bias3

    # backprop
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, 4), logits=outputs))
    train = tf.train.AdamOptimizer().minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(outputs), axis=1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        previous_test_loss_output, previous_test_prediction_output = 999999, 0.0
        done = False
        while not done:
            batch = random.sample(train_dataset, 25)
            inputs_batch, labels_batch = zip(*batch)
            loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: inputs_batch, labels: labels_batch})

            # print("Prediction output", prediction_output)
            # print("Labels batch", labels_batch)

            # accuracy = np.mean(labels_batch == prediction_output)

            # print("train", "loss", loss_output, "accuracy", accuracy)
            batch = random.sample(test_dataset, 15)
            test_inputs_batch, test_labels_batch = zip(*batch)    
            test_loss_output, test_prediction_output = sess.run([loss, predictions], feed_dict={inputs: test_inputs_batch, labels: test_labels_batch})
            
            if  previous_test_loss_output < test_loss_output :
                done = True
            
            previous_test_loss_output, previous_test_prediction_output = test_loss_output, test_prediction_output

            accuracy = np.mean(labels_batch == prediction_output)

            print("train", "loss", loss_output, "accuracy", accuracy)


            



if __name__ == "__main__":
    main()
