import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
    Tensor is the basic data structure used in tensorflow. It represents a multidimensional array

    rank: Each tensor is described by a unit of dimensionality called rank. 
    It identifies the number of dimensions of the tensor. For this reason, a rank is known as 
    order or n-dimensions of a tensor (for example, a rank 2 tensor is a matrix and a rank 1 tensor is a vector).

    shape: The shape of a tensor is the number of rows and columns it has.

    type: It is the data type assigned to the tensor's elements.
'''

def sample_cycle():
    # We create variables with this code
    counter = tf.Variable(0)
    # Defines a constant
    one = tf.constant(1)
    # Defines a proccess to be done
    new_value = tf.add(one, counter)
    # Defines an upgrade proccess to be run
    upgrade = tf.assign(counter, new_value)
    # initializes all variables when run
    __init_ops__ = tf.initialize_all_variables()
    with tf.Session() as session:
        # Initializes all variables
        session.run(__init_ops__)
        # adds 1 to counter initially
        session.run(upgrade)
        # prints counter
        print(session.run(counter))
        # simple cycle
        while session.run(counter) != 3:
            print(session.run(counter))
            # upgrades counter each iteration
            session.run(upgrade)


# A place holder is a variable that won't receive it's data until a later point
def sample_placeholder():
    # define a place holder to have a float
    a = tf.placeholder(tf.float32)

    dictionary = {a :[ [5, 4, 4], [5, 6, 7], [6, 8, 9] ]}
    # using the place holder before it being defined
    b = 3*a
    with tf.Session() as sess:
        # we run the operation but feeding the info to the session
        # the info must have all placeholders defined
        result = sess.run(b, feed_dict=dictionary)
        print(result)


# sample linear regression
def sample_linear_regression():
    # geerationg random data for sample testing
    x_data = np.random.rand(100).astype(np.float32)
    y_data = 5*x_data +2
    y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)
    zip(x_data, y_data)
    # random guesses for a and b but they will be optimized
    a = tf.Variable(0.2) # Since they are variables tensorflow can change them
    b = tf.Variable(0.7)
    y = a*x_data + b
    # Median square method, we will make y's as close to y_data as possible
    # Using the gradient descent algorithm
    loss = tf.reduce_mean(tf.square(y - y_data)) # calcultes the mean of a tensor and # the resulting tensor may have less dimesnions
    # th class that is incharge of optimizing the parameter a and b of the function y = ax+b
    # 0.5 is the number that will multiply the gradient vector in order to have a faster
    # or more accurate solution
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    # Minimizing the average square error in order to make a b as close as possible to the actual
    # a and b
    train = optimizer.minimize(loss)
    # initiate variables
    init_vars = tf.initialize_all_variables()
    with tf.Session() as session:
        # we must then initialize all variables to enable their use
        session.run(init_vars)
        # just to format the different values of a and b
        train_data = []
        '''
        we will take the gradient descent a total amount of 100 times
        where the step is 5; feel free to increment the range
        this will result in it converging to a set of a and b
        note that the to values a and b are really close to the values of your initial equation stored in y_data
        '''
        for step in range(100):
            '''
            this will run the train optimizer which will minimize the a and b 
            using gradient descent
            afterwards we store the resulting tensor excluding the optimizer
            '''
            eval = session.run([train, a, b])[1:]
            if step%5 == 0:
                print(step, eval)
                train_data.append(eval)

        cr, cg, cb =(1.0, 1.0, 1.0) # colors
        for f in train_data:
            '''
            increments the intensity of the color of the line
            making it so that when it converges many lines will be drawn on top of it
            in the end the resulting line will be really dark because the values of a and
            b converging.
            
            At first the color will be dull because they start at 1 and increment and decrement
            as the train data reaches the end
            '''
            cb += 1.0/len(train_data)
            cg -= 1.0/len(train_data)
            if cb > 1.0: cb = 1.0
            if cg < 0.0: cg = 0
            [a, b] = f # this is the same as setting a = t[0] and b to t[1]
            f_y = np.vectorize(lambda x: a*x + b)(x_data)
            line = plt.plot(x_data, f_y)
            plt.setp(line, color =(cr, cg, cb))
        plt.plot(x_data, y_data, 'ro') # this draws the randomly aquired points
        green_line = mpatches.Patch(color='red', label='Data Points') # sets up the legend
        plt.legend(handles=[green_line]) # makes matplotlib show the legend
        plt.show() # shows the resulting graph


def sample_quadratic_regression():
    EPSILON = 10**(-6)
    # generate random number samples
    x_data = np.random.rand(100)
    # quadratic equations can be written as ax^2 + bx + c
    y_data = 5*x_data**2 + 3
    # Since y_data wasn't produced by numpy we need to vectorize it and move it a little bit so it is more scattered
    y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.7))(y_data)
    zip(x_data, y_data)
    print(y_data)
    # Initial guesses to be optimized
    a = tf.Variable(4.0)
    b = tf.Variable(7.0)
    c = tf.Variable(9.0)
    # Create the tensor that represents the model
    y = a*x_data**2.0 + b*x_data + c
    # we define our loss function
    loss = tf.reduce_mean(tf.square(y - y_data))
    # we define our optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.3)
    # we then define the process
    train = optimizer.minimize(loss)
    # we init vars
    init_vars = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_vars)
        train_data = []
        step = 0
        befores = {"a":session.run(a), "b":session.run(b), "c":session.run(c)}
        # keeps evaluating the gradient til it converges to a digit
        while True:
            befores["a"] = session.run(a)
            befores["b"] = session.run(b)
            befores["c"] = session.run(c)
            eval = session.run([train, a, b, c])[1:]
            if step % 5 == 0:
                print(step, eval)
                train_data.append(eval)
            if abs(befores["a"] - session.run(a)) < EPSILON \
                    and abs(befores["b"] - session.run(b)) < EPSILON \
                    and abs(befores["c"] - session.run(c)) < EPSILON:
                print("total steps:", step)
                break
            step += 1
        cr, cg, cb = (1.0, 1.0, 1.0)  # colors
        for f in train_data:
            cb += 1.0 / len(train_data)
            cg -= 1.0 / len(train_data)
            if cb > 1.0: cb = 1.0
            if cg < 0.0: cg = 0
            [a, b, c] = f  # this is the same as setting a = t[0] and b to t[1]
            f_y = np.vectorize(lambda x: a * x**2 + b*x + c)(x_data)
            line = plt.plot(x_data, f_y)
            plt.setp(line, color=(cr, cg, cb))

        plt.plot(x_data, y_data, 'ro')  # this draws the randomly aquired points
        green_line = mpatches.Patch(color='red', label='Data Points')  # sets up the legend
        plt.legend(handles=[green_line])  # makes matplotlib show the legend
        plt.show()  # shows the resulting graph


'''
    Logistic regression is a classification model which uses an activation function. In this 
    case the sigmond function 
    first a weight matrix is defined giving a certain importance to certain fields,
    then we multiply our data set by that weight resulting in some features being more important than others
    aftr wards a bias is added 
    after this bias is added we use the sigmond function 
'''

sample_linear_regression()