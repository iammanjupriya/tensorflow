
1. **Import TensorFlow**:
   ```python
   import tensorflow as tf
   ```
   This cell imports the TensorFlow library.

2. **Print TensorFlow Version**:
   ```python
   print(tf.__version__)
   ```
   This cell prints the version of TensorFlow being used (e.g., 2.15.0).

3. **Create a Constant Tensor**:
   ```python
   A = tf.constant([[4, 3], [6, 1]])  # constant
   print(A)
   ```
   This cell creates a constant tensor `A` with the specified values and prints it.

4. **Create a Variable Tensor**:
   ```python
   V = tf.Variable([[1, 2, 3], [4, 5, 6]])  # Variable
   print(V)
   B = tf.constant([[5, 8], [9, 5]])  # constant
   print(B)
   ```
   This cell creates a variable tensor `V` and a constant tensor `B`, and then prints them.

5. **Concatenate Tensors**:
   ```python
   AB = tf.concat([A, B], 0)
   print(AB)
   ```
   This cell concatenates tensors `A` and `B` along the first axis and prints the result.

6. **Create a Zero Tensor**:
   ```python
   tensor = tf.zeros([2, 3])
   print(tensor)
   ```
   This cell creates a tensor of zeros with the specified shape and prints it.

7. **Building a Sequential Model**:
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Flatten(input_shape=(28, 28)),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)
   ])
   ```
   This cell defines a Sequential model with three layers: a Flatten layer, a Dense layer with ReLU activation, and another Dense layer.

8. **Compile the Model**:
   ```python
   model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
   ```
   This cell compiles the model with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as a metric.

9. **Load and Preprocess the MNIST Dataset**:
   ```python
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```
   This cell loads the MNIST dataset and normalizes the pixel values to the range [0, 1].

10. **Train the Model**:
    ```python
    model.fit(x_train, y_train, epochs=5)
    ```
    This cell trains the model on the training data for 5 epochs.

11. **Evaluate the Model**:
    ```python
    model.evaluate(x_test, y_test, verbose=2)
    ```
    This cell evaluates the model on the test data and prints the loss and accuracy.

12. **Create a Probability Model**:
    ```python
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    ```
    This cell creates a new model that adds a Softmax layer to the previously defined model to convert logits to probabilities.

13. **Predict with the Probability Model**:
    ```python
    probability_model(x_test[:5])
    ```
    This cell makes predictions on the first 5 test samples and prints the predicted probabilities.

The notebook demonstrates basic operations with TensorFlow, such as creating tensors, building and training a neural network, and making predictions with the trained model.
