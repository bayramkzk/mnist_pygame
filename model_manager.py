import tensorflow as tf

class MnistModelManager:
	def __init__(self):
		self.model = None
	
	def fit_model(self):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		
		x_train = x_train / 255
		x_test = x_test / 255
		
		self.model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(128, activation=tf.nn.relu),
			tf.keras.layers.Dense(10, activation=tf.nn.softmax)
		])

		self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy')

		self.model.fit(x_train, y_train, epochs=5, verbose=0)
	
	def save_model(self, json_path, h5_path):
		assert self.model is not None, "a model was not read or created"
		
		model_json = self.model.to_json()
		with open(json_path, "w") as json_file:
		    json_file.write(model_json)

		self.model.save_weights(h5_path)

	def read_model(self, json_path, h5_path):
		with open(json_path, 'r') as json_file:
			loaded_model_json = json_file.read()
		
		self.model = tf.keras.models.model_from_json(loaded_model_json)
		
		self.model.load_weights(h5_path)
		
		self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
