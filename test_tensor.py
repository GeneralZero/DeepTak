import tensorflow as tf

###Start

def train():
	# size x size x (bits representing square) + white capstone + black capstone + is white playing?
	# 5x5x64 + 3
	input_size = 5*5*64+3

	# Hidden layers by board
	hidden_layer_dimmentions = [2048] * 3

	GAMMA = 0.1
	TEST_SIZE_MAX = 10000.0
	TRAIN_BATCH_SIZE = 2000

	##Set weights and biases
	weight_set, bias_set = get_parameters(input_layer=size, hidden_units=hidden_units_dim)

	BATCH_SIZE = min(TRAIN_BATCH_SIZE, curr_train.shape[0])

	train_func = get_function(weight_set, bias_set, update=True, dropout=False)
	test_func = get_function(weight_set, bias_set, update=False, dropout=False)


def get_parameters(input_layer=None, hidden_units=2048, hidden_layers=1, weights=None, biases=None):
	"""
	Returns a set of TensorFlow Variable weights and a set of TensorFlowxx
	Variable biases. These are used in our training function
	Notes:
		The Hidden Layer in a neural network is an intermediary layer that
		processes the Input Layer into the Output Layer
	Params:
		n_in - Input Layer
		n_hidden_units - Units for the hidden layer
		n_hidden_layers - Number of hidden layers
		weights - Initial array of weights. If None, get_parameters will
				  generate it
		biases - Initial array of biases. If None, get_parameters will generate
				 it
	Returns:
		A weight set of TensorFlow variables, and a bias set of TensorFlow
		variables
	"""
	if weights is None or biases is None:
		if type(hidden_units) != list:
			hidden_units = [hidden_units] * hidden_layers
		else:
			hidden_layers = len(hidden_units)
		weights = []
		biases = []

		# Define internally
		def weight_values(n_in, n_out):
			return numpy.asarray(RNG.uniform(
				low = -numpy.sqrt(6. / (n_in + n_out)),
				high = numpy.sqrt(6. / (n_in + n_out)),
				size = (n_in, n_out)), dtype=numpy.float32)

		for i in xrange(hidden_layers):
			tmp_input_layer = input_layer
			weight = None
			bias = None
			if i != 0:
				tmp_input_layer = hidden_units[i - 1]
			if i < hidden_layers - 1:
				tmp_output_layers = hidden_units[i]
				weight = weight_values(tmp_input_layer, tmp_output_layers)
				bias = numpy.ones(tmp_output_layers, dtype=numpy.float32) * GAMMA
			else:
				weight = numpy.zeros(tmp_input_layer, dtype=numpy.float32)
				bias = floatX(0.)
			weights.append(weight)
			biases.append(bias)

	weight_set = [tf.Variable(w) for w in weights]
	bias_set = [tf.Variable(b) for b in biases]

	# We have to explicitily tell the session to initialize the TF Variables
	for var in weight_set:
		sess.run(var.initializer)
	for var in bias_set:
		sess.run(var.initializer)

	return weight_set, bias_set



def get_function(weight_set, bias_set, dropout=False, update=False):
	"""
	Generate a learning function
	Params:
	    weight_set - list of TensorFlow tensor values for weights
	    bias_set - list of TensorFlow bias values for biases
	    dropout - If True, will drop dead or insignificant neurons
	    update - If True, will use Nesterov's momentum optimization updates
	Returns:
	    A function that can be called to train a data set
	"""
    # Set up variable values
    board_il, board_rand_il, board_parent_il, loss_net, reg, loss_a, loss_b, loss_c = get_training_model(weight_set, bias_set, dropout=dropout)
    objective = loss_net + reg
    learning_rate = tf.placeholder(pconst.FLOAT_TYPE, shape=[])
    momentum = floatX(0.9)
    updates = []

    if update:
        assert(len(weight_set) == len(bias_set))
        params = []
        for i in xrange(len(weight_set)):
            params.append(tf.add(weight_set[i], bias_set[i]))
        updates = nesterov_update(objective, params, learning_rate, momentum)
    func = tfunc.function(
            inputs=[board_il, board_rand_il, board_parent_il, learning_rate],
            outputs=[loss_net, reg, loss_a, loss_b, loss_c],
            updates=updates,
            session=sess)
    return func


"""
Returns the training model for our training function
Params:
    weight_set - list of TensorFlow tensor (matrix) of weight values
    bias_set - list of TensorFlow tensor (matrix) of bias values
    dropout - If True, will remove all neurons that are below threshold
    multiplier - An independent variable for our training function
    kappa - An independent variable for our training function
Returns:
    A tuple with the following indices:
        0 - board input layer
        1 - board rand input layer
        2 - board parent input layer
        3 - statistical net loss of probability of making a move
        4 - Regularization of weights and bias
        5 - statistical net loss of probability between current board state, and
            after a random move
        6 - statistical net loss of probability between current board state and
            parent
        7 - inverse of index 6
"""
def get_training_model(weight_set, bias_set, dropout=False, multiplier=10.0,kappa=1.0):
    # Build a dual network, one for the real move, one for a fake random move
    # Train on a negative log likelihood of classifying the right move
    # il = input layer, ol = output layer
    board_il, board_ol = get_model(weight_set, bias_set, dropout)
    board_rand_il, board_rand_ol = get_model(weight_set, bias_set, dropout)
    board_parent_il, board_parent_ol = get_model(weight_set, bias_set, dropout)

    rand_diff = board_ol - board_rand_ol
    loss_a = -tf.reduce_mean(tf.log(tf.nn.sigmoid(rand_diff)))
    parent_diff = kappa * (board_ol + board_parent_ol)
    loss_b = -tf.reduce_mean(tf.log(tf.nn.sigmoid(parent_diff)))
    loss_c = -tf.reduce_mean(tf.log(tf.nn.sigmoid(-parent_diff)))

    # Regularization
    reg = 0
    for x in weight_set + bias_set:
        reg += multiplier * tf.reduce_mean(tf.square(x))
    loss_net = loss_a + loss_b + loss_c
return board_il, board_rand_il, board_parent_il, loss_net, reg, loss_a, loss_b, loss_c


"""
Returns a basic Tensor model for training or testing
Params:
    weight_set - list of TensorFlow tensor (matrix) of weight values
    bias_set - list of TensorFlow tensor (matrix) of bias values
    dropout - If True, will remove all neurons that are below threshold
Returns:
    The input layer and output layer (TensorFlow tensor objects)
"""
def get_model(weight_set, bias_set, dropout=False):
    # Create an input layer to process the weights and biases
    input_layer = tf.placeholder(numpy.float32, shape=[24, 64])

    # Make a list of dropouts if not already a list
    if type(dropout) != list:
        dropout = [dropout] * len(weight_set)

    # Build a matrix of pieces based on the input layer
    pieces = []
    for piece in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
        pieces.append(tf.cast(tf.equal(input_layer, piece), numpy.float32))
        
    # Build the final model (output_layer) from the weights and biases
    binary_layer = tf.concat(1, pieces)
    last_layer = binary_layer
    n = len(weight_set)
    for index in xrange(n - 1):
        intermediary = tf.matmul(last_layer, weight_set[index]) + bias_set[index]
        intermediary = intermediary * tf.cast((intermediary > 0),
                numpy.float32)
        if dropout[index]:
            mask = numpy.random.binomial(1, 0.5, shape=intermediary.get_shape())
            intermediary = tf.mul(intermediary, tf.cast(mask, numpy.float32))
            intermediary = tf.mul(intermediary,
                    tf.fill(intermediary.get_shape().as_list(), 2))

        last_layer = intermediary
    output_layer = tf.matmul(last_layer, tf.reshape(weight_set[-1], [2048,1])) + bias_set[-1]
    return input_layer, output_layer
