"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow as tf
from keras.layers import Dense
import numpy as np
from src.time_gan.utils import extract_time, rnn_cell, random_generator, batch_generator

# needed to get the tf1 code to work
tf.compat.v1.disable_eager_execution()


def TimeGan(ori_data, parameters, X_test, y_test, num_fault_types):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        """Min-Max Normalizer.

        Args:
          - data: raw data

        Returns:
          - norm_data: normalized data
          - min_val: minimum values (for renormalization)
          - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    ## Build a RNN networks

    # Network Parameters
    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    z_dim = dim
    gamma = 1

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(
        tf.float32, [None, max_seq_len, z_dim], name="myinput_z"
    )
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    def embedder(X, T):
        """Embedding network between original feature space to latent space.

        Args:
          - X: input time-series features
          - T: input time information

        Returns:
          - H: embeddings
        """
        with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            )
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                e_cell, X, dtype=tf.float32, sequence_length=T
            )
            H = Dense(hidden_dim, activation="sigmoid")(e_outputs)
        return H

    def recovery(H, T):
        """Recovery network from latent space to original space.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - X_tilde: recovered data
        """
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            )
            r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(
                r_cell, H, dtype=tf.float32, sequence_length=T
            )
            X_tilde = Dense(dim, activation="sigmoid")(r_outputs)
        return X_tilde

    def generator(Z, T):
        """Generator function: Generate time-series data in latent space.

        Args:
          - Z: random variables
          - T: input time information

        Returns:
          - E: generated embedding
        """
        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            )
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                e_cell, Z, dtype=tf.float32, sequence_length=T
            )
            E = Dense(hidden_dim, activation="sigmoid")(e_outputs)
        return E

    def supervisor(H, T):
        """Generate next sequence using the previous sequence.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - S: generated sequence based on the latent representations generated by the generator
        """
        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)]
            )
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(
                e_cell, H, dtype=tf.float32, sequence_length=T
            )
            S = Dense(hidden_dim, activation="sigmoid")(e_outputs)
        return S

    def discriminator(H, T):
        """Discriminate the original and synthetic time-series data.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - Y_hat: classification results between original and synthetic time-series
        """
        with tf.compat.v1.variable_scope(
            "discriminator", reuse=tf.compat.v1.AUTO_REUSE
        ):
            d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            )
            d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(
                d_cell, H, dtype=tf.float32, sequence_length=T
            )
            Y_hat = Dense(1)(d_outputs)
        return Y_hat

    # Embedder & Recovery
    H = embedder(X, T)
    X_tilde = recovery(H, T)

    # Generator
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)

    # Synthetic data
    X_hat = recovery(H_hat, T)

    # Discriminator
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)
    Y_fake_e = discriminator(E_hat, T)


    # Variables
    e_vars = [
        v for v in tf.compat.v1.trainable_variables() if v.name.startswith("embedder")
    ]
    r_vars = [
        v for v in tf.compat.v1.trainable_variables() if v.name.startswith("recovery")
    ]
    g_vars = [
        v for v in tf.compat.v1.trainable_variables() if v.name.startswith("generator")
    ]
    s_vars = [
        v for v in tf.compat.v1.trainable_variables() if v.name.startswith("supervisor")
    ]
    d_vars = [
        v
        for v in tf.compat.v1.trainable_variables()
        if v.name.startswith("discriminator")
    ]

    # Discriminator loss
    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(Y_real), Y_real
    )
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.zeros_like(Y_fake), Y_fake
    )
    D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.zeros_like(Y_fake_e), Y_fake_e
    )
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # Generator loss
    # 1. Adversarial loss
    G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(
        tf.ones_like(Y_fake_e), Y_fake_e
    )

    # 2. Supervised loss
    G_loss_S = tf.compat.v1.losses.mean_squared_error(
        H[:, 1:, :], H_hat_supervise[:, :-1, :]
    )

    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(
        tf.abs(
            tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6)
            - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)
        )
    )
    G_loss_V2 = tf.reduce_mean(
        tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0]))
    )

    G_loss_V = G_loss_V1 + G_loss_V2

    # 4. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    # Embedder network loss
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        E_loss0, var_list=e_vars + r_vars
    )
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        E_loss, var_list=e_vars + r_vars
    )
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        G_loss, var_list=g_vars + s_vars
    )
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(
        G_loss_S, var_list=g_vars + s_vars
    )

    ## TimeGAN training
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # 1. Embedding network training
    print("Start Embedding Network Training")

    best_e_loss = np.inf
    counter = 0
    MAX_COUNTER = 50
    EARLY_STOP_THRESHOLD = 0.00001
    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Train embedder
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})

        if best_e_loss > step_e_loss + EARLY_STOP_THRESHOLD:
            counter = 0
            best_e_loss = step_e_loss
        else:
            counter += 1
        # Checkpoint
        if itt % 50 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(iterations)
                + ", e_loss: "
                + str(np.round(np.sqrt(step_e_loss), 4)) 
                + ", counter: " 
                + str(counter) 
                + ", best_loss: "
                + str(np.round(np.sqrt(best_e_loss), 4))
            )
        if counter >= MAX_COUNTER:
            print(f"No progress made for {MAX_COUNTER} steps, stopping early...")
            break

    print("Finish Embedding Network Training")

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")

    best_g_loss_s = np.inf
    counter = 0
    EARLY_STOP_THRESHOLD = 0.00001
    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Train generator
        _, step_g_loss_s = sess.run(
            [GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb}
        )

        if best_g_loss_s > step_g_loss_s + EARLY_STOP_THRESHOLD:
            counter = 0
            best_g_loss_s = step_g_loss_s
        else:
            counter += 1
        # Checkpoint
        if itt % 50 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(iterations)
                + ", s_loss: "
                + str(np.round(np.sqrt(step_g_loss_s), 4)) 
                + ", counter: " 
                + str(counter)  
                + ", counter: " 
                + str(counter) 
                + ", best_loss: "
                + str(np.round(np.sqrt(best_g_loss_s), 4))
            )
        if counter >= MAX_COUNTER:
            print(f"No progress made for {MAX_COUNTER} steps, stopping early...")
            break

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")

    THRESHOLD_LOW = 0.15
    ABOVE_THRESHOLD = 0.1
    EARLY_STOP_THRESHOLD = 0.001
    never_reached_low = True
    best_step_d_loss = np.inf
    best_step_g_loss_u = np.inf
    best_step_g_loss_s = np.inf
    best_step_g_loss_v = np.inf
    best_step_e_loss_t0 = np.inf
    counter = 0
    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run(
                [G_solver, G_loss_U, G_loss_S, G_loss_V],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb},
            )
            # Train embedder
            _, step_e_loss_t0 = sess.run(
                [E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb}
            )

        # Discriminator training
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Check discriminator loss before updating
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        # Train discriminator (only when the discriminator does not work well)

        if check_d_loss > THRESHOLD_LOW and never_reached_low:
            _, step_d_loss = sess.run(
                [D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb}
            )
        else:
            if check_d_loss <= THRESHOLD_LOW:
                never_reached_low = False
            elif check_d_loss > THRESHOLD_LOW + ABOVE_THRESHOLD:
                never_reached_low = True  # Start training discriminator again
                best_step_d_loss = np.inf
                best_step_g_loss_u = np.inf
                best_step_g_loss_s = np.inf
                best_step_g_loss_v = np.inf
                best_step_e_loss_t0 = np.inf
                counter = 0

        no_loss_gain = True
        if best_step_d_loss > step_d_loss + EARLY_STOP_THRESHOLD:
            best_step_d_loss = step_d_loss
            no_loss_gain = False
        if best_step_g_loss_u > step_g_loss_u + EARLY_STOP_THRESHOLD:
            best_step_g_loss_u = step_g_loss_u
            no_loss_gain = False
        if best_step_g_loss_s > step_g_loss_s + EARLY_STOP_THRESHOLD:
            best_step_g_loss_s = step_g_loss_s
            no_loss_gain = False
        if best_step_g_loss_v > step_g_loss_v + EARLY_STOP_THRESHOLD:
            best_step_g_loss_v = step_g_loss_v
            no_loss_gain = False
        if best_step_e_loss_t0 > step_e_loss_t0 + EARLY_STOP_THRESHOLD:
            best_step_e_loss_t0 = step_e_loss_t0
            no_loss_gain = False
        if no_loss_gain:
            counter += 1
        else:
            counter = 0
        # Print multiple checkpoints
        if itt % 10 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(iterations)
                + ", d_loss: "
                + str(np.round(step_d_loss, 4))
                + ", g_loss_u: "
                + str(np.round(step_g_loss_u, 4))
                + ", g_loss_s: "
                + str(np.round(np.sqrt(step_g_loss_s), 4))
                + ", g_loss_v: "
                + str(np.round(step_g_loss_v, 4))
                + ", e_loss_t0: "
                + str(np.round(np.sqrt(step_e_loss_t0), 4))
                + ", counter: " 
                + str(counter)
            )
        if counter >= MAX_COUNTER:
            print(f"No progress made for {MAX_COUNTER} steps, stopping early...")
            break
    print("Finish Joint Training")

    ### Uncomment if you want generatro data
    ## Synthetic data generation
    # Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    # generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

    # generated_data = list()

    # for i in range(no):
    #     temp = generated_data_curr[i, : ori_time[i], :]
    #     generated_data.append(temp)

    # # Renormalization
    # generated_data = generated_data * max_val
    # generated_data = generated_data + min_val

    # print("Finished Getting Generated Data\n")


    ## Testing of the discriminator
    TEST_BATCH_SIZE = 1000
    no_test, seq_len, dim_test = np.asarray(X_test).shape
    X_test, min_val, max_val = MinMaxScaler(X_test)
    num_batches = int(float(no_test) / float(TEST_BATCH_SIZE))
    all_labels = np.array(np.squeeze(y_test), dtype=int)
    y_pred_list = []
    totals_list = [[], [], []]
    for batch_num in range(1, num_batches+1):
        starting_idx = ((batch_num-1) * TEST_BATCH_SIZE)
        next_batch_idx = (batch_num * TEST_BATCH_SIZE) - 1
        last_idx = next_batch_idx if batch_num < (num_batches) else no_test-1
        batch = X_test[starting_idx:(last_idx+1)]
        no_batch, seq_len, dim_batch = np.asarray(batch).shape
        batch_time, max_seq_len = extract_time(batch)
        Z_mb = random_generator(no_batch, dim_batch, batch_time, max_seq_len)
        real_curr, fake_curr, fake_e_curr = sess.run([Y_real, Y_fake, Y_fake_e], feed_dict={Z: Z_mb, X: batch, T: batch_time})

        def populate_output(curr, labels, mean_idx, median_idx, orig_idx):
            output = [[[] for _ in range(3)] for _ in range(num_fault_types)]
            for i in range(curr.shape[0]):
                instance = curr[i]
                instance_1D = np.squeeze(instance)
                fault_type = labels[i]
                output[fault_type][mean_idx].append(np.mean(instance_1D))
                output[fault_type][median_idx].append(np.median(instance_1D))
                output[fault_type][orig_idx].append(i)

            return output

            
        
        MEAN_INDEX = 0
        MEDIAN_INDEX = 1
        ORIGINAL_INDEX = 2
        labels = all_labels[starting_idx:(last_idx+1)]
        real_output = populate_output(real_curr, labels, MEAN_INDEX, MEDIAN_INDEX, ORIGINAL_INDEX)
        fake_output = populate_output(fake_curr, labels, MEAN_INDEX, MEDIAN_INDEX, ORIGINAL_INDEX)
        fake_e_output = populate_output(fake_e_curr, labels, MEAN_INDEX, MEDIAN_INDEX, ORIGINAL_INDEX)

        def get_totals(output, mean_index, median_index):
            totals = []
            for i in range(num_fault_types):
                mean_average = np.mean(output[i][mean_index])
                median_average = np.mean(output[i][median_index])
                totals.append((mean_average, median_average))

            return totals

        real_averages = get_totals(real_output, MEAN_INDEX, MEDIAN_INDEX)
        fake_averages = get_totals(fake_output, MEAN_INDEX, MEDIAN_INDEX)
        fake_e_averages = get_totals(fake_e_output, MEAN_INDEX, MEDIAN_INDEX)

        # Goes through each fault type
        y_pred = np.ones_like(labels)
        for i in range(len(real_output)):
            # Goes through data for each fault type. In this case we use the mean of each window.
            for j in range(len(real_output[i][MEAN_INDEX])):
                score = real_output[i][MEAN_INDEX][j] - fake_output[i][MEAN_INDEX][j] - fake_e_output[i][MEAN_INDEX][j]
                if score > 0:
                    y_pred[real_output[i][ORIGINAL_INDEX][j]] = 0


        y_pred_list.append(y_pred)
        totals_list[0].append(real_averages)
        totals_list[1].append(fake_averages)
        totals_list[2].append(fake_e_averages)

    totals = totals_list
    y_pred = np.concatenate(y_pred_list)

    return y_pred, totals
