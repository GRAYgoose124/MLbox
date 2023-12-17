import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the tokenization and dataset
vocab_size = 10000  # Replace with actual vocabulary size
bos_token = 1
eos_token = 2

epochs = 10
batch_size = 32
latent_dim = 100
embedding_dim = 128
learning_rate = 0.001
temperature = 1.0  # Adjust temperature for sampling diversity


def tokenize(text, vocab, max_vocab_size, oov_token="<UNK>"):
    tokens = text.split()
    ids = []

    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            if len(vocab) < max_vocab_size:
                # Add new token to vocab if there's space
                new_id = len(vocab)
                vocab[token] = new_id
                ids.append(new_id)
            else:
                # Use OOV token if vocab is full
                ids.append(vocab[oov_token])

    # Add BOS and EOS tokens
    ids = [vocab["<BOS>"]] + ids + [vocab["<EOS>"]]
    return ids


# Example usage
max_vocab_size = 10000  # Set a limit for the vocabulary size
vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

dataset = tf.data.TextLineDataset(tf.io.gfile.glob("text_data/*.txt"))
dataset = dataset.shuffle(10000).batch(batch_size)

# Define the embedding logic
embedding_matrix = tf.Variable(
    initial_value=tf.random.normal((vocab_size, embedding_dim)),
    name="embeddings",
    trainable=True,
)


def embed(ids):
    return tf.nn.embedding_lookup(embedding_matrix, ids)


# Define the next token sampling function
def sample_next_token(posterior_mean, posterior_var, temperature):
    # Sample from the normal distribution with mean and variance
    sampled_latent_vector = posterior_mean + tf.random.normal(
        shape=posterior_mean.shape
    ) * tf.sqrt(posterior_var)
    # Predict the next token using the generator
    predicted_logits = generator(sampled_latent_vector, embedding_dim)
    # Apply temperature scaling
    scaled_logits = predicted_logits / temperature
    # Sample the next token using softmax distribution
    next_token = tf.random.categorical(scaled_logits, 1)
    return next_token


# Define the generator
def generator(prior_mean, noise):
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# Define the discriminator with comparison logic
def discriminator(embedding):
    inputs = tf.keras.layers.Input(shape=(embedding_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# Define the Bayesian integrator
def bayesian_integrator(embeddings, prior_mean, prior_var):
    # Embed the tokens
    embedded_tokens = embed(embeddings)
    # Pass through the generator
    generator_output = generator(prior_mean, embedding_dim)
    # Pass through the discriminator
    discriminator_output = discriminator(embedded_tokens)
    # Update the posterior mean and variance based on the discriminator's output
    posterior_mean = prior_mean + tf.exp(-discriminator_output) * noise
    posterior_var = prior_var / (1 + tf.exp(-discriminator_output) ** 2)
    return posterior_mean, posterior_var


generator = generator(latent_dim, embedding_dim)
discriminator = discriminator(embedding_dim)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(epochs):
    for batch in dataset:
        # Generate noise and prior mean
        noise = tf.random.normal((batch_size, latent_dim))
        prior_mean = tf.zeros((batch_size, latent_dim))
        prior_var = tf.ones((batch_size, latent_dim))

        # Update the posterior
        posterior_mean, posterior_var = bayesian_integrator(
            batch, prior_mean, prior_var
        )

        # Sample next token from posterior distribution
        next_token = sample_next_token(posterior_mean, posterior_var, temperature)

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Fetch real embedding for first token
            real_embedding = embed(batch[:, 1])
            # Calculate discriminator outputs
            discriminator_real_output = discriminator(real_embedding)
            discriminator_fake_output = discriminator(embed(next_token))
            # Compute discriminator loss
            discriminator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(discriminator_real_output),
                    logits=discriminator_real_output,
                )
            ) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(discriminator_fake_output),
                    logits=discriminator_fake_output,
                )
            )

        # Train the generator
        with tf.GradientTape() as tape:
            generator_fake_output = discriminator(generator(posterior_mean, noise))
            generator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(generator_fake_output),
                    logits=generator_fake_output,
                )
            )

        # Apply gradients to update model parameters
        generator_optimizer.apply_gradients(
            zip(gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients, discriminator.trainable_variables)
        )

    # Print training progress
    print(f"Epoch: {epoch + 1}")

# online user interaction / training loop:
while True:
    # Get user input
    user_input = input("Enter a prompt: ")
    # Tokenize the input
    tokenized_input = tokenize(user_input, vocab, max_vocab_size)
    if not tokenized_input:
        continue

    # Convert tokenized input to a TensorFlow tensor
    tokenized_input_tensor = tf.convert_to_tensor(tokenized_input, dtype=tf.int32)

    # Generate noise and prior mean
    noise = tf.random.normal((1, latent_dim))
    prior_mean = tf.zeros((1, latent_dim))
    prior_var = tf.ones((1, latent_dim))
    # Update the posterior
    posterior_mean, posterior_var = bayesian_integrator(
        tokenized_input, prior_mean, prior_var
    )
    # Sample next token from posterior distribution
    next_token = sample_next_token(posterior_mean, posterior_var, temperature)
    # Print the predicted tokens
    print(" ".join([vocab[token] for token in next_token.numpy().squeeze()]))
