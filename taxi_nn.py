import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


env = gym.make('Taxi-v3')

# Define the neural network
model = tf.keras.Sequential([
    Dense(64, activation='relu', input_shape=(env.observation_space.n,)),
    Dense(64, activation='relu'),
    Dense(env.action_space.n, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        # Convert the state to a one-hot encoded vector
        state_vector = np.eye(env.observation_space.n)[state]
        state_vector[0][int(state)] = 1


        # Choose an action based on the model's predictions
        action_probs = model.predict(state_vector)[0]
        action = np.random.choice(env.action_space.n, p=action_probs)

        # Take the action and observe the next state and reward
        step_result = env.step(action)
        next_state, reward, done, info, _ = step_result

        # Convert the next state to a one-hot encoded vector
        next_state_vector = np.eye(env.observation_space.n)[next_state]
        next_state_vector[0][int(next_state)] = 1



        # Update the model using the observed transition
        target = reward
        if not done:
            target += 0.99 * np.amax(model.predict(next_state_vector)[0])
        target_vector = model.predict(state_vector)
        target_vector[0][action] = target
        model.fit(state_vector, target_vector, epochs=1, verbose=0)

        state = next_state

# Test
total_reward = 0
episodes = 100
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        # Convert the state to a one-hot encoded vector
        state_vector = np.eye(env.observation_space.n)[state]
        state_vector[0][int(state)] = 1


        # Choose an action based on the model's predictions
        action_probs = model.predict(state_vector)[0]
        action = np.argmax(action_probs)

        # Take the action and observe the next state and reward
        step_result = env.step(action)
        next_state, reward, done, info, _ = step_result

        state = next_state
        total_reward += reward

print("Average reward per episode:", total_reward / episodes)
