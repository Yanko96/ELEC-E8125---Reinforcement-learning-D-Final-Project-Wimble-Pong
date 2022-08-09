# Two Player Wimblepong

This project is part of the final project of Reinforcement Learning course at Aalto University, implementing A2C with pretrained CNN-VAE encoder on a two player version of the pong-v0 OpenAI Gym environment.

# Pretrained CNN-VAE
A CNN-VAE is pre-trained on collected observations of the wimble pong environment in order to accelerate the converge of the agent training. Some of the results on the test set are shown below.

<img src="imgs/reconstructed_0.png" width="400">
<img src="imgs/reconstructed_1.png" width="400">
<img src="imgs/reconstructed_3.png" width="400">
<img src="imgs/reconstructed_4.png" width="400">

# A2C Agent
The encoder of the Agent is loaded from the checkpoint of the encoder of the pre-trained CNN-VAE. Then the agent is trained by A2C algorithm with entropy loss to encourage exploration.
