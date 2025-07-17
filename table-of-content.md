# 12-Hour Deep Learning Intensive Study Plan

## Overview
This plan covers neural networks, deep learning, and reinforcement learning in 12 intensive hours. Each session includes core concepts, key formulas, and practical understanding.

---

## Hour 1: Introduction to Machine Learning and Neural Networks
**Time: 60 minutes**

### Core Concepts (20 mins)
- **Machine Learning Types**: Supervised, unsupervised, reinforcement learning
- **Neural Network Basics**: Perceptron, multi-layer perceptrons
- **Key Components**: Neurons, weights, biases, layers

### Mathematical Foundations (20 mins)
- **Linear Algebra**: Vectors, matrices, dot products
- **Basic Calculus**: Partial derivatives, chain rule
- **Probability**: Basic probability distributions

### Practical Understanding (20 mins)
- **Perceptron Formula**: y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
- **Simple Neural Network Architecture**: Input → Hidden → Output
- **Learning Process**: Adjust weights based on errors

---

## Hour 2: Forward Propagation and Activation Functions
**Time: 60 minutes**

### Forward Propagation (30 mins)
- **Process**: Input → Weighted sum → Activation → Next layer
- **Matrix Operations**: Z = W·X + B, A = f(Z)
- **Layer-by-layer computation**: From input to output

### Activation Functions (30 mins)
- **Sigmoid**: σ(x) = 1/(1 + e⁻ˣ) - smooth, bounded [0,1]
- **ReLU**: f(x) = max(0, x) - simple, addresses vanishing gradients
- **Tanh**: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) - bounded [-1,1]
- **Softmax**: For multi-class classification
- **When to use each**: ReLU for hidden layers, Sigmoid for binary output

---

## Hour 3: Backpropagation and Gradient Descent
**Time: 60 minutes**

### Backpropagation Theory (30 mins)
- **Chain Rule Application**: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
- **Error Propagation**: From output layer back to input
- **Gradient Computation**: Calculate gradients for all weights and biases

### Gradient Descent (30 mins)
- **Basic Formula**: w = w - α · ∂L/∂w
- **Learning Rate (α)**: Controls step size
- **Variants**: 
  - Batch GD: Use all data
  - Stochastic GD: Use single sample
  - Mini-batch GD: Use small batches
- **Momentum**: Accelerates convergence

---

## Hour 4: Deep Learning Frameworks (TensorFlow/PyTorch)
**Time: 60 minutes**

### Framework Basics (30 mins)
- **TensorFlow/Keras**: High-level API, eager execution
- **PyTorch**: Dynamic computation graphs, research-friendly
- **Key Concepts**: Tensors, automatic differentiation, GPU acceleration

### Essential Operations (30 mins)
- **Model Definition**: Sequential vs Functional API
- **Layers**: Dense, Conv2D, LSTM, Dropout
- **Compilation**: Optimizer, loss function, metrics
- **Training Loop**: fit(), evaluate(), predict()
- **Saving/Loading**: Model persistence

---

## Hour 5: Recurrent Neural Networks (RNNs) and LSTMs
**Time: 60 minutes**

### RNN Fundamentals (30 mins)
- **Sequential Data**: Time series, text, speech
- **Hidden State**: h_t = f(W_h·h_{t-1} + W_x·x_t + b)
- **Vanishing Gradient Problem**: Why standard RNNs struggle
- **Applications**: Language modeling, sentiment analysis

### LSTM Architecture (30 mins)
- **Cell State**: Long-term memory
- **Gates**: Forget, input, output gates
- **Forget Gate**: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
- **Input Gate**: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
- **Output Gate**: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
- **Advantages**: Handles long sequences, solves vanishing gradients

---

## Hour 6: Convolutional Neural Networks (CNNs)
**Time: 60 minutes**

### CNN Architecture (30 mins)
- **Convolution Operation**: Feature detection through filters
- **Pooling**: Max pooling, average pooling for dimensionality reduction
- **Typical Architecture**: Conv → ReLU → Pool → Conv → ReLU → Pool → FC
- **Parameters**: Filter size, stride, padding

### Key Concepts (30 mins)
- **Feature Maps**: Output of convolution layers
- **Receptive Field**: Region of input that affects output
- **Translation Invariance**: Same features detected regardless of position
- **Applications**: Image classification, object detection, medical imaging
- **Famous Architectures**: LeNet, AlexNet, VGG, ResNet

---

## Hour 7: Introduction to Reinforcement Learning
**Time: 60 minutes**

### RL Fundamentals (30 mins)
- **Agent-Environment Interaction**: Actions, states, rewards
- **Policy (π)**: Strategy for action selection
- **Value Function**: Expected cumulative reward
- **Exploration vs Exploitation**: Balance between trying new actions and using known good actions

### Key Concepts (30 mins)
- **Episode**: Complete sequence from start to terminal state
- **Discount Factor (γ)**: Importance of future rewards
- **Return**: G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
- **Bellman Equation**: Foundation of dynamic programming
- **Types**: Model-based vs model-free, on-policy vs off-policy

---

## Hour 8: Markov Decision Process (MDP) and Markov Reward Process
**Time: 60 minutes**

### Markov Property (20 mins)
- **Definition**: Future depends only on current state
- **Mathematical**: P(S_{t+1}|S_t) = P(S_{t+1}|S_1, S_2, ..., S_t)
- **State Space**: All possible states
- **Transition Matrix**: Probabilities between states

### MDP Components (20 mins)
- **States (S)**: All possible situations
- **Actions (A)**: Available choices in each state
- **Rewards (R)**: Immediate feedback
- **Transition Probabilities**: P(s'|s,a)
- **Policy**: π(a|s) - probability of taking action a in state s

### Value Functions (20 mins)
- **State Value**: V^π(s) = E[G_t|S_t = s]
- **Action Value**: Q^π(s,a) = E[G_t|S_t = s, A_t = a]
- **Bellman Equations**: Recursive relationship for optimal values
- **Optimal Policy**: π* that maximizes expected return

---

## Hour 9: Q-Learning
**Time: 60 minutes**

### Q-Learning Algorithm (30 mins)
- **Temporal Difference Learning**: Learn from experience
- **Q-Table**: Store Q-values for state-action pairs
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Off-policy**: Learn optimal policy while following another policy

### Implementation Details (30 mins)
- **ε-greedy**: Balance exploration and exploitation
- **Learning Rate (α)**: Controls how much to update
- **Convergence**: Conditions for guaranteed convergence
- **Limitations**: Requires discrete, small state spaces
- **Applications**: Grid world, simple games

---

## Hour 10: Deep Q-Learning (DQN)
**Time: 60 minutes**

### DQN Architecture (30 mins)
- **Neural Network**: Approximate Q-function for large state spaces
- **Experience Replay**: Store and sample past experiences
- **Target Network**: Separate network for stable targets
- **Loss Function**: MSE between predicted and target Q-values

### Key Innovations (30 mins)
- **Experience Replay Buffer**: Break correlation between consecutive samples
- **Target Network Updates**: Periodic updates for stability
- **Double DQN**: Reduce overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Applications**: Atari games, robotics, game playing

---

## Hour 11: Advanced MDP Concepts
**Time: 60 minutes**

### Policy Iteration (20 mins)
- **Policy Evaluation**: Compute V^π for given policy
- **Policy Improvement**: Update policy based on value function
- **Convergence**: Guaranteed to find optimal policy

### Value Iteration (20 mins)
- **Direct Optimization**: Find optimal value function
- **Bellman Optimality**: V*(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV*(s')]
- **Comparison**: Policy iteration vs value iteration

### Advanced Topics (20 mins)
- **Partial Observability**: When agent can't see full state
- **Continuous Spaces**: Function approximation
- **Multi-agent**: Multiple learning agents
- **Hierarchical RL**: Learning at multiple levels

---

## Hour 12: Integration and Review
**Time: 60 minutes**

### Connections Between Topics (20 mins)
- **Neural Networks → Deep Learning**: Scaling up networks
- **Deep Learning → CNNs/RNNs**: Specialized architectures
- **RL → Deep RL**: Function approximation with neural networks
- **MDPs → Algorithms**: Theoretical foundation to practical methods

### Key Formulas Summary (20 mins)
- **Backpropagation**: Chain rule for gradient computation
- **LSTM Gates**: Forget, input, output gate equations
- **Bellman Equation**: V(s) = Σ π(a|s) Σ P(s'|s,a)[R + γV(s')]
- **Q-Learning**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

### Practical Applications (20 mins)
- **Computer Vision**: CNNs for image recognition
- **Natural Language**: RNNs/LSTMs for text processing
- **Game Playing**: Deep Q-learning for strategic games
- **Robotics**: RL for control and navigation

---

## Study Tips for Success

### Active Learning Strategies
- **Draw diagrams**: Visualize network architectures
- **Work examples**: Trace through forward/backward propagation
- **Code snippets**: Write simple implementations
- **Connect concepts**: Link theory to applications

### Time Management
- **Pomodoro**: 25-minute focused sessions with 5-minute breaks
- **Priority**: Focus on understanding over memorization
- **Review**: Spend 5 minutes at end of each hour reviewing
- **Notes**: Keep a formula sheet for quick reference

### Common Pitfalls to Avoid
- **Passive reading**: Actively engage with material
- **Skipping math**: Understand the underlying mathematics
- **Isolated learning**: Connect topics to each other
- **Perfectionism**: Aim for solid understanding, not mastery

### Final Recommendations
This is an intensive crash course. For deeper understanding, plan to revisit these topics with hands-on practice, coding exercises, and real projects. The goal is to build a strong foundation you can expand upon.
