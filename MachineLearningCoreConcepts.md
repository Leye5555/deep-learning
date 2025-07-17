# Machine Learning Core Concepts - Hour 1 (20 minutes)

## 1. Machine Learning Types (7 minutes)

### Supervised Learning
**Definition**: Learning from labeled data (input-output pairs)
- **Goal**: Predict outputs for new inputs
- **Examples**: 
  - Email spam detection (email → spam/not spam)
  - Image classification (image → cat/dog)
  - House price prediction (features → price)
- **Key Characteristics**:
  - Has "ground truth" labels
  - Performance measured by accuracy on test data
  - Two main types: Classification (discrete outputs) and Regression (continuous outputs)

### Unsupervised Learning
**Definition**: Learning patterns from data without labels
- **Goal**: Discover hidden structure in data
- **Examples**:
  - Customer segmentation (group similar customers)
  - Anomaly detection (find unusual patterns)
  - Data compression (reduce dimensionality)
- **Key Characteristics**:
  - No "correct" answers provided
  - Harder to evaluate performance
  - Main types: Clustering, Dimensionality reduction, Association rules

### Reinforcement Learning
**Definition**: Learning through interaction with environment via rewards/penalties
- **Goal**: Learn optimal actions to maximize cumulative reward
- **Examples**:
  - Game playing (chess, Go, video games)
  - Robot navigation
  - Trading algorithms
- **Key Characteristics**:
  - Agent takes actions and receives feedback
  - Delayed rewards (actions now affect future outcomes)
  - Balance between exploration and exploitation

**Quick Memory Tip**: Supervised = teacher present, Unsupervised = no teacher, Reinforcement = learning through trial and error

---

## 2. Neural Network Basics (7 minutes)

### Perceptron (Single Neuron)
**Definition**: Simplest neural network unit that makes binary decisions

**Mathematical Model**:
```
Input: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b
Output: y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

**How it works**:
1. **Weighted Sum**: Multiply each input by its weight
2. **Add Bias**: Add bias term (shifts decision boundary)
3. **Activation**: Apply activation function (usually step function for classic perceptron)

**Example**: Email spam detection
- Inputs: word counts (x₁ = "free", x₂ = "money", x₃ = "urgent")
- Weights: importance of each word for spam detection
- Output: spam (1) or not spam (0)

**Limitation**: Can only solve linearly separable problems (like AND, OR gates, but not XOR)

### Multi-Layer Perceptrons (MLPs)
**Definition**: Neural networks with multiple layers of perceptrons

**Architecture**:
- **Input Layer**: Receives data
- **Hidden Layer(s)**: Process information (can have multiple)
- **Output Layer**: Produces final result

**Why Multiple Layers**:
- **Universal Approximation**: Can approximate any continuous function
- **Complex Patterns**: Learn non-linear relationships
- **Hierarchical Features**: Each layer learns increasingly complex features

**Example Structure**:
```
Input (784 pixels) → Hidden (128 neurons) → Hidden (64 neurons) → Output (10 classes)
```

---

## 3. Key Components (6 minutes)

### Neurons (Nodes)
**Definition**: Basic processing units that receive inputs, process them, and produce output

**Function**: Each neuron:
1. Receives multiple inputs
2. Multiplies each input by corresponding weight
3. Sums all weighted inputs
4. Adds bias term
5. Applies activation function
6. Sends output to next layer

**Biological Inspiration**: 
- Inputs = dendrites receiving signals
- Processing = cell body
- Output = axon sending signal

### Weights
**Definition**: Parameters that determine strength of connections between neurons

**Purpose**:
- **Signal Strength**: How much influence one neuron has on another
- **Learning**: Weights are adjusted during training to improve performance
- **Pattern Recognition**: Different weight patterns recognize different features

**Key Points**:
- **Positive weights**: Excitatory connections (increase activation)
- **Negative weights**: Inhibitory connections (decrease activation)
- **Large weights**: Strong influence
- **Small weights**: Weak influence

**Learning Process**: Start with random weights, then adjust based on errors

### Biases
**Definition**: Additional parameters that shift the activation function

**Purpose**:
- **Flexibility**: Allow neuron to activate even when all inputs are zero
- **Decision Boundary**: Shift where neuron "fires"
- **Offset**: Similar to y-intercept in linear equations

**Mathematical Role**:
```
Without bias: y = f(w₁x₁ + w₂x₂)
With bias: y = f(w₁x₁ + w₂x₂ + b)
```

**Analogy**: Like adjusting the sensitivity of a light switch

### Layers
**Definition**: Groups of neurons that process information at the same level

**Types**:
1. **Input Layer**: 
   - Receives raw data
   - Number of neurons = number of input features
   - No processing, just passes data forward

2. **Hidden Layer(s)**:
   - Process and transform information
   - Extract features and patterns
   - Can have multiple hidden layers (deep networks)

3. **Output Layer**:
   - Produces final predictions
   - Number of neurons depends on task:
     - Binary classification: 1 neuron
     - Multi-class classification: 1 neuron per class
     - Regression: 1 neuron per output value

**Information Flow**: Always forward (in basic networks)
Input → Hidden → Hidden → ... → Output

---

## Visual Summary

```
MACHINE LEARNING TYPES:
Supervised: [Data with labels] → [Learn mapping] → [Predict new data]
Unsupervised: [Data without labels] → [Find patterns] → [Discover structure]
Reinforcement: [Agent] ↔ [Environment] → [Learn optimal actions]

NEURAL NETWORK STRUCTURE:
Input Layer → Hidden Layer(s) → Output Layer
[x₁, x₂, x₃] → [neurons with weights & biases] → [predictions]

PERCEPTRON FORMULA:
output = activation_function(Σ(weight_i × input_i) + bias)
```

## Key Takeaways for Next Section:
- Neural networks are inspired by biological neurons
- Multiple layers enable complex pattern recognition
- Weights and biases are the learnable parameters
- Different ML types solve different kinds of problems
- Understanding these basics is crucial for forward propagation (next topic)

## Quick Self-Check Questions:
1. What's the difference between supervised and unsupervised learning?
2. Why do we need multiple layers in neural networks?
3. What role do weights and biases play in a neuron?
4. Can a single perceptron solve the XOR problem? Why or why not?

*Time Check: This should take about 20 minutes to read and understand. Move to forward propagation next!*
