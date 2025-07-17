# Mathematical Foundations - Hour 1 (20 minutes)

## 1. Linear Algebra (8 minutes)

### Vectors
**Definition**: Arrays of numbers representing data points or directions

**Types**:
- **Row vector**: [3, 1, 4] (horizontal)
- **Column vector**: [3; 1; 4] (vertical)

**In Neural Networks**:
- **Input vector**: [pixel₁, pixel₂, ..., pixelₙ]
- **Weight vector**: [w₁, w₂, ..., wₙ]
- **Output vector**: [class₁_prob, class₂_prob, ..., classₙ_prob]

**Key Operations**:
- **Addition**: [1, 2] + [3, 4] = [4, 6]
- **Scalar multiplication**: 2 × [1, 2] = [2, 4]
- **Magnitude**: ||[3, 4]|| = √(3² + 4²) = 5

### Matrices
**Definition**: 2D arrays of numbers (rows × columns)

**Examples**:
```
Weight matrix W:
[0.5  0.3  0.1]
[0.2  0.8  0.4]
[0.1  0.6  0.9]
```

**In Neural Networks**:
- **Weight matrices**: Connect layers (rows = output neurons, columns = input neurons)
- **Data matrices**: Each row = one training example
- **Activation matrices**: Store neuron outputs

**Key Operations**:
- **Matrix addition**: Element-wise addition
- **Scalar multiplication**: Multiply every element
- **Transpose**: A^T flips rows and columns

### Dot Products (Scalar Product)
**Definition**: Multiply corresponding elements and sum

**Vector dot product**:
```
a = [1, 2, 3]
b = [4, 5, 6]
a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

**Neural Network Application**:
```
Input: x = [x₁, x₂, x₃]
Weights: w = [w₁, w₂, w₃]
Neuron activation: z = w · x + b = w₁x₁ + w₂x₂ + w₃x₃ + b
```

**Matrix Multiplication**:
```
Result[i,j] = Σ(A[i,k] × B[k,j])
```

**Why Important**: Every neuron computation is essentially a dot product followed by activation!

**Example in Neural Networks**:
```
Layer input: X = [2, 3, 1]
Weights: W = [0.5, 0.3, 0.2]
Neuron output before activation: z = W · X = 0.5×2 + 0.3×3 + 0.2×1 = 1.0 + 0.9 + 0.2 = 2.1
```

---

## 2. Basic Calculus (8 minutes)

### Partial Derivatives
**Definition**: Rate of change of a function with respect to one variable, keeping others constant

**Notation**: ∂f/∂x (partial derivative of f with respect to x)

**Simple Examples**:
```
f(x,y) = x² + 3xy + y²
∂f/∂x = 2x + 3y (treat y as constant)
∂f/∂y = 3x + 2y (treat x as constant)
```

**In Neural Networks**:
- **Loss function**: L(w₁, w₂, ..., wₙ, b₁, b₂, ..., bₘ)
- **Gradient**: Vector of all partial derivatives
- **Goal**: Find ∂L/∂w and ∂L/∂b to update weights and biases

**Practical Example**:
```
Loss function: L = (y_predicted - y_actual)²
If y_predicted = wx + b, then:
∂L/∂w = 2(wx + b - y_actual) × x
∂L/∂b = 2(wx + b - y_actual) × 1
```

### Chain Rule
**Definition**: Method to find derivative of composite functions

**Formula**: If y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx)

**Neural Network Context**:
```
Input → Linear → Activation → Output → Loss
x → z = wx + b → a = σ(z) → L(a)

To find ∂L/∂w:
∂L/∂w = (∂L/∂a) × (∂a/∂z) × (∂z/∂w)
```

**Step-by-step Example**:
```
z = wx + b
a = σ(z) = 1/(1 + e^(-z))
L = (a - y)²

∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
       = 2(a - y) × σ(z)(1 - σ(z)) × x
```

**Why Critical**: Backpropagation is essentially repeated application of chain rule!

**Multi-layer Chain Rule**:
```
Input → Hidden → Output → Loss
∂L/∂w₁ = (∂L/∂output) × (∂output/∂hidden) × (∂hidden/∂w₁)
```

---

## 3. Probability (4 minutes)

### Basic Probability Concepts
**Definition**: Measure of likelihood of events (0 to 1)

**Key Rules**:
- **P(A) + P(not A) = 1**
- **P(A and B) = P(A) × P(B)** (if independent)
- **P(A or B) = P(A) + P(B) - P(A and B)**

### Probability Distributions
**Definition**: Functions that describe probability of different outcomes

### Normal (Gaussian) Distribution
**Formula**: f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
- **μ**: Mean (center)
- **σ**: Standard deviation (spread)
- **Properties**: Bell curve, symmetric around mean

**In Neural Networks**:
- **Weight initialization**: Often use normal distribution
- **Noise modeling**: Assume errors follow normal distribution
- **Regularization**: Some techniques assume normal priors

### Bernoulli Distribution
**Definition**: Binary outcomes (0 or 1)
- **P(X = 1) = p**
- **P(X = 0) = 1 - p**

**In Neural Networks**:
- **Binary classification**: Output represents probability
- **Dropout**: Randomly set neurons to 0 with probability p

### Categorical Distribution
**Definition**: Multiple discrete outcomes
- **Softmax output**: Converts scores to probabilities
- **Example**: P(cat) = 0.7, P(dog) = 0.2, P(bird) = 0.1

**In Neural Networks**:
- **Multi-class classification**: Each class has probability
- **Cross-entropy loss**: Measures difference between predicted and true distributions

---

## How These Connect to Neural Networks

### Forward Propagation (Linear Algebra)
```
Layer computation: A = σ(W × X + B)
Where:
- X: Input matrix (batch_size × input_features)
- W: Weight matrix (input_features × output_neurons)
- B: Bias vector (output_neurons)
- σ: Activation function
```

### Backpropagation (Calculus)
```
Weight update: W_new = W_old - α × ∂L/∂W
Where:
- α: Learning rate
- ∂L/∂W: Gradient (computed using chain rule)
```

### Loss Functions (Probability)
```
Cross-entropy loss: L = -Σ y_true × log(y_predicted)
Where:
- y_true: True probability distribution
- y_predicted: Predicted probability distribution
```

---

## Quick Reference Formulas

### Linear Algebra
```
Dot product: a · b = Σ(aᵢ × bᵢ)
Matrix multiplication: C[i,j] = Σ(A[i,k] × B[k,j])
```

### Calculus
```
Chain rule: ∂f/∂x = (∂f/∂u) × (∂u/∂x)
Gradient: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

### Probability
```
Normal: f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
Softmax: P(class i) = e^(zᵢ)/Σe^(zⱼ)
```

---

## Visual Summary

```
NEURAL NETWORK MATH FLOW:

Input Vector → Matrix Multiplication → Add Bias → Activation
    [x₁]         [w₁₁ w₁₂] [x₁]      [b₁]        σ(z₁)
    [x₂]    ×    [w₂₁ w₂₂] [x₂]  +   [b₂]   =    σ(z₂)
    
Forward: Linear Algebra (matrix operations)
Backward: Calculus (gradients via chain rule)
Output: Probability (interpret as class probabilities)
```

## Key Takeaways:
1. **Linear algebra** enables efficient computation of neuron activations
2. **Calculus** allows us to find gradients for learning
3. **Probability** helps interpret outputs and design loss functions
4. These three areas work together in every neural network operation

## Self-Check Questions:
1. How do you compute the output of a layer given input and weights?
2. What is the chain rule and why is it important for backpropagation?
3. Why do we use probability distributions in neural network outputs?

*Time Check: This should take about 20 minutes. Next: Practical Understanding of neural networks!*
