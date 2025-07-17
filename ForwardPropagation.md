# Forward Propagation - Hour 2 (30 minutes)

## Overview: What is Forward Propagation?

**Definition**: The process of passing input data through a neural network to generate predictions
**Goal**: Transform raw input into meaningful output through learned transformations
**Flow**: Data moves forward through layers, each applying weights, biases, and activation functions

---

## 1. The Core Process (10 minutes)

### Four-Step Process at Each Layer

```
Input → Weighted Sum → Activation → Next Layer
```

### Step 1: Input
- **Raw data** enters the network
- **Format**: Vector or matrix of numerical values
- **Examples**: 
  - Image: [0.2, 0.8, 0.1, 0.9, ...] (pixel values)
  - Text: [0, 1, 0, 1, 0, ...] (word presence)
  - Sensor: [23.5, 45.2, 12.8] (temperature, humidity, pressure)

### Step 2: Weighted Sum
- **Each neuron** computes: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
- **Purpose**: Combine inputs with learned importance weights
- **Result**: Raw activation value (pre-activation)

### Step 3: Activation Function
- **Apply non-linear function**: `a = f(z)`
- **Purpose**: Introduce non-linearity, control output range
- **Result**: Neuron's final output (post-activation)

### Step 4: Pass to Next Layer
- **Current layer's output** becomes next layer's input
- **Repeat process** until final output layer
- **Final result**: Network's prediction

---

## 2. Matrix Operations (10 minutes)

### The Mathematical Foundation

**Key Equations**:
```
Z = W·X + B    (Linear transformation)
A = f(Z)       (Non-linear activation)
```

### Matrix Dimensions

**For a layer with:**
- **Input size**: n (number of features coming in)
- **Output size**: m (number of neurons in current layer)
- **Batch size**: k (number of examples processed together)

**Matrix shapes**:
```
X: (n × k) - Input matrix
W: (m × n) - Weight matrix  
B: (m × 1) - Bias vector
Z: (m × k) - Pre-activation matrix
A: (m × k) - Post-activation matrix
```

### Why Matrix Operations?

**Efficiency**: Process multiple examples simultaneously
**Parallelization**: GPU can compute many operations at once
**Clean Code**: One line instead of nested loops

### Detailed Matrix Example

**Network**: 3 inputs → 2 hidden neurons → 1 output
**Batch size**: 2 examples

**Input Matrix X (3 × 2)**:
```
X = [0.5  0.8]  ← Feature 1 for examples 1 & 2
    [0.3  0.2]  ← Feature 2 for examples 1 & 2  
    [0.1  0.9]  ← Feature 3 for examples 1 & 2
```

**Weight Matrix W (2 × 3)**:
```
W = [0.2  0.4  0.1]  ← Weights for neuron 1
    [0.3  0.1  0.5]  ← Weights for neuron 2
```

**Bias Vector B (2 × 1)**:
```
B = [0.1]  ← Bias for neuron 1
    [0.2]  ← Bias for neuron 2
```

**Matrix Multiplication Z = W·X + B**:
```
Z = [0.2  0.4  0.1] × [0.5  0.8] + [0.1]
    [0.3  0.1  0.5]   [0.3  0.2]   [0.2]
                       [0.1  0.9]

Step by step:
Neuron 1, Example 1: 0.2×0.5 + 0.4×0.3 + 0.1×0.1 + 0.1 = 0.1 + 0.12 + 0.01 + 0.1 = 0.33
Neuron 1, Example 2: 0.2×0.8 + 0.4×0.2 + 0.1×0.9 + 0.1 = 0.16 + 0.08 + 0.09 + 0.1 = 0.43
Neuron 2, Example 1: 0.3×0.5 + 0.1×0.3 + 0.5×0.1 + 0.2 = 0.15 + 0.03 + 0.05 + 0.2 = 0.43
Neuron 2, Example 2: 0.3×0.8 + 0.1×0.2 + 0.5×0.9 + 0.2 = 0.24 + 0.02 + 0.45 + 0.2 = 0.91

Z = [0.33  0.43]
    [0.43  0.91]
```

**Apply Activation A = f(Z)**:
```
If using sigmoid: A = 1/(1 + e^(-Z))
A = [σ(0.33)  σ(0.43)] = [0.582  0.606]
    [σ(0.43)  σ(0.91)]   [0.606  0.713]
```

---

## 3. Layer-by-Layer Computation (10 minutes)

### Complete Network Example

**Problem**: Classify images as cat (0) or dog (1)
**Architecture**: 784 (input) → 128 (hidden) → 64 (hidden) → 1 (output)

### Layer 1: Input Layer
```
Input: Flattened 28×28 image
X₁ = [0.2, 0.8, 0.1, 0.9, ..., 0.3]  (784 values)
Shape: (784, 1)
```

### Layer 2: First Hidden Layer
```
Computation:
Z₂ = W₂·X₁ + B₂
A₂ = ReLU(Z₂)

Where:
W₂: (128 × 784) - connects 784 inputs to 128 neurons
B₂: (128 × 1) - bias for each of 128 neurons
Z₂: (128 × 1) - pre-activation values
A₂: (128 × 1) - post-activation values

Example neuron:
Z₂[0] = Σ(W₂[0,i] × X₁[i]) + B₂[0] = 0.45
A₂[0] = ReLU(0.45) = 0.45
```

### Layer 3: Second Hidden Layer
```
Computation:
Z₃ = W₃·A₂ + B₃
A₃ = ReLU(Z₃)

Where:
W₃: (64 × 128) - connects 128 neurons to 64 neurons
B₃: (64 × 1) - bias for each of 64 neurons
Z₃: (64 × 1) - pre-activation values
A₃: (64 × 1) - post-activation values
```

### Layer 4: Output Layer
```
Computation:
Z₄ = W₄·A₃ + B₄
A₄ = Sigmoid(Z₄)

Where:
W₄: (1 × 64) - connects 64 neurons to 1 output
B₄: (1 × 1) - bias for output neuron
Z₄: (1 × 1) - pre-activation value
A₄: (1 × 1) - final prediction

Final output: A₄ = 0.73 → 73% probability it's a dog
```

### Information Flow Summary
```
784 pixels → 128 feature detectors → 64 pattern recognizers → 1 classifier
   [Raw]  →    [Edge detection]    →   [Shape recognition]  →  [Cat vs Dog]
```

---

## Complete Worked Example

### Problem Setup
**Task**: Recognize handwritten digit (0-9)
**Architecture**: 4 → 3 → 2 → 1 (simplified for demonstration)
**Input**: [0.8, 0.2, 0.1, 0.9] (4 pixel values)

### Network Parameters
**Layer 1 to 2 (4 → 3)**:
```
W₁ = [0.1  0.2  0.3  0.4]
     [0.2  0.1  0.4  0.3]
     [0.3  0.4  0.1  0.2]

B₁ = [0.1]
     [0.2]
     [0.1]
```

**Layer 2 to 3 (3 → 2)**:
```
W₂ = [0.5  0.3  0.2]
     [0.4  0.1  0.6]

B₂ = [0.1]
     [0.2]
```

**Layer 3 to 4 (2 → 1)**:
```
W₃ = [0.7  0.3]
B₃ = [0.1]
```

### Forward Pass Computation

**Input**: X = [0.8, 0.2, 0.1, 0.9]ᵀ

**Layer 1 → 2**:
```
Z₁ = W₁·X + B₁
Z₁ = [0.1×0.8 + 0.2×0.2 + 0.3×0.1 + 0.4×0.9] + [0.1] = [0.53]
     [0.2×0.8 + 0.1×0.2 + 0.4×0.1 + 0.3×0.9]   [0.2]   [0.61]
     [0.3×0.8 + 0.4×0.2 + 0.1×0.1 + 0.2×0.9]   [0.1]   [0.63]

A₁ = ReLU(Z₁) = [0.53, 0.61, 0.63]ᵀ
```

**Layer 2 → 3**:
```
Z₂ = W₂·A₁ + B₂
Z₂ = [0.5×0.53 + 0.3×0.61 + 0.2×0.63] + [0.1] = [0.574]
     [0.4×0.53 + 0.1×0.61 + 0.6×0.63]   [0.2]   [0.751]

A₂ = ReLU(Z₂) = [0.574, 0.751]ᵀ
```

**Layer 3 → 4**:
```
Z₃ = W₃·A₂ + B₃
Z₃ = [0.7×0.574 + 0.3×0.751] + [0.1] = [0.627]

A₃ = Sigmoid(Z₃) = [0.652]
```

**Final Output**: 0.652 → 65.2% confidence in classification

---

## Key Insights

### Why Forward Propagation Works

1. **Hierarchical Feature Learning**:
   - Early layers: Simple features (edges, corners)
   - Middle layers: Complex patterns (shapes, textures)
   - Final layers: High-level concepts (objects, classes)

2. **Non-linear Transformations**:
   - Each layer applies: Linear → Non-linear → Linear → Non-linear...
   - Builds increasingly complex decision boundaries

3. **Distributed Representation**:
   - Information spread across multiple neurons
   - Robust to individual neuron failures
   - Rich feature representations

### Common Mistakes to Avoid

1. **Dimension Mismatches**:
   - Always check matrix dimensions align
   - (m×n) × (n×k) = (m×k)

2. **Forgetting Bias**:
   - Bias terms are crucial for flexibility
   - Don't skip them in calculations

3. **Wrong Activation Functions**:
   - Hidden layers: ReLU (usually)
   - Output layer: Sigmoid (binary), Softmax (multi-class)

### Computational Efficiency

**Sequential Processing**: O(n) time
```python
for layer in network:
    output = layer.forward(input)
    input = output
```

**Batch Processing**: Same O(n) time, but processes multiple examples
```python
# Process 32 examples simultaneously
batch_output = network.forward(batch_input)  # Shape: (output_size, 32)
```

---

## Visual Summary

```
FORWARD PROPAGATION FLOW:

Input Data → Layer 1 → Layer 2 → ... → Output
   [X]         [Z₁=W₁X+B₁]   [Z₂=W₂A₁+B₂]      [Prediction]
               [A₁=f(Z₁)]     [A₂=f(Z₂)]

MATRIX OPERATIONS:
Each layer: Z = W·A_prev + B, then A = f(Z)

INFORMATION EXTRACTION:
Raw Data → Low-level Features → High-level Patterns → Final Decision
```

## Self-Check Questions:
1. Can you trace data through a 3-layer network step by step?
2. What are the matrix dimensions for each operation?
3. Why do we need activation functions between layers?
4. How does batch processing improve efficiency?

**Next**: Activation Functions - the non-linear components that make deep learning possible!

*Time check: 30 minutes total for forward propagation concepts*
