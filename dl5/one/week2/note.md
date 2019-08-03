# 3 - General Architecture of the learning algorithm ##

It's time to design a simple algorithm to distinguish cat images from non-cat images.

You will build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**

<img src="images/LogReg_kiank.png" style="width:650px;height:400px;">

**Mathematical expression of the algorithm**:

For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

**Key steps**:
In this exercise, you will carry out the following steps: 
    - Initialize the parameters of the model
    - Learn the parameters for the model by minimizing the cost  
    - Use the learned parameters to make predictions (on the test set)
    - Analyse the results and conclude



# Forward and Backward propagation

Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.

**Exercise:** Implement a function `propagate()` that computes the cost function and its gradient.

**Hints**:

Forward Propagation:
- You get X
- You compute $A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})$
- You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$

Here are the two formulas you will be using: 

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

### 推导
$$
J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}) \\

\frac{\partial J}{\partial a^{(i)}} = -\frac{1}{m} [\frac{y^{(i)}}{a^{(i)}} \frac{\partial J}{\partial a^{(i)}} + \frac{1-y^{(i)}}{1-a^{(i)}}(-\frac{\partial J}{\partial a^{(i)}})] = -\frac{1}{m} \frac{y^{(i)}-a^{(i)}}{a^{(i)}(1-a^{(i)})} = \frac{1}{m} \frac{a^{(i)}-y^{(i)}}{a^{(i)}(1-a^{(i)})} \\

\frac{\partial a^{(i)}}{\partial z^{(i)}} = {a^{(i)}(1-a^{(i)})} \\

\frac{{\partial z^{(i)}}}{\partial w^{(i)}_{j}} = x^{(i)}_{j} \\

\frac{\partial J}{\partial w^{(i)}_{j}} = \frac{\partial J}{\partial a^{(i)}} \frac{\partial a^{(i)}}{\partial z^{(i)}} \frac{\partial z^{(i)}}{\partial w^{(i)}_{j}} = \frac{1}{m} x^{(i)}_{j} (a^{(i)} - y^{(i)})

$$

### $\frac{\partial J}{\partial W}$
$$
W = \frac{1}{m}{
    \begin{pmatrix}
      \frac{\partial J}{\partial w^{(1)}_{0}} + \frac{\partial J}{\partial w^{(2)}_{0}} +...+ \frac{\partial J}{\partial w^{(m)}_{0}}\\
      \frac{\partial J}{\partial w^{(1)}_{1}} + \frac{\partial J}{\partial w^{(2)}_{1}} +...+ \frac{\partial J}{\partial w^{(m)}_{1}}\\
      ...................................\\
      \frac{\partial J}{\partial w^{(1)}_{j}} + \frac{\partial J}{\partial w^{(2)}_{j}} +...+ \frac{\partial J}{\partial w^{(m)}_{j}}\\
      ....................................\\
      \frac{\partial J}{\partial w^{(1)}_{n-1}} + \frac{\partial J}{\partial w^{(2)}_{n-1}} +...+ \frac{\partial J}{\partial w^{(m)}_{n-1}}\\
    \end{pmatrix}
  }_{n×1}
$$


### 代价函数对W求偏导
$$
  \frac{\partial J}{\partial W} = \frac{1}{m} {
    \begin{pmatrix} 

    x^{(1)}_{0}(a^{(1)} - y^{(1)}) & x^{(2)}_{0}(a^{(2)} - y^{(2)}) &...& x^{(i)}_{0}(a^{(i)} - y^{(i)}) &...& x^{(m)}_{0}(a^{(m)} - y^{(m)})\\ 

    x^{(1)}_{1}(a^{(1)} - y^{(1)}) & x^{(2)}_{1}(a^{(2)} - y^{(2)}) &...& x^{(i)}_{1}(a^{(i)} - y^{(i)}) &...& x^{(m)}_{1}(a^{(m)} - y^{(m)})\\

    x^{(1)}_{2}(a^{(1)} - y^{(1)}) & x^{(2)}_{2}(a^{(2)} - y^{(2)}) &...& x^{(i)}_{2}(a^{(i)} - y^{(i)}) &...& x^{(m)}_{2}(a^{(m)} - y^{(m)})\\

    ...&...&...&...&...\\

    x^{(1)}_{j}(a^{(1)} - y^{(1)}) & x^{(2)}_{j}(a^{(2)} - y^{(2)}) &...& x^{(i)}_{j}(a^{(i)} - y^{(i)}) &...& x^{(m)}_{j}(a^{(m)} - y^{(m)})\\
    ...&...&...&...&...\\

    x^{(1)}_{n-1}(a^{(1)} - y^{(1)}) & x^{(2)}_{n-1}(a^{(2)} - y^{(2)}) &...& x^{(i)}_{n-1}(a^{(i)} - y^{(i)}) &...& x^{(m)}_{n-1}(a^{(m)} - y^{(m)})\\

    \end{pmatrix} 
  }_{n×m}
$$

### 继续
$$
  \frac{\partial J}{\partial W} = \frac{1}{m} X(A-Y)^{T}
$$


$$
Z = w^T X + b = (z^{(0)}, z^{(1)}, ..., z^{(m-1)}, z^{(m)}) \\
A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})\\
Y = (y^{(0)}, y^{(1)}, ..., y^{(m-1)}, y^{(m)}) \\
A-Y=(a^{(0)}-y^{(0)}, a^{(1)}-y^{(1)}, ..., a^{(m-1)}-y^{(m-1)}, a^{(m)}-y^{(m)}) \\
$$

### 继续推导

### $z^{(i)}$
$$
  z^{(i)}=x^{(i)}_{0}w_{0}+x^{(i)}_{1}w_{1}+x^{(i)}_{2}w_{2}+...+x^{(i)}_{n-1}w_{n-1}\\
  a^{(i)}=\sigma(z^{(i)})
$$


$$
A = (a^{(1)}, ..., a^{(m-1)}, a^{(m)})\\
  Z = (z^{(1)}, ..., z^{(m-1)}, z^{(m)})
$$


### $X$
$$
X= {
    \begin{pmatrix} 

    x^{(1)}_{0} & x^{(2)}_{0} &...& x^{(i)}_{0} &...& x^{(m)}_{0}\\ 
    x^{(1)}_{1} & x^{(2)}_{1} &...& x^{(i)}_{1} &...& x^{(m)}_{1}\\
    x^{(1)}_{2} & x^{(2)}_{2} &...& x^{(i)}_{2} &...& x^{(m)}_{2}\\
    ...&...&...&...&...\\
    x^{(1)}_{j} & x^{(2)}_{j} &...& x^{(i)}_{j} &...& x^{(m)}_{j}\\
    ...&...&...&...&...\\
    x^{(1)}_{n-1} & x^{(2)}_{n-1} &...& x^{(i)}_{n-1} &...& x^{(m)}_{n-1}\\

    \end{pmatrix} 
  }_{n×m}
$$

### $(A-Y)^{T}$
$$
(A-Y)^{T} = {
    \begin{pmatrix}
      a^{(1)}-y^{(1)}\\
      a^{(2)}-y^{(2)}\\ 
      ...\\
      a^{(i)}-y^{(i)}\\
      ...\\
      a^{(m)}-y^{(m)}
    \end{pmatrix}
  }_{m×1}
$$

$$
W = {
    \begin{pmatrix}
      w_{0}\\
      w_{1} \\
      w_{2}\\
      ...\\
      w_{j}\\
      ...\\
      w_{n-1}
    \end{pmatrix}
  }_{n×1}
$$


$$
 \left\{
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix}
  \right\} \tag{2}
$$


$$
\left[
\begin{matrix}
 1      & 2      & \cdots & 4      \\
 7      & 6      & \cdots & 5      \\
 \vdots & \vdots & \ddots & \vdots \\
 8      & 9      & \cdots & 0      \\
\end{matrix}
\right]
$$

$$
\begin{bmatrix}
a_{11}&a_{12}&\cdots &a_{1n} \\
a_{21}&a_{22}&\cdots &a_{2n} \\
\vdots & \vdots & \ddots & \vdots\\
a_{n1}&a_{n2}&\cdots &a_{nn}
\end{bmatrix}
$$
