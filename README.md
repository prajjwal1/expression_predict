# expression_predict

The task is the following:
Given the expression on the left hand side (before '=' sign), build a neural network that can predict what the expanded expression would be. 
Some examples are given below:

```
(7-3*z)*(-5*z-9)=15*z**2-8*z-63
-9*s**2=-9*s**2
(2-2*n)*(n-1)=-2*n**2+4*n-2
x**2=x**2
(4-x)*(x-23)=-x**2+27*x-92
(7-5*c)*(3*c-17)=-15*c**2+106*c-119
-8*x*(3*x+14)=-24*x**2-112*x
-2*k*(5*k-9)=-10*k**2+18*k
(3*cos(c)-19)*(7*cos(c)+13)=21*cos(c)**2-94*cos(c)-247
-8*j*(-8*j-3)=64*j**2+24*j
(h+8)*(7*h-3)=7*h**2+53*h-24
18*h**2=18*h**2
```
In the first example, if we are provided with the expression `(7-3*z)*(-5*z-9)`, our model needs to predict the expression `15*z**2-8*z-63`. 

This project is an implementation of a transformer based autoregressive neural LM that can predict the expression with very high accuracy.


### Running the code
```
python3 main.py
```

This will run evaluation on entire training set.
