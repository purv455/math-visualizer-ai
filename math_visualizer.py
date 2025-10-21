import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from transformers import pipeline

# Load AI model for explanation
generator = pipeline('text-generation', model='distilgpt2')

def visualize_math(expression, operation):
    """
    Visualize a mathematical expression.
    expression: string, e.g., 'x**2 + 3*x - 4'
    operation: 'plot', 'solve', 'derivative', 'integral'
    """
    try:
        x = sp.symbols('x')
        expr = sp.sympify(expression)
        
        if operation == 'plot':
            # Plot the function
            func = sp.lambdify(x, expr, 'numpy')
            x_vals = np.linspace(-10, 10, 400)
            y_vals = func(x_vals)
            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals)
            plt.title(f'Plot of {expression}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            plt.savefig('plot.png')
            return 'plot.png', f"Plotted {expression}"
        
        elif operation == 'solve':
            # Solve equation if it's set to 0
            solution = sp.solve(expr, x)
            return None, f"Solutions: {solution}"
        
        elif operation == 'derivative':
            deriv = sp.diff(expr, x)
            return None, f"Derivative: {deriv}"
        
        elif operation == 'integral':
            integ = sp.integrate(expr, x)
            return None, f"Integral: {integ} + C"
        
        else:
            return None, "Unknown operation"
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_explanation(expression, operation, result):
    prompt = f"Explain the {operation} of the math expression '{expression}'. Result: {result}"
    explanation = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return explanation

# Streamlit interface
st.title("Math Visualizer AI")
st.write("Enter a math expression and choose an operation to visualize or compute.")

expression = st.text_input("Math Expression", "x**2 + 3*x - 4")
operation = st.selectbox("Operation", ['plot', 'solve', 'derivative', 'integral'])

if st.button("Compute"):
    plot_path, text = visualize_math(expression, operation)
    explanation = generate_explanation(expression, operation, text)
    if plot_path:
        st.image(plot_path)
    st.write("Result:", text)
    st.write("AI Explanation:", explanation)
