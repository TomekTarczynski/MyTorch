import pytest
import torch
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MyTorch import Value  # Importing Value class from MyTorch.py

def test_value_vs_torch():
    # Define values for a, b, and c
    val_a, val_b, val_c = 3.0, 4.0, 5.0
    
    # Initialize the Value class instances
    a_value = Value(val_a, name="a")
    b_value = Value(val_b, name="b")
    c_value = Value(val_c, name="c")

    # Perform operations with the Value class
    d_value = 2 * a_value + 3 * b_value
    e_value = d_value * c_value
    e_value.backward()

    # Initialize PyTorch tensors
    a_torch = torch.tensor(val_a, requires_grad=True)
    b_torch = torch.tensor(val_b, requires_grad=True)
    c_torch = torch.tensor(val_c, requires_grad=True)

    # Perform operations with PyTorch
    d_torch = 2 * a_torch + 3 * b_torch
    e_torch = d_torch * c_torch
    d_torch.retain_grad()  # Store gradient for d
    e_torch.retain_grad()  # Store gradient for e
    e_torch.backward()

    # Compare values and gradients
    assert pytest.approx(a_value.data) == a_torch.item(), "Values for a do not match"
    assert pytest.approx(b_value.data) == b_torch.item(), "Values for b do not match"
    assert pytest.approx(c_value.data) == c_torch.item(), "Values for c do not match"
    assert pytest.approx(d_value.data) == d_torch.item(), "Values for d do not match"
    assert pytest.approx(e_value.data) == e_torch.item(), "Values for e do not match"

    assert pytest.approx(a_value.grad) == a_torch.grad.item(), "Gradients for a do not match"
    assert pytest.approx(b_value.grad) == b_torch.grad.item(), "Gradients for b do not match"
    assert pytest.approx(c_value.grad) == c_torch.grad.item(), "Gradients for c do not match"
    assert pytest.approx(d_value.grad) == d_torch.grad.item(), "Gradients for c do not match"
    assert pytest.approx(e_value.grad) == e_torch.grad.item(), "Gradients for c do not match"    

def test_complex_value_vs_torch():
    # Define initial values
    val_a, val_b, val_c = 3.0, 4.0, 5.0

    # Initialize Value instances
    a_value = Value(val_a, name="a")
    b_value = Value(val_b, name="b")
    c_value = Value(val_c, name="c")

    # Perform complex operations with Value
    f_value = a_value * b_value  # f = a * b
    g_value = f_value + c_value  # g = f + c
    h_value = g_value * (a_value + b_value)  # h = g * (a + b)
    i_value = h_value * c_value  # i = h * c
    j_value = i_value + f_value  # j = i + f
    j_value.backward()  # Backpropagation to calculate gradients

    # Initialize PyTorch tensors
    a_torch = torch.tensor(val_a, requires_grad=True)
    b_torch = torch.tensor(val_b, requires_grad=True)
    c_torch = torch.tensor(val_c, requires_grad=True)

    # Perform complex operations with PyTorch, retaining gradients for each intermediate step
    f_torch = a_torch * b_torch  # f = a * b
    f_torch.retain_grad()
    g_torch = f_torch + c_torch  # g = f + c
    g_torch.retain_grad()
    h_torch = g_torch * (a_torch + b_torch)  # h = g * (a + b)
    h_torch.retain_grad()
    i_torch = h_torch * c_torch  # i = h * c
    i_torch.retain_grad()
    j_torch = i_torch + f_torch  # j = i + f
    j_torch.retain_grad()
    j_torch.backward()  # Backpropagation to calculate gradients

    # Compare values and gradients for each variable and intermediate calculation
    assert pytest.approx(a_value.data) == a_torch.item(), "Values for a do not match"
    assert pytest.approx(b_value.data) == b_torch.item(), "Values for b do not match"
    assert pytest.approx(c_value.data) == c_torch.item(), "Values for c do not match"
    assert pytest.approx(f_value.data) == f_torch.item(), "Intermediate values for f do not match"
    assert pytest.approx(g_value.data) == g_torch.item(), "Intermediate values for g do not match"
    assert pytest.approx(h_value.data) == h_torch.item(), "Intermediate values for h do not match"
    assert pytest.approx(i_value.data) == i_torch.item(), "Intermediate values for i do not match"
    assert pytest.approx(j_value.data) == j_torch.item(), "Final values for j do not match"

    # Compare gradients for each variable and intermediate calculation
    assert pytest.approx(a_value.grad) == a_torch.grad.item(), "Gradients for a do not match"
    assert pytest.approx(b_value.grad) == b_torch.grad.item(), "Gradients for b do not match"
    assert pytest.approx(c_value.grad) == c_torch.grad.item(), "Gradients for c do not match"
    assert pytest.approx(f_value.grad) == f_torch.grad.item(), "Gradients for f do not match"
    assert pytest.approx(g_value.grad) == g_torch.grad.item(), "Gradients for g do not match"
    assert pytest.approx(h_value.grad) == h_torch.grad.item(), "Gradients for h do not match"
    assert pytest.approx(i_value.grad) == i_torch.grad.item(), "Gradients for i do not match"
    assert pytest.approx(j_value.grad) == j_torch.grad.item(), "Gradients for j do not match"

