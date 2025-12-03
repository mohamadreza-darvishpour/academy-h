# Program to find the 8th Fibonacci number

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(8))  # Output: 21
