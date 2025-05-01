
# Define range
n = int(input("Please enter N: "))

# Collect inputs from the user
numbers = []
for i in range(n):
    num = int(input(f"Please enter number {i+1}: "))
    numbers.append(num)

# Collect target
X = int(input("Please enter X: "))

# Validate if target exists and return the position
if X in numbers:
    print(numbers.index(X) + 1)
else:
    print(-1)

