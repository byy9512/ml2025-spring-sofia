
# Define range
n = int(input("Please enter N: "))

# Collect inputs from the user
numbers = []
for i in range(n):
    num = int(input(f"Please enter positive integer {i+1}: "))
    numbers.append(num)

# Collect target
X = int(input("Please enter X: "))

# Validate if the target exists and return the position
def get_index(numbers, X):
    if X in numbers:
        return numbers.index(X)
    else:
        return -1

index = get_index(numbers, X)

if index == -1:
    print(-1)
else:
    print(index + 1)



