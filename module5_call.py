
from module5_mod import NumberProcessor

def main():
    # Initialize the processor
    processor = NumberProcessor()
    
    # Number of numbers to be collected
    n = int(input("Please enter N: "))
    
    # Ask for numbers input one by one
    for i in range(n):
        num = int(input(f"Please enter positive integer {i+1}: "))
        processor.add_number(num)
    
    # Ask for the target
    X = int(input("Please enter X: "))
    
    # Find the index
    result = processor.find_index(X)
    print(result)

if __name__ == "__main__":
    main()


