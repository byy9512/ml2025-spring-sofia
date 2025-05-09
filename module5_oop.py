

class NumberProcessor:
    def __init__(self):
        self.numbers = []
    
    def add_number(self, num):
        """
        Insert N numbers into the list one by one
        """
        self.numbers.append(num)
    
    def find_index(self, target):
        """
        Find the index of the target, if not, return -1
        """
        if target in self.numbers:
            return self.numbers.index(target) + 1
        return -1

def main():
       
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

