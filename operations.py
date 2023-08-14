import math

class Operations():

    def __init__(self, operations):
        self.operations = operations
    
    def get_operations(self):
        return {idx: getattr(self,x) for idx, x in enumerate(self.operations) if x in dir(self)} # k=idx, v=func

    ### Binary operations ###

    @staticmethod
    def addition(op1,op2):
        try:
            return op1+op2
        except:
            print("caught except")
            return 1.0e6

    @staticmethod
    def subtraction(op1,op2):
        try:
            return op1-op2
        except:
            return 1.0e6

    @staticmethod
    def multiplication(op1,op2):
        try:
            return op1*op2
        except:
            print("caught except")
            return 1.0e6

    @staticmethod
    def division(op1,op2):
        return op1/op2 if op2 != 0 else 1.0e6

    @staticmethod
    def conditional_branch(op1,op2):
        return op1 > op2

    ### Unary operations ###

    @staticmethod
    def sin(op1,_):
        try:
            return math.sin(op1)
        except:
            print(op1)
            return 1.0e6

    @staticmethod
    def cos(op1,_):
        try:
            return math.cos(op1)
        except:
            print(op1)
            return 1.0e6

    @staticmethod
    def square(op1,_):
        try:
            return op1**2
        except:
            return 1.0e6

    @staticmethod
    def sqrt(op1,_):
        return math.sqrt(op1)

    @staticmethod
    def exponentiation(op1,_):
        return math.exp(op1)


if __name__ == "__main__":

    operations = ["addition","division","square"]
    operators = Operations(operations).get_operations()
    operation_mapping = {idx:v for idx, (_,v) in enumerate(operators.items())} # idx to operation

    print(operation_mapping)







