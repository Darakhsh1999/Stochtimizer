import math
import sympy

class Operations():

    def __init__(self, operations=None):
        if operations is None:
            operations = [x for x in dir(self) if not x.startswith("__")]
        self.operations = operations 
    
    def get_operations(self):
        return {idx: getattr(self,x) for idx, x in enumerate(self.operations) if x in dir(self)} # k=idx, v=func

    ### Binary operations ###

    @staticmethod
    def addition(op1, op2, sym=False):
        result = op1+op2
        if sym: return result
        if result > 1.0e30:
            return 1.0e10
        else:
            return result

    @staticmethod
    def subtraction(op1, op2, sym=False):
        result = op1-op2
        if sym: return result
        if result > 1.0e30:
            return 1.0e10
        else:
            return result

    @staticmethod
    def multiplication(op1, op2, sym=False):
        result = op1*op2
        if sym: return result
        if result > 1.0e30:
            return 1.0e10
        else:
            return result

    @staticmethod
    def division(op1, op2, sym=False):
        result = op1/op2 if op2 != 0 else 1.0e6
        if sym: return result
        if result > 1.0e30:
            return 1.0e10
        else:
            return result

    @staticmethod
    def conditional_branch(op1, op2, _=False):
        return op1 > op2

    ### Unary operations ###

    @staticmethod
    def sin(op1, _, sym=False):
        if sym: return sympy.sin(op1)
        return math.sin(op1)

    @staticmethod
    def cos(op1, _, sym=False):
        if sym: return sympy.cos(op1)
        return math.cos(op1)

    @staticmethod
    def square(op1, _, sym=False):
        result = op1**2
        if sym: return result
        if result > 1.0e30:
            return 1.0e10
        else:
            return result

    @staticmethod
    def sqrt(op1, _, sym=False):
        if sym: return sympy.sqrt(op1)
        return math.sqrt(op1)

    @staticmethod
    def exponentiation(op1, _, sym=False):
        result = math.exp(op1)
        if sym: return sympy.exp(op1)
        if result > 1.0e30:
            return 1.0e10
        else:
            return result





