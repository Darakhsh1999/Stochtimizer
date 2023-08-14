import types
import numpy as np

def error_check(args, algorithm):
    """ Checks the input arguments for error """

    if algorithm == "PSO":

        if not isinstance(args["object_fn"], types.LambdaType) or not args["object_fn"].__name__ == "<lambda>":
            raise ValueError("object_fn variable must be a lambda function")
        elif not isinstance(args["N"], int) or args["N"] <= 0:
            raise ValueError("Population must be a positive integer")
        elif args["x_max"] < args["x_min"]:
            raise ValueError("x_max must be larger than x_min arguments")
        elif not (0 < args["alpha"] < 1.0):
            raise ValueError("Alpha parameter must be positive float in range [0,1]")
        elif args["dt"] <= 0:
            raise ValueError("Time step dt must be positive float")
        elif args["c1"] <= 0 or args["c2"] <= 0:
            raise ValueError("c1 and c2 parameters must be positive floats")
        elif args["beta"] <= 0:
            raise ValueError("Beta parameter must be positive float")
        elif args["W"] <= 0:
            raise ValueError("W parameter must be positive float")
        elif args["W"] < args["W_min"]:
            raise ValueError("W has to be larger than W_min")
        elif not isinstance(args["verbatim"], bool):
            raise ValueError("verbatim has to be non negative integer")
    
    elif algorithm == "GA":

        if not isinstance(args["object_fn"], types.LambdaType) or not args["object_fn"].__name__ == "<lambda>":
            raise ValueError("object_fn variable must be a lambda function")
        elif not isinstance(args["n_chromosomes"], int) or args["n_chromosomes"] <= 0:
            raise ValueError("Population must be a positive integer")
        elif args["encoding"] not in ["real", "binary"]:
            raise ValueError("Encoding must either be real or binary")
        elif not isinstance(args["variable_length"], int) or args["variable_length"] <= 0:
            raise ValueError("variable_length must be a positive integer")
        elif args["range"] <= 0:
            raise ValueError("Range must be positive float")
        elif args["c_mult"] <= 0:
            raise ValueError("c_mult must be positive float")
        elif not (0 <= args["p_c"] <= 1.0):
            raise ValueError("p_c must be in range [0,1]")
        elif not isinstance(args["verbatim"], bool):
            raise ValueError("verbatim has to be a boolean")
        elif not isinstance(args["elitism"], bool):
            raise ValueError("elitism must be a boolean")
        elif not str(args["selector"].__class__).startswith("<class 'fitness_selectors"):
            raise ValueError("Selector must be from fitness_selectors module")


    elif algorithm == "ACO":

        if not isinstance(args["X"], np.ndarray):
            raise ValueError("X must be a numpy array")
        elif args["mode"] not in ["AS", "MMAS"]:
            raise ValueError("Available modes are AS or MMAS")
        elif not isinstance(args["alpha"], float) or args["alpha"] <= 0:
            raise ValueError("alpha must be a positive float")
        elif not isinstance(args["beta"], float) or args["beta"] <= 0:
            raise ValueError("beta must be a positive float")
        elif not isinstance(args["rho"], float) or args["rho"] <= 0:
            raise ValueError("rho must be a positive float")
        elif not isinstance(args["verbatim"], bool):
            raise ValueError("verbatim has to be a boolean")
        if args["N"] is not None:
            if not isinstance(args["N"], int) or args["N"] <= 0:
                raise ValueError("Population must be a positive integer")

    elif algorithm == "LGP":
        pass



# Testing boolean expressions
if __name__ == "__main__":

    q = 0.1

    print(0 < q < 1.0)