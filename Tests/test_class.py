


class Person():

    population_size = 0

    # Init function
    def __init__(self, name, age, height, sex = "M"):
        
        Person.population_size += 1
        self.name = name
        self.height = height
        self.age = age
        self.sex = sex

        Person.ShirtSize(self)
        self.sock_size = Person.SockSize_(height)


    #  Method

    def Tester(self):

        a = self.SockSize(150) 
        b = Person.SockSize_(20) # need to be called on "Person" class cause not static

        print(a,b)

        self.Birthday()

        self.PrintPopulationSize(1)
        self.PrintPopulationSize_(1)

        Person.PrintPopulationSize(3)
        Person.PrintPopulationSize_(Person, 3)



    def Birthday(self):
        self.age += 1
    
    def ShirtSize(self):
        
        if self.height < 150:
            self.shirt_size = "small"
        elif self.height < 180:
            self.shirt_size = "medium"
        else:
            self.shirt_size = "large"

    def PrintPersonInfo(self):
        sex = "male" if self.sex == "M" else "female" 
        print(f"{self.name} is a {self.age} years old {sex} and is {self.height} cm tall with shirt size {self.shirt_size}.")


    # Methods without "self"

    @staticmethod 
    def SockSize(height):

        if height < 75:
            return "38-42"
        else:
            return "42-46"

    def SockSize_(height):

        if height < 75:
            return "38-42"
        else:
            return "42-46"


    # Class methods

    @classmethod # implicitely passes class "Person" as argument
    def PrintPopulationSize(cls, bias):
        print("Population size is =", cls.population_size+bias)

    def PrintPopulationSize_(cls, bias): # need to explicetly pass "Person" class
        print("Population size is =", cls.population_size+bias)




##### Testing #####

person1 = Person("Tom", 13, 175)
person1.Birthday()
person1.Birthday()
person2 = Person("Sarah", 15, 149,"F")
person3 = Person("JigaChad",23, 184)


person1.Tester()


person1.PrintPersonInfo()
person2.PrintPersonInfo()
person3.PrintPersonInfo()


