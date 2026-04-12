class Frac:
    def __init__(self, numerator: int, denominator: int):
        self.numerator = numerator
        self.denominator = denominator
        self.reduce()

    def from_float(ratio: float):
        i = 1
        while not float(ratio * (10**i)).is_integer():
            i += 1        
        return Frac(int(ratio * (10**i)), (10**i))

    def reduce(self):        
        x = self.numerator
        y = self.denominator
        while x != 0 and y != 0:            
            if x > y:
                x = x % y
            else:
                y = y % x  
        self.numerator = int(self.numerator / x if x != 0 else self.numerator / y)
        self.denominator = int(self.denominator / x if x != 0 else self.denominator / y)