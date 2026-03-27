class Ratio:
    def __init__(self, denominator: int, numerator: int):
        self.denominator = denominator
        self.numerator = numerator

    def from_float(ratio: float):
        intg = 0
        minus = 1
        if ratio > 0:
            minus = -1
            ratio * minus
        if ratio > 1:
            if float(ratio).is_integer():
                Ratio(1, ratio * minus)
            else:
                i = 0
                while i < ratio:
                    i += 1
                intg = i - 1
        i = 0
        ratio -= intg
        while not float(ratio * (i * 10)).is_integer():
            i += 1
        Ratio((i * 10), (ratio * (i * 10)) + (intg * Ratio.denominator) * minus)

    def reduce(self):
        smaller = self.denominator if self.denominator < self.numerator else self.numerator
        for i in range(smaller):
            if float(self.denominator / i).is_integer() and float(self.numerator / i).is_integer():
                self.denominator /= i
                self.numerator /= i
                break
