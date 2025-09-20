p = float(input("Enter Principal amount (P): "))
r = float(input("Enter Rate of interest (R) per annum: "))
t = float(input("Enter Time (T) in years: "))
amount=p*(1+r/100)**t
ci=amount-p
print("compound interest is:",ci)
print("total amount is:",amount)