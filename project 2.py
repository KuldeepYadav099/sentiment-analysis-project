tba=float(input("enter the total bill amount(before adding tip):"))
tip=float(input("enter the tip percentage they wish to give:"))
ta=tba*tip/100
fb=tba+ta
ef=fb/3
print("the original bill amount is:",tba)
print("the tip amount is:",ta)
print("final bill amount is:",fb)
print("the amount each person should pay is:",ef)
