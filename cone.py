import math
r=float(input("enter the radius of cone:"))
h=float(input("enter the height of cone:"))
sl=math.sqrt(r**2+h**2)
lsa=math.pi*r*sl
tsa=math.pi*r*(r+sl)
vol=(1/3)*math.pi*r**2*h
print(f"lateral surface area of cone is:{lsa} cm²")
print(f"total surface area of cone is:{tsa} cm²")
print(f"volume of cone is:{vol} cm³")