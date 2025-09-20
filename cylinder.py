import math

r = float(input("Enter the Radius of the Cylinder: "))
h = float(input("Enter the Height of the Cylinder: "))
lsa = 2 * math.pi * r * h
tsa = 2 * math.pi * r * (r + h)
vol = math.pi * r**2 * h
print(f"Lateral Surface Area of the Cylinder:{lsa} cm²")
print(f"Total Surface Area of the Cylinder:{tsa} cm²")
print(f"Volume of the Cylinder:{vol} cm³")
