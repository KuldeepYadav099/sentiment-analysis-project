l=int(input("enter the length of cuboid(in cm):"))
b=int(input("enter the breadth of cuboid(in cm):"))
h=int(input("enter the height of cuboid(in cm):"))
tsa=2*(l*b+b*h+h*l)
perim=4*(l+b+h)
vol=l*b*h
print(f"total surface area of cuboid is:{tsa} cm²")
print(f"perimeter of cuboid is:{perim} cm")
print(f"volume of cuboid is:{vol} cm³")