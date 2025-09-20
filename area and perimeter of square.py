choice=input("'area'or 'perimeter':")
side=int(input("enter the side of square:"))
if choice=="area":
    print("area of square:",side*side)
elif choice=="perimeter":
    print("perimeter of square:",4*side)
else:
    print("invalid choice")        