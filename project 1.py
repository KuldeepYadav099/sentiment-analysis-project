choice=input("choose 'area' or 'perimeter':")
len=int(input("enter the length:"))
bre=int(input("enter the breadth:"))
if choice=="area":
    print("area of rectangle:",len*bre)
elif choice=="perimeter":
    print("perimeter of rectangle:",2*(len+bre))
else:
    print("invalid choice")        

