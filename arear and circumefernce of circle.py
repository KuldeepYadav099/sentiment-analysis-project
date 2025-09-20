pi=3.14
choice=input("'area' or 'circumference':")
r=int(input("enter the radius:"))
if choice=="area":
    print("area of circle:",pi*r*r)
elif choice=="circumference":
    print("circumference of circle:",2*pi*r)
else:
    print("invalid choice")
            