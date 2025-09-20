sub1=int(input("enter the marks of maths here:"))
sub2=int(input("enter the marks of soft skill here:"))
sub3=int(input("enter the marks of pps here:"))
sub4=int(input("enter the marks of evs here:"))
sub5=int(input("enter the marks of mec here:"))
if sub1>100 or sub2>100 or sub3>100 or sub4>100 or sub5>100:
   print("enter marks are invalid")
else:
    t=sub1+sub2+sub3+sub4+sub5
    a=t/5
    tl=(t*100)/500
    print("grand total of your entered marks are:",t)
    print("average marks enterd by you:",a)
    print(f"total percentage gained by you is:{tl}%")      
  