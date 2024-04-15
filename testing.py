def profit(q,w,f): 
    return 5*q - w*q - f*(q>0)

print(profit(5,1,30))