list1 = [1, 3, 5, 7, 8, 12]


max = list1[0]
for ele in list1:
    if ele > max:
        max = ele

min = list1[0]
for elem in list1:
    if elem < min:
        min = ele
        
sum = 0
for element in list1:
    sum += element
    
totallenght = len(list1)
average = sum/totallenght
        
print("Minimum Number:", min)
print("Mxamium Number:", max)
print("Total Sum:", sum)
print("Total Average:", average)
        