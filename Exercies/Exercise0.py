import math

print("=== Exercise 1: Pythagorean theorem ===")
# a) Calculate hypotenuse with a=3, b=4
a = 3
b = 4
c = math.sqrt(a**2 + b**2)
print(f"a) The hypothenuse is {c} length units")

# b) Calculate other cathetus with c=7.0, a=5.0
c = 7.0
a = 5.0
b = math.sqrt(c**2 - a**2)
print(f"b) The other cathetus is {b:.1f} length units")

print("\n=== Exercise 2: Classification accuracy ===")
# Calculate accuracy with 300 correct out of 365 predictions
correct = 300
total = 365
accuracy = correct / total
print(f"The accuracy of the model is {accuracy:.3f}")

print("\n=== Exercise 3: Classification accuracy with confusion matrix ===")
# Calculate accuracy from confusion matrix
TP = 2  # True Positive
FP = 2  # False Positive
FN = 11 # False Negative
TN = 985 # True Negative

accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"The accuracy of this model is {accuracy:.3f}")

# Analysis of the model
print("\nAnalysis:")
print("This model has very high accuracy (98.7%), but let's examine the error types:")
print(f"- {FN} times it failed to detect actual fires (False Negatives)")
print(f"- {FP} times it falsely detected fires (False Positives)")
print("For fire detection, False Negatives are much more dangerous than False Positives!")
print("While the accuracy looks good, the high number of missed fires makes this model potentially unsafe.")

print("\n=== Exercise 4: Line equation ===")
# Calculate slope k and constant m from points A(4,4) and B(0,1)
x1, y1 = 4, 4
x2, y2 = 0, 1

# Calculate slope k
k = (y2 - y1) / (x2 - x1)

# Calculate constant m (using point B where x=0, y=1)
m = y2 - k * x2  # Since when x=0, y=m

print(f"k = {k}, m = {m}, so the equation for the line is y = {k}x + {m}")

print("\n=== Exercise 5: Euclidean distance in 2D ===")
# Calculate distance between P1(3,5) and P2(-2,4)
x1, y1 = 3, 5
x2, y2 = -2, 4

distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
print(f"The distance is {distance:.1f} length units")

print("\n=== Exercise 6: Euclidean distance in 3D ===")
# Calculate distance between P1(2,1,4) and P2(3,1,0)
x1, y1, z1 = 2, 1, 4
x2, y2, z2 = 3, 1, 0

distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
print(f"The distance is {distance:.2f} length units")