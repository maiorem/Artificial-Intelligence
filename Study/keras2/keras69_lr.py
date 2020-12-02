weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 10

for iteration in range(1101) :
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2

    print("Error : " + str(error) + "\tPrediction : " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if (down_error < up_error) :
        weight = weight - lr
    if (down_error > up_error) :
        weight = weight + lr

'''
lr = 0.1
Error : 1.232595164407831e-32   Prediction : 0.8000000000000002
Error : 0.0024999999999999935   Prediction : 0.7500000000000001

lr = 0.01
Error : 2.4999999999995603e-05  Prediction : 0.7950000000000005
Error : 1.9721522630525295e-31  Prediction : 0.8000000000000005

lr = 1
Error : 0.0025000000000000044   Prediction : 0.75
Error : 0.20249999999999996     Prediction : 1.25

lr = 10
Error : 19.802500000000002      Prediction : 5.25
Error : 0.30250000000000005     Prediction : 0.25
'''