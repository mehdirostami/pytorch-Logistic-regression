
"""
Logistic regression using pytorch. 
"""

import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn.functional as func



#to simulate some dataset, fix the number of observations and inputs
n, p = 100, 5

x = np.random.normal(0., 1., size = (n, p))

true_w = np.random.uniform(.5, 3.5, size = (p, 1))

z = np.dot(x, true_w) + np.random.normal(0., 1., size = (n, 1))/3
t = 2. * np.tanh(z/2.) + 1

#Dichotomizing the continuous variable for logit model.
t = np.where(t > 0., 1., 0.)


x_data = Variable(torch.from_numpy(x).float(), requires_grad = False)
t_data = Variable(torch.from_numpy(t).float(), requires_grad = False)


class logit_model(torch.nn.Module):

	def __init__(self):
		super(logit_model, self).__init__()
		self.linear = torch.nn.Linear(p, 1)

	def forward(self, x):
		return(func.sigmoid(self.linear(x)))#For prediction in logit, we need to transform Xw to [0, 1] interval using sigmoid


my_logit = logit_model()


loss = torch.nn.BCELoss(size_average = False)#note that CrossEntropyLoss is for targets with more than 2 classes.
optimizer = torch.optim.SGD(my_logit.parameters(), lr = .01)

for i in range(50):
	#forward, loss, backward

	y = my_logit(x_data)

	l = loss(y, t_data)#Note the order of arguments. In the function, log is been taken from the predicted values, so order matters.

	optimizer.zero_grad()

	l.backward()

	optimizer.step()


#To check how the observed and predicted values are close. The test data is taken from the training data.
x_test = Variable(torch.from_numpy(x[0, :]).float())
t_test = Variable(torch.from_numpy(t[0]).float())

print(my_logit.forward(x_test))
print(t_test)


#To check how the observed and predicted values are close. The test data is independet of the training data.
xx = np.random.normal(0, 1, size = (1, p))
x_test = Variable(torch.from_numpy(xx).float(), requires_grad = False)


z_test = np.dot(xx, true_w) + np.random.normal(0., 1., size = 1)/3
t_test = 2. * np.tanh(z_test/2.) + 1

t_test = np.where(t_test > 0., 1., 0.)

t_test = Variable(torch.from_numpy(t_test).float(), requires_grad = False)

#Now we use the forward method we defined our model class.
print(my_logit.forward(x_test))
print(t_test)
