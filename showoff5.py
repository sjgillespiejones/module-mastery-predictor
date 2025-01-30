import matplotlib.pyplot as plt

from utils import filteredModuleData

data = filteredModuleData.getFilteredModuleData()

correlation = data.corr()
print('Test Accuracy Rate')
print(correlation['TestAccuracyRate'])

print('')
print('Reflection Accuracy Rate')
print(correlation['ReflectionAccuracyRate'])
print('')

print('Other things to note')
print('')
print('Reflection and test correlation')
print(correlation['StudentTestPassRate']['StudentReflectionPassRate'])
# Can probably only use one of these two in whatever model we choose.

print('')
print('Verification and test correlation')
print(correlation['StudentTestPassRate']['StudentVerificationPassRate'])
# Surprising. They do correlate, but not as much as expected. Can probably treat these as independent
# Is a good indicator for an overall pass rate, but not so much for an individual module

plt.matshow(correlation)
cb = plt.colorbar()
cb.ax.tick_params()
plt.show()