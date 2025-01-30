import numpy as np
import statsmodels.api as sm

from utils import filteredModuleData

data = filteredModuleData.getFilteredModuleData()

print(f'There are {np.sum(data["HadMiniLesson_int"])} module attempts that had a mini lesson run on them')
print(f'There are {np.sum(data["HadTargetedIntervention_int"])} module attempts that had a targeted intervention run on them')

both_run = np.sum(data['HadMiniLesson_int'] & data['HadTargetedIntervention_int'])
print(f'There were {both_run} module attempts where both were run at the same time')

model = sm.formula.ols(formula="ReflectionAccuracyRate ~ HadMiniLesson_int + HadTargetedIntervention_int", data=data)
results = model.fit()
print(results.summary())
print("-----------")
print("")

miniLessonCorrelation = data['TestAccuracyRate'].corr(data['HadMiniLesson_int'])
targeted_intervention_correlation = data['TestAccuracyRate'].corr(data['HadTargetedIntervention_int'])
print(f'Mini lesson correlation with test score is: {miniLessonCorrelation}')
print(f'Targeted intervention correlation with test score is: {targeted_intervention_correlation}')

miniLessonCorrelation = data['ReflectionAccuracyRate'].corr(data['HadMiniLesson_int'])
targeted_intervention_correlation = data['ReflectionAccuracyRate'].corr(data['HadTargetedIntervention_int'])
print(f'Mini lesson correlation with reflection score is: {miniLessonCorrelation}')
print(f'Targeted intervention correlation with reflection score is: {targeted_intervention_correlation}')