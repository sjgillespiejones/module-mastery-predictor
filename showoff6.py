from utils import filteredModuleData
import statsmodels.api as sm

data = filteredModuleData.getFilteredModuleData()
print(data.columns)

model = sm.formula.ols(formula="TestAccuracyRate ~ DaysBetweenModuleAndTest + AttemptNumber + EntranceTicketQuestionPassRate + NumberOfAttemptsBeforeExitTicketCorrect + StudentTestPassRate + StudentReflectionPassRate + IsUsingVerificationAsIntended", data=data)
results = model.fit()
print(results.summary())