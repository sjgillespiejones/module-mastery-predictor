import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import subplots

from utils import studentProfile

module_data = pd.read_csv('./ModuleMasteryExtract.csv')

fig, axes = subplots(2, 2, figsize=(12,12))
student_profiles = studentProfile.getStudentProfileData(module_data)

trimmed_data = student_profiles[student_profiles["ModuleAttempts"] >= 15]

studentProfile.showTestAndReflectionGraph(trimmed_data, axes[0][0])
studentProfile.showVerificationPassRateGraph(trimmed_data, axes[0][1])
studentProfile.showTestAttemptsGraph(trimmed_data, axes[1][0])
studentProfile.showVerificationAttemptsGraph(trimmed_data, axes[1][1])
studentProfile.printSummaryData(trimmed_data)
plt.show()