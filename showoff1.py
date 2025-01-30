import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import subplots

from utils import studentProfile

module_data = pd.read_csv('./ModuleMasteryExtract.csv')
print(f'There are {len(module_data)} total attempted modules')

fig, axes = subplots(2, 2, figsize=(12,12))
student_profiles = studentProfile.getStudentProfileData(module_data)

studentProfile.showTestAndReflectionGraph(student_profiles, axes[0][0])
studentProfile.showVerificationPassRateGraph(student_profiles, axes[0][1])
studentProfile.showTestAttemptsGraph(student_profiles, axes[1][0])
studentProfile.showVerificationAttemptsGraph(student_profiles, axes[1][1])
studentProfile.printSummaryData(student_profiles)
plt.show()


print(f'There are {len((student_profiles[student_profiles["ModuleAttempts"] < 15]))} students who have attempted fewer than 15 modules')
serious_mastery_verification_students = student_profiles[student_profiles["VerificationPassRate"] >= 75]
print(f'There are {len((serious_mastery_verification_students))} students have a mastery verification score. This is works out to be {(len(serious_mastery_verification_students) / len(student_profiles)) * 100}% of students')

# Ideas to validate
# 1. Remove students with not very many modules.
# Using 15 as a threshold of "have you done 3 tests yet after being mostly diagnosed?"
# 2. Instead of using a verification pass rate in analysis,
# use whether they're taking the feature seriously or not.