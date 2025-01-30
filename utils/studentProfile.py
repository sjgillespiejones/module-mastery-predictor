import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import subplots

def getStudentProfileData(module_attempt_data):
    test_pass_rate = []
    reflection_pass_rate = []
    verification_pass_rate = []
    module_attempts = []
    module_passes = []
    verification_attempts = []
    verification_passes = []
    for group, df in module_attempt_data.groupby("StudentId"):
        test_pass_rate.append(df["StudentTestPassRate"].iloc[0] * 100)
        reflection_pass_rate.append(df["StudentReflectionPassRate"].iloc[0] * 100)
        verification_pass_rate.append(df['StudentVerificationPassRate'].iloc[0] * 100)
        module_attempts.append(df['StudentTotalModuleAttempts'].iloc[0])
        module_passes.append(df['StudentTotalModulePasses'].iloc[0])
        verification_attempts.append(df['StudentVerificationAttempts'].iloc[0])
        verification_passes.append(df['StudentVerificationMastered'].iloc[0])

    return pd.DataFrame({
        'TestPassRate': test_pass_rate,
        'ReflectionPassRate': reflection_pass_rate,
        'VerificationPassRate': verification_pass_rate,
        'ModuleAttempts': module_attempts,
        'ModulePasses': module_passes,
        'VerificationAttempts': verification_attempts,
        'VerificationPasses': verification_passes
    })

def showTestAndReflectionGraph(student_data, axis):
    student_data.hist('TestPassRate', density=False, bins=20, label='Test', alpha=1, ax=axis)
    student_data.hist('ReflectionPassRate', density=False, bins=20, alpha=0.5, label='Reflection', ax=axis)
    axis.set_xlabel('Pass Rate (%)')
    axis.set_ylabel('Number of students')
    axis.legend(loc="upper left")
    axis.set_title('Student Test and Reflection Pass Rates')

def showVerificationPassRateGraph(student_data, axis):
    student_data.hist('VerificationPassRate', density=False, bins=20, label='Mastery Verification', ax=axis,color='green')
    axis.set_xlabel('Pass Rate (%)')
    axis.set_ylabel('Number of students')
    axis.set_title('Mastery Verification Pass Rate')

def showTestAttemptsGraph(student_data, axis):
    student_data.hist('ModuleAttempts', density=False, bins=20, label='Module Attempts', alpha=1, ax=axis)
    student_data.hist('ModulePasses', density=False, bins=20, alpha=0.5, label='Modules Mastered', ax=axis)
    axis.set_xlabel('Number of attempts/mastered')
    axis.set_ylabel('Number of students')
    axis.legend(loc="upper right")
    axis.set_title('Student Module Attempts and Mastery in Test/Reflection')

def showVerificationAttemptsGraph(student_data, axis):
    student_data.hist('VerificationAttempts', density=False, bins=20, label='Module Attempts', alpha=1, ax=axis)
    student_data.hist('VerificationPasses', density=False, bins=20, alpha=0.5, label='Modules Mastered',ax=axis)
    axis.set_xlabel('Number of attempts/mastered')
    axis.set_ylabel('Number of students')
    axis.legend(loc="upper right")
    axis.set_title('Attempts and Mastery in Verification')

def printSummaryData(student_data):
    print(f'There are {len(student_data)} total students')
    print(f'The mean test pass rate is: {np.mean(student_data["TestPassRate"])}%')
    print(f'The median test pass rate is: {np.median(student_data["TestPassRate"])}%')
    print(f'The mean reflection pass rate is: {np.mean(student_data["ReflectionPassRate"])}%')
    print(f'The median reflection pass rate is: {np.median(student_data["ReflectionPassRate"])}%')
    print(f'The median modules attempted per student is: {np.median(student_data["ModuleAttempts"])}')
    print(f'The median modules attempted per student is: {np.median(student_data["ModuleAttempts"])}')