import numpy as np
import pandas as pd


def moduleGroupFunction(group):
    passed_on_test = np.sum(group['TestAccuracyRate'] == 100)
    passed_on_reflection = np.sum(group['ReflectionAccuracyRate'] == 100)
    return pd.Series({
        'Attempts': len(group),
        'PassedOnTest': passed_on_test,
        'PassedOnReflection': passed_on_reflection,
        'TestAccuracyRate': (passed_on_test / len(group)) * 100,
        'ReflectionAccuracyRate': (passed_on_reflection / len(group)) * 100,
        'MiniLessonCount': np.sum(group['HadMiniLesson']),
        'TargetedInterventionCount': np.sum(group['HadTargetedIntervention'])
    })

def showAttemptsGraph(module_group_data, axis):
    module_group_data.hist('Attempts', density=False, bins=20, label='Attempts', ax=axis)
    axis.set_xlabel('Attempts')
    axis.set_ylabel('Frequency')
    axis.set_title('Module Attempts')

def showTestAndReflectionGraph(module_group_data, axis):
    module_group_data.hist('TestAccuracyRate', density=False, bins=20, label='Test', alpha=1, ax=axis)
    module_group_data.hist('ReflectionAccuracyRate', density=False, bins=20, alpha=0.5, label='Reflection', ax=axis)
    axis.set_xlabel('Pass Rate (%)')
    axis.set_ylabel('Frequency')
    axis.legend(loc="upper left")
    axis.set_title('Test and Reflection Pass Rates')

def showMiniLessonsGraph(module_group_data, axis):
    module_group_data.hist('MiniLessonCount', density=False, bins=20, label='Mini lessons', ax=axis)
    axis.set_xlabel('Mini Lessons')
    axis.set_ylabel('Frequency')
    axis.set_title('Mini Lessons')

def showTargetedInterventionGraph(module_group_data, axis):
    module_group_data.hist('TargetedInterventionCount', density=False, bins=20, label='Targeted Intervention', ax=axis)
    axis.set_xlabel('Targeted Interventions')
    axis.set_ylabel('Frequency')
    axis.set_title('Targeted Interventions')