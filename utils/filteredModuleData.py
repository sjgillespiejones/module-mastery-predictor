import pandas as pd

pd.options.mode.chained_assignment = None


def getFilteredModuleData():
    module_data = pd.read_csv('./ModuleMasteryExtract.csv')
    filtered_module_data = module_data[module_data['StudentTotalModuleAttempts'] > 15]
    filtered_module_data['IsUsingVerificationAsIntended'] = filtered_module_data['StudentVerificationPassRate'].apply(lambda x: 1 if x >= 0.75 else 0)
    filtered_module_data['HadMiniLesson_int'] = filtered_module_data['HadMiniLesson'].astype(int)
    filtered_module_data['HadTargetedIntervention_int'] = filtered_module_data['HadTargetedIntervention'].astype(int)
    filtered_module_data = filtered_module_data.drop(['ModuleId', 'StudentId', 'HadMiniLesson', 'HadTargetedIntervention'], axis=1)
    return filtered_module_data


def getTestClassificationData():
    filtered_module_data = getFilteredModuleData()
    filtered_module_data['PassedTest'] = (filtered_module_data['TestAccuracyRate'] == 100).astype(int)
    return filtered_module_data

def getPassedTestModelData():
    classification_data = getTestClassificationData()

    y = classification_data['PassedTest']
    X = classification_data.drop(
        ['TestAccuracyRate', 'ReflectionAccuracyRate', 'StudentTotalModuleAttempts', 'StudentTotalModulePasses',
         'StudentVerificationMastered', 'StudentVerificationPassRate', 'HadMiniLesson_int',
         'HadTargetedIntervention_int', 'PassedTest'], axis=1)

    return X, y