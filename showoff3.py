import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import subplots

from utils import moduleProfile

module_data = pd.read_csv('./ModuleMasteryExtract.csv')

module_groups = module_data.groupby("ModuleId").apply(moduleProfile.moduleGroupFunction)

fig, axes = subplots(2, 2, figsize=(12,12))
moduleProfile.showAttemptsGraph(module_groups, axes[0][0])
moduleProfile.showTestAndReflectionGraph(module_groups, axes[0][1])
moduleProfile.showMiniLessonsGraph(module_groups, axes[1][0])
moduleProfile.showTargetedInterventionGraph(module_groups, axes[1][1])
plt.show()

# Some things to note
# 1. Some modules were always passed on reflection, but none were always failed.
# Taken with the module attempts graph, I think it indicates that there are some outlier modules that aren't attempted very often.
# 2. Test and reflection pass rates seem to follow a similar trend to the student profiles.
# 3. Targeted interventions and mini lessons show a very similar graph.
# It looks like they both have long tails. It would be interesting to look into that further.
# Do teachers like teaching certain modules/key concepts themselves?
# Or are some students getting stuck on the same ones?
# My guess is it looks a bit like the pareto principle.
# 20% of the modules get 80% of the interventions.
