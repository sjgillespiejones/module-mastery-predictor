# Module mastery predictor

A repository to run a few machine learning experiements on a dataset from Maths Pathway. There were two goals with this project.
1. To work out if there was a correlation between teachers running mini lessons and doing targeted interventions and whether a student passes a module.
2. To create a predictive model for whether a student will pass a module with a number of inputs.

This repository is laid out in the format of "showoff1.py" to "showoff24.py". They are meant to run sequentially, as it was a presentation given to my team.

All predictive models use a custom scorer that heavily rewards true negatives and heavily punishes false negatives. The intent was to create a model that minimises intervening with the wrong students rather than purely going for the highest accuracy.
