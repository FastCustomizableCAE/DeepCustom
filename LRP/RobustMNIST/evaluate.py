from evaluator import Evaluator


'''
Choose a model type to evaluate 
model_type: original/fgsm_robust/pgd_robust
'''

model_type = 'pgd_robust'

Evaluator().evaluate(model_type= model_type)

