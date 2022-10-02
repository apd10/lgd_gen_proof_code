from evaluators.list_evaluators import *

class ProgressEvaluator:
    def get(params, datas, device_id):
        if params["name"] == "simple_print":
            evaluator = SimplePrintEvaluator(params["simple_print"], datas, device_id)
        elif params["name"] == "mean_loglikelihood":
            evaluator = SampleMeanLLEvaluator(params["mean_loglikelihood"], train_data, device_id) # change train_data to datas
        else:
            raise NotImplementedError
        return evaluator
