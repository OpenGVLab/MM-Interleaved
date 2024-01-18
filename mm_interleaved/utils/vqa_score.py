import os
import json

from .vqav2_metrics_src.vqa import VQA as VQAV2_VQA
from .vqav2_metrics_src.vqaEval import VQAEval as VQAV2_VQAEval
from .vizwiz_metrics_src.vqa import VQA as Vizwiz_VQA
from .vizwiz_metrics_src.vqaEval import VQAEval as Vizwiz_VQAEval

def extract_answer(response):
    response = response.replace('\"', '')
    # response = response.strip().split('.')[0].split(',')[0].split('!')[0].lower()
    response = response.strip().split('\n')[0].split('.')[0].split(',')[0].split('!')[0].lower()

    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]

    if ' or ' in response:
        response = response.split(' or ')[0]
    if ' and ' in response:
        response = response.split(' and ')[0]

    return response.strip()

def vqa_eval(
    question_file,
    annotation_file,
    results_file,
    use_extract_answer=True,
):
    answers = json.load(open(results_file))
    for item in answers:
        answer = item['answer']
        
        if use_extract_answer:
            answer = extract_answer(answer)
    
        item['answer'] = answer
        
    if use_extract_answer:
        with open(results_file.replace('.json', '_processed.json'), 'w') as file:
            json.dump(answers, file)
    
    annotation_file = annotation_file
    question_file = question_file
    vqa = VQAV2_VQA(annotation_file, question_file)
    vqaRes = vqa.loadRes(answers, question_file)
    vqaEval = VQAV2_VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval.evaluate()

    return {'overall_accuracy': vqaEval.accuracy['overall']}

def vizwiz_vqa_eval(
    annotation_file,
    results_file,
    use_extract_answer=True,
):
    answers = json.load(open(results_file))
    for item in answers:
        answer = item['answer']
        
        if use_extract_answer:
            answer = extract_answer(answer)
    
        item['answer'] = answer
    
    if use_extract_answer:
        with open(results_file.replace('.json', '_processed.json'), 'w') as file:
            json.dump(answers, file)
    
    vqa = Vizwiz_VQA(annotation_file)
    vqaRes = Vizwiz_VQA(annotation=answers)
    vqaEval = Vizwiz_VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval.evaluate()

    res = {'overall_accuracy': vqaEval.accuracy['overall']}
    res.update(vqaEval.caption_metric.items())
    return res
