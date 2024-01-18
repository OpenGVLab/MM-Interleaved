import os
import json

from .misc import barrier, get_rank, get_world_size


def collect_vqa_result(result, result_dir, filename, is_vizwiz=False):
    result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json' % filename)

    for item in result:
        image_id = item.pop("image_id")
        answer = item.pop("caption")
        
        if is_vizwiz:
            item['image'] = f'VizWiz_val_{image_id:08d}.jpg'
        else:
            item['question_id'] = image_id
        item['answer'] = answer

    json.dump(result, open(result_file, 'w'))

    barrier()
    if get_rank() == 0: 
        # combine results from all processes
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
            res = json.load(open(result_file, 'r'))
            result += res
            os.remove(result_file)

        json.dump(result, open(final_result_file, 'w'))
        print('result file saved to %s' % final_result_file)

    return final_result_file