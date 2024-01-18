import os
import json

from .misc import barrier, get_rank, get_world_size


def collect_caption_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json' % filename)

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

        if remove_duplicate:
            result_new = []
            id_list = set()
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.add(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, 'w'))
        print('result file saved to %s' % final_result_file)

    return final_result_file