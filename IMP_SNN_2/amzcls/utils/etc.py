import torch


def get_entropy(x):
    import torch, operator, math
    from collections import Counter

    def pattern_index(np_array):
        return int(''.join(np_array.astype(str)), 2)

    def pattern_index_torch(x: torch.Tensor):
        z = torch.transpose(torch.flatten(x, 1), 0, 1).detach().long().cpu().numpy()
        return [pattern_index(item) for item in z]

    def entropy(counter_list):
        r = 0.
        total_element = sum([v for (k, v) in counter_list])
        for (k, v) in counter_list:
            v = - (v / total_element) * math.log2((v / total_element))
            r += v
        return counter_list, r

    # counter = Counter(pattern_index_torch(x))
    counter = Counter(pattern_index_torch(x[:, :, :, :, x.shape[-1] // 2]))
    sorted_counter = sorted(counter.items(), key=operator.itemgetter(0), reverse=False)
    sorted_counter, r = entropy(sorted_counter)
    return sorted_counter, r


if __name__ == '__main__':
    inp = torch.randint(0, 2, size=(4, 1024))
    sorted_counter, r = get_entropy(inp)
    print(r)
    print(sorted_counter)
