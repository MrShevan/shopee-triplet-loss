import os
import random
import argparse
import pandas as pd


def union_labels_by_hash(dataframe: pd.DataFrame):
    def split_unique(x):
        new = []
        for arr in list(x):
            new.extend(list(arr))
        return set(new)

    def dfs(node: str, graph: dict, visited: dict):
        visited[node] = True
        result = [node]

        for linked_node in graph[node]:
            if not visited[linked_node]:
                nodes = dfs(linked_node, graph, visited)
                result.extend(nodes)

        return result

    def get_labels(graph: dict):
        visited = dict(zip(graph.keys(), [False] * len(graph)))
        labels_to_hashes = dict()
        curr_label = 1

        for node in graph.keys():
            if not visited[node]:
                nodes = dfs(node, graph, visited)

                labels_to_hashes[curr_label] = nodes
                curr_label += 1

        return labels_to_hashes

    tmp = dataframe.groupby('label_group').image_phash.agg('unique').to_dict()
    dataframe['label_to_hashes'] = dataframe.label_group.map(tmp)

    graph = dataframe.groupby('image_phash').label_to_hashes.agg(
        lambda x: split_unique(x)
    ).to_dict()

    graph = {k: v - set([k]) for k, v in graph.items()}

    result = get_labels(graph)

    hash_to_label = {}

    for k, v in result.items():
        for h in v:
            hash_to_label[h] = k

    dataframe['label_group'] = dataframe.image_phash.map(hash_to_label)

    return dataframe


def prepare_dataframe(dataframe: pd.DataFrame, mode: str):
    """
    Read dataset, make target columns with all images in the same `label_group`, make
    `filepath` column for every row.

    Args:
        dataframe:
        mode:
    """
    # remove little groups from train dataset
    # dataframe = union_labels_by_hash(dataframe)

    tmp = dataframe.groupby('label_group').posting_id.agg('unique').to_dict()
    dataframe['target_labels'] = dataframe.label_group.map(tmp)
    dataframe['target_len'] = dataframe['target_labels'].apply(lambda x: len(x))

    dataframe = dataframe[dataframe.target_len >= 4]

    dataframe = dataframe.reset_index()
    del dataframe['index']


    return dataframe


def split_by_label_group(
    dataframe: pd.DataFrame,
    train_size: float,
    random_state: int = 0
):
    """
    Split dataframe in train and test parts by unique label_group

    Args:
        dataframe: pandas dataframe with `label_group` column
        train_size:
        random_state: fix state
    """
    if 'label_group' not in dataframe.columns:
        Exception('Datarame columns not consists `label_group` column.')

    labels = list(set(dataframe.label_group))

    random.seed(random_state)
    random.shuffle(labels)

    train_labels = labels[:int(train_size * len(labels))]

    train = dataframe[dataframe.label_group.isin(train_labels)].reset_index()
    test = dataframe[~dataframe.label_group.isin(train_labels)].reset_index()

    del train['index']
    del test['index']

    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print('DataFrame loaded!')

    print('Splitting with train_size...')
    train, val = split_by_label_group(
        dataframe=df,
        train_size=(1 - args.val_size),
        random_state=args.random_state
    )

    train = prepare_dataframe(train, mode='train')
    val = prepare_dataframe(val, mode='val')

    train, test = split_by_label_group(
        dataframe=train,
        train_size=(1 - args.test_size),
        random_state=args.random_state
    )

    filepath = args.csv_path.split('/')

    train_filepath = os.path.join('/'.join(filepath[:-1]), 'new_train.csv')
    print(f'Saved train file: {train_filepath}, shape: {train.shape}')
    train.to_csv(train_filepath, header=True, index=False)

    test = prepare_dataframe(test, mode='train')
    test_filepath = os.path.join('/'.join(filepath[:-1]), 'new_test.csv')
    print(f'Saved test file: {test_filepath}, shape: {test.shape}')
    test.to_csv(test_filepath, header=True, index=False)

    # val = prepare_dataframe(val, mode='val')
    val_filepath = os.path.join('/'.join(filepath[:-1]), 'new_val.csv')
    print(f'Saved val file: {val_filepath}, shape: {val.shape}')
    val.to_csv(val_filepath, header=True, index=False)
