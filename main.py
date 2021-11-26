from datasketch import MinHash, MinHashLSH
import pandas as pd


def read_file(path):
    data = pd.read_csv(path)
    return data


def hash_generator(dataframe, similarity_threshold, num_perm):
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    for index, row in dataframe.iterrows():
        anchor_hash = get_min_hash(row['Text'], num_perm)
        lsh.insert(row['Id'], anchor_hash)
    return lsh


def get_min_hash(anchor_set, num_perm):
    anchor_hash = MinHash(num_perm=num_perm)
    try:
        if anchor_set:
            phrases = anchor_set.split(',')
            for d in phrases:
                anchor_hash.update(d.encode('utf8'))

    except Exception as e:
        print(e)

    return anchor_hash


def find_similarity(minhashlsh, test_set, num_perm):
    test_minhash = get_min_hash(test_set, num_perm)
    return minhashlsh.query(test_minhash)


if __name__ == '__main__':
    threshold = 0.8
    num_perm = 256

    path_to_anchorPhrase = 'Testset.csv'

    test_str = "AI,and,humans,have,always,been,friendly"

    dataFrame = read_file(path_to_anchorPhrase)
    minhashLsh = hash_generator(dataFrame, threshold, num_perm)
    probability = find_similarity(minhashLsh, test_str, num_perm)
    print(f'Approximate neighbours with Jaccard similarity > {threshold}', probability)
