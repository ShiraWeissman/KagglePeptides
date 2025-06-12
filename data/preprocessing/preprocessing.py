from utils.data import *

def add_modification_to_sequence(sequence, modification):
    s = list(sequence)
    modi_dict = {'O': '+', 'A': '-'}
    if isinstance(modification, str):
        m = modification.split('|')
        s[int(m[0]) - 1] = s[int(m[0]) - 1] + modi_dict[m[1][0]]
        if len(m) == 4:
            s[int(m[2]) - 1] = s[int(m[2]) - 1] + modi_dict[m[3][0]]
    return ''.join(s)

def preprocess_data(data):
    data = data.rename(columns={"PeptideSequence": "text", "RetentionTime": "label"})
    data['text'] = data.apply(lambda x: add_modification_to_sequence(x.text, x.Modifications), axis=1)
    data = data.drop(['ID', 'Modifications'], axis=1)
    return data


if __name__=='__main__':
    train_data = load_data(os.path.join(dataset_path,  'train.csv'))
    train_data = preprocess_data(train_data.iloc[:100, :])
    save_data(train_data, os.path.join(dataset_path,  'preprocessed', 'train.csv'), file_type='csv')

    test_data = load_data(os.path.join(dataset_path,  'test.csv'))
    test_data = preprocess_data(test_data.iloc[:100, :])
    save_data(test_data, os.path.join(dataset_path,  'preprocessed', 'test.csv'), file_type='csv')


