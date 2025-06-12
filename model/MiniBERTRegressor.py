import os
import sys

sys.path.append(os.path.abspath(os.curdir))
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
from utils.huggingface import *
from utils.schemes import *
from utils.general import *
from utils.metric import calculate_regression_metric


def train_tokenizer(data, scheme):
    if scheme['pretrained_tokenizer'] == None:
        tokenizer = AutoTokenizer.from_pretrained("arindamatcalgm/w266_model4_BERT_AutoModelForSequenceClassification")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(saved_models_path, scheme['pretrained_tokenizer']))
    tokenizer = tokenizer.train_new_from_iterator(data['text'], scheme["vocab_size"])
    return tokenizer


def build_model(scheme):
    if scheme["pretrained_model"] == None:
        configuration = BertConfig(num_labels=1,
                                   vocab_size=scheme["vocab_size"],
                                   hidden_size=scheme["hidden_size"],
                                   num_hidden_layers=scheme["num_hidden_layers"],
                                   num_attention_heads=scheme["num_attention_heads"],
                                   intermediate_size=scheme["intermediate_size"],
                                   hidden_act=scheme["hidden_act"],
                                   hidden_dropout_prob=scheme["hidden_dropout_prob"],
                                   attention_probs_dropout_prob=scheme["attention_probs_dropout_prob"],
                                   max_position_embeddings=scheme["max_position_embeddings"],
                                   initializer_range=scheme["initializer_range"],
                                   layer_norm_eps=scheme["layer_norm_eps"])
        model = BertForSequenceClassification(configuration)
    else:
        model = BertForSequenceClassification.from_pretrained(os.path.join(saved_models_path,
                                                                           scheme["pretrained_model"]))
    return model


def train_model(model, data, scheme):
    print("Preparing for training..")
    train_dataset, valid_dataset = data['train'], data['valid']
    training_args = TrainingArguments(
        output_dir=scheme["output_dir"],
        eval_strategy=scheme["evaluation_strategy"],
        save_strategy=scheme["save_strategy"],
        learning_rate=scheme["learning_rate"],
        per_device_train_batch_size=scheme["per_device_train_batch_size"],
        per_device_eval_batch_size=scheme["per_device_train_batch_size"],
        num_train_epochs=scheme["num_train_epochs"],
        weight_decay=scheme["weight_decay"],
        logging_dir=scheme["logging_dir"],
        logging_steps=scheme["logging_steps"],
        metric_for_best_model=scheme["metric_for_best_model"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=calculate_regression_metric
    )

    print("Training the model..")
    trainer.train()
    return trainer


def get_predictions(trainer, data):
    print("Collecting predictions..")
    train_pred = trainer.predict(test_dataset=data['train'])
    valid_pred = trainer.predict(test_dataset=data['valid'])
    test_pred = trainer.predict(test_dataset=data['test'])
    train_metric = calculate_regression_metric((train_pred.predictions, data['train']['label']))
    valid_metric = calculate_regression_metric((valid_pred.predictions, data['valid']['label']))
    predictions = {'y_train': data['train']['label'], 'y_valid': data['valid']['label'], 'y_test': None,
                   'y_train_pred': train_pred.predictions, 'y_valid_pred': valid_pred.predictions,
                   'y_test_pred': test_pred.predictions, 'train_metric': train_metric, 'valid_metric': valid_metric
                   }
    predictions['y_train'] = predictions['y_train'].detach().tolist()
    predictions['y_train_pred'] = predictions['y_train_pred'].reshape(-1).tolist()
    predictions['y_valid'] = predictions['y_valid'].detach().tolist()
    predictions['y_valid_pred'] = predictions['y_valid_pred'].reshape(-1).tolist()
    predictions['y_test_pred'] = predictions['y_test_pred'].reshape(-1).tolist()
    print("train_metric:", train_metric)
    print("valid_metric:", valid_metric)
    return predictions


def run_model_scheme(scheme, data):
    print("Building model..")
    model = build_model(scheme['model'])
    trainer = train_model(model, data, scheme['model'])
    print('Evaluating model..')
    model_eval = trainer.evaluate()
    if model_eval['eval_mse'] <= scheme['model']['loss_threshold']:
        print("Validation MSE below required threshold..")
        for k in ['eval_mse', 'eval_mae', 'eval_r2']:
            scheme['model'][k] = model_eval[k]
        print("Saving scheme to best schemes folder..")
        save_scheme(scheme, os.path.join(best_scheme_path, scheme['data']["dataset"],
                                         f'{scheme["model"]["model_name"]}_{scheme["model"]["run_name"]}.yaml'))
    if scheme['model']['save_predictions']:
        predictions = get_predictions(trainer, data)
        save_predictions(predictions,
                         os.path.join(predictions_path, f'MiniBERTRegressor_{scheme["model"]["run_name"]}.json'))
    if scheme['model']['save_model']:
        save_trained_model(model, f'model_{scheme["model"]["run_name"]}')
    del trainer
    del model


def prepare_data(data_scheme):
    data = load_dataset(data_scheme['data_path'],
                        data_files={'train': 'train.csv', 'test': 'test.csv'})
    print("Train-Validation splitting..")
    train_valid_data = data["train"].train_test_split(test_size=data_scheme['train_valid_ratio'],
                                                      seed=data_scheme['seed'])
    train_data, valid_data = train_valid_data['train'], train_valid_data['test']
    if data_scheme['train_tokenizer'] == True:
        print("Training tokenizer..")
        tokenizer = train_tokenizer(train_data, data_scheme)
        if data_scheme['save_tokenizer'] == True:
            print("Saving tokenizer..")
            save_trained_tokenizer(tokenizer, tokenizer_name=f'tokenizer_{data_scheme["run_name"]}')
    else:
        print("Loading tokenizer..")
        tokenizer = load_pretrained_tokenizer(data_scheme['pretrained_tokenizer'])
    print("Tokenizing data..")
    train_data = tokenize_dataset(train_data, tokenizer)
    valid_data = tokenize_dataset(valid_data, tokenizer)
    test_data = tokenize_dataset(data['test'], tokenizer)
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    valid_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    del tokenizer
    return {'train': train_data,
            'valid': valid_data,
            'test': test_data}


def main():
    parser = argparse.ArgumentParser(description="Load scheme from YAML file.")
    defualt_dataset_name = 'peptides'
    default_schemes_files_names = load_schemes_files_names(dataset_name=defualt_dataset_name)

    parser.add_argument("dataset_name", type=str, default=defualt_dataset_name,
                        help="Name of dataset.")
    parser.add_argument("schemes_filenames", nargs='*', default=default_schemes_files_names,
                        help="Files names of YAML schemes files.")
    args = parser.parse_args()
    with open(os.path.join(schemes_folder_path, args.dataset_name, args.schemes_filenames[0]), 'r') as f:
        first_scheme = yaml.safe_load(f)
    data_scheme = first_scheme['data']
    data = prepare_data(data_scheme)
    for i, scheme_filename in enumerate(args.schemes_filenames):
        try:
            with open(os.path.join(schemes_folder_path, args.dataset_name, scheme_filename), 'r') as f:
                scheme = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: File '{scheme_filename}' not found.")
            return
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in '{scheme_filename}': {e}")
            return
        print(f'\nScheme: {scheme["model"]["run_name"]}, {i} out of {len(args.schemes_filenames)}')
        if not are_dicts_identical(data_scheme, scheme['data']):
            data_scheme = scheme['data']
            data = prepare_data(data_scheme)
        run_model_scheme(scheme, data)
        os.remove(os.path.join(schemes_folder_path, args.dataset_name, scheme_filename))


if __name__ == '__main__':
    main()
