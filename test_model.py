import json
import numpy as np
import os
import data_generator
import time
import h5py
import xlwt

head_row = ["Label", "True Positive", "False Positive", "False Negative", "Accuracy", "Precision", "Recall"]


def calculate_accuracy(model_predictions, actual_labels=None):
    model_labels = np.zeros(len(model_predictions))
    for inst_num, softmax in enumerate(model_predictions):
        predicted_class = np.argmax(softmax)
        model_labels[inst_num] = predicted_class

    defence_true = 0
    undefence_true = 0
    test_size = len(actual_labels)

    predicted_wrong_statistic = {}
    true_positive = {}
    false_negative = {}
    false_positive = {}

    for i in range(0, 100):
        predicted_wrong_statistic[i] = 0
        true_positive[i] = 0
        false_positive[i] = 0
        false_negative[i] = 0

    predicted_2_class_statistic = {
        'Undefence_Positive': 0,
        'Undefence_Negative': 0,
        'Defence_Positive': 0,
        'Defence_Negative': 0
    }

    for inst_num, label in enumerate(actual_labels):
        predicted_label = model_labels[inst_num]
        if model_labels[inst_num] == label:
            true_positive[label] += 1
            if label >= 50:
                undefence_true += 1
                predicted_2_class_statistic['Undefence_Positive'] += 1
            elif label < 50:
                predicted_2_class_statistic['Defence_Positive'] += 1
                defence_true += 1
        elif model_labels[inst_num] != label:
            false_negative[label] += 1
            false_positive[predicted_label] += 1
            if int(label) not in predicted_wrong_statistic:
                predicted_wrong_statistic[int(label)] = 0
            predicted_wrong_statistic[int(label)] += 1
            if label >= 50 and predicted_label < 50:
                predicted_2_class_statistic['Undefence_Negative'] += 1
            elif label >= 50 and predicted_label >= 50:
                predicted_2_class_statistic['Undefence_Positive'] += 1
            elif label < 50 and predicted_label >= 50:
                predicted_2_class_statistic['Defence_Negative'] += 1
            elif label < 50 and predicted_label < 50:
                predicted_2_class_statistic['Defence_Positive'] += 1

    defence_tpr = defence_true / (test_size / 2) * 100
    undefence_tpr = undefence_true / (test_size / 2) * 100
    total_tpr = (defence_tpr + undefence_tpr) / 2
    return '%.2f' % defence_tpr + '%', '%.2f' % undefence_tpr + '%', '%.2f' % total_tpr + '%', model_labels, \
           predicted_wrong_statistic, predicted_2_class_statistic, true_positive, false_positive, false_negative


def log_setting(setting, predictions, results, actual_labels=None):
    print(setting + '-world results')
    actual_labels = np.argmax(actual_labels, axis=1)
    bookWrite = xlwt.Workbook()
    for sub_model_name, softmax in predictions.items():
        defence_tpr, undefence_tpr, total_tpr, model_labels, \
        predicted_wrong_statistic, predicted_2_class_statistic, \
        true_positive, false_positive, false_negative = calculate_accuracy(softmax, actual_labels)
        results["%s_defence_acc" % sub_model_name] = defence_tpr
        results["%s_undefence_acc" % sub_model_name] = undefence_tpr
        results["%s_total_acc" % sub_model_name] = total_tpr
        with open(statistic_dir + 'predicted_wrong_statistic_%s.json' % sub_model_name, 'w') as f:
            json.dump(predicted_wrong_statistic, f, sort_keys=True, indent=4)
        with open(statistic_dir + 'predicted_confusion_matrix_%s.json' % sub_model_name, 'w') as f:
            json.dump(predicted_2_class_statistic, f, sort_keys=True, indent=4)
        with open(statistic_dir + 'true_positive_%s.json' % sub_model_name, 'w') as f:
            json.dump(true_positive, f, sort_keys=True, indent=4)
        with open(statistic_dir + 'false_positive_%s.json' % sub_model_name, 'w') as f:
            json.dump(false_positive, f, sort_keys=True, indent=4)
        with open(statistic_dir + 'false_negative_%s.json' % sub_model_name, 'w') as f:
            json.dump(false_negative, f, sort_keys=True, indent=4)
        with open(statistic_dir + 'predicted_labels_%s.txt' % sub_model_name, 'w') as f:
            for i in model_labels:
                f.write(str(int(i)) + '\n')
        sheet = bookWrite.add_sheet("%s" % sub_model_name)
        for i in range(len(head_row)):
            sheet.write(0, i, head_row[i])
        for i in range(0, 100):
            line_to_write = [i]
            line_to_write.extend([true_positive[i], false_positive[i], false_negative[i]])
            for j in range(4):
                sheet.write(i + 1, j, line_to_write[j])
    bookWrite.save(statistic_dir + "Statistics.xls")


def predict(config, model, mixture_num, sub_model_name):
    """Compute and save final predictions on test set."""
    print('generating predictions for %s %s model'
          % (model_name, sub_model_name))

    if model_name == 'var-cnn':
        model.load_weights('model_weights_{}.h5'.format(mixture_num))

    test_size = num_mon_sites * num_mon_inst_test + num_unmon_sites_test
    test_steps = test_size // batch_size

    with h5py.File('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                         num_mon_inst, num_unmon_sites_train,
                                         num_unmon_sites_test), 'r') as f:
        test_labels = f['test_data/labels'][:]
        dir_seqs = f['test_data/dir_seq'][:]
        time_seqs = f['test_data/time_seq'][:]
        metadata = f['test_data/metadata'][:]
        assert len(test_labels) == test_size and len(dir_seqs) == test_size \
               and len(time_seqs) == test_size and len(metadata) == test_size
    test_time_start = time.time()
    predictions = model.predict(
        data_generator.generate(config, 'test_data', mixture_num),
        steps=test_steps if test_size % batch_size == 0 else test_steps + 1,
        verbose=0)
    test_time_end = time.time()
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    np.save(file='%s%s_model' % (predictions_dir, sub_model_name),
            arr=predictions)

    print('Total test time: %f' % (test_time_end - test_time_start))


def get_accuracy():
    test_size = num_mon_sites * num_mon_inst_test + num_unmon_sites_test

    with h5py.File('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                         num_mon_inst, num_unmon_sites_train,
                                         num_unmon_sites_test), 'r') as f:
        test_labels = f['test_data/labels'][:]
        dir_seqs = f['test_data/dir_seq'][:]
        time_seqs = f['test_data/time_seq'][:]
        metadata = f['test_data/metadata'][:]
        assert len(test_labels) == test_size and len(dir_seqs) == test_size \
               and len(time_seqs) == test_size and len(metadata) == test_size

    num_test_defence = 0
    num_test_undefence = 0

    for i in range(test_size):
        if np.argmax(test_labels[i]) >= 50:
            num_test_undefence += 1
        elif np.argmax(test_labels[i]) < 50:
            num_test_defence += 1

    assert num_test_undefence == num_test_defence and num_test_undefence == test_size // 2
    print("Num test undefence %d" % num_test_undefence + " Num test defence %d" % num_test_defence)
    predictions = {}
    ensemble_softmax = None

    for inner_comb in mixture:
        sub_model_name = '_'.join(inner_comb)
        softmax = np.load('%s%s_model.npy' % (predictions_dir, sub_model_name))
        if ensemble_softmax is None:
            ensemble_softmax = np.zeros_like(softmax)
        predictions[sub_model_name] = softmax

    for softmax in predictions.values():
        ensemble_softmax += softmax
    ensemble_softmax /= len(predictions)
    if len(predictions) > 1:
        predictions['ensemble'] = ensemble_softmax

    results = {}
    actual_labels = np.argmax(test_labels, axis=1)
    log_setting('closed', predictions, results, test_labels)
    with open('actual_labels.txt', 'w') as f:
        for i in actual_labels:
            f.write(str(int(i)) + '\n')
    with open('job_result_defence_undefence_new.json', 'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)


with open('config.json') as config_file:
    config = json.load(config_file)

num_mon_sites = config['num_mon_sites']
num_mon_inst_test = config['num_mon_inst_test']
num_mon_inst_train = config['num_mon_inst_train']
num_mon_inst = num_mon_inst_test + num_mon_inst_train
num_unmon_sites_test = config['num_unmon_sites_test']
num_unmon_sites_train = config['num_unmon_sites_train']
num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

data_dir = config['data_dir']
model_name = config['model_name']
mixture = config['mixture']
batch_size = config['batch_size']
predictions_dir = config['predictions_dir']
seq_length = config['seq_length']

statistic_dir = 'statistic\\'

if not os.path.exists(statistic_dir):
    os.mkdir(statistic_dir)
get_accuracy()
