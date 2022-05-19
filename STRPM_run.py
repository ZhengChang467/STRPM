import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
import pynvml

pynvml.nvmlInit()
configs = argparse.ArgumentParser(description='STRPM')
configs.add_argument('--dataset', type=str, default='ucfsport')
configs = configs.parse_args()
configs.tied = True
args = configs
if configs.dataset == 'ucfsport':
    from configs.ucfsport_configs import parser
elif configs.dataset == 'human36m':
    from configs.human36m_configs import parser
elif configs.dataset == 'sjtu4k':
    from configs.sjtu4k_configs import parser

parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
args.tied = True

def schedule_sampling(eta, itr, batch_size):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      # args.img_time,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_wrapper(model):
    begin = 0
    # pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model_pm and args.pretrained_model_d:
        model.load(args.pretrained_model_pm, args.pretrained_model_d)
        begin = int(args.pretrained_model_pm.split('-')[-1])

    # load data
    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:
            if itr > args.max_iterations:
                break
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr, batch_size)
            if itr % args.test_interval == 0 and itr > 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)
            trainer.train(model, ims, real_input_flag, args, itr)
            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)
            itr += 1
            meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))


def test_wrapper(model):
    model.load(args.pretrained_model_pm, args.pretrained_model_d)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)


if __name__ == '__main__':
    print('Initializing models')
    model = Model(args)
    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        test_wrapper(model)
