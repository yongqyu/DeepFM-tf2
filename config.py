import argparse

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=0.01, help='learning rate', type=float)
    parser.add_argument('--train_batch_size', default=128, help='batch size', type=int)
    parser.add_argument('--test_batch_size', default=512, help='batch size', type=int)
    parser.add_argument('--epochs', default=50, help='number of epochs', type=int)
    parser.add_argument('--print_step', default=100, help='step size for print log', type=int)
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate', type=float)

    parser.add_argument('--dataset_dir', default='/data/private/Ad/ml-20m/np_prepro/', help='dataset path')
    parser.add_argument('--model_path', default='./models/', help='model load path', type=str)
    parser.add_argument('--log_path', default='./logs/', help='log path fot tensorboard', type=str)
    parser.add_argument('--is_reuse', default=False)
    parser.add_argument('--multi_gpu', default=False)

    parser.add_argument('--sparse_emb_dim', default=8, help='dimension for sparse feature', type=int)

    parser.add_argument('--dnn_layers', default=[256,128], type=int)

    args = parser.parse_args()

    return args
