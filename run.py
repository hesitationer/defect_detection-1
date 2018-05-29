import argparse
import importlib
from models import util as utils

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="ld_covnet_rnn_augm", help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--mode', type=str, default="train", help='Can be train or predict')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='no commment')
parser.add_argument('--save_every', type=int, default=50000, help='save state every x iteration')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--no-batch_norm', dest="batch_norm", action='store_false', help='batch normalization')
parser.add_argument('--png_folder', type=str, default="./data/", help='The folder where images are placed')
parser.add_argument('--train_file', type=str, default="./trainEqual.csv", help='The train csv file')
parser.add_argument('--val_file', type=str, default="./valEqual.csv", help='The validation csv file')

parser.set_defaults(batch_norm=False)

args = parser.parse_args()
args_dict = dict(args._get_kwargs())

print args_dict

train_list = []

with open(args.train_file, "r") as train_listfile:
        train_list = utils.read_csv(train_listfile)

val_list = []

with open(args.val_file, "r") as val_listfile:
        val_list = utils.read_csv(val_listfile)

print "==> %d training examples" % len(train_list)
print "==> %d validation examples" % len(val_list)


args_dict = dict(args._get_kwargs())

args_dict['train_list'] = train_list
args_dict['val_list'] = val_list
    
print "==> using network %s" % args.network

network = importlib.import_module("models." + args.network)


network_name = '%s.bs%d%s%s' % (
    network.name(),
    args.batch_size, 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else "")
    
print "==> network_name:", network_name

if args.mode == 'train':
    print "==> training"
    network.do_train(args_dict)
elif args.mode == 'test':
    network.do_evaluate(args_dict)
else:
    raise Exception("unknown mode")
