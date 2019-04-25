# steps:
# gen hyperparam set
# run code
# analize grid new    [can be run afterwards]
# update new best set

# check if thread by subprocess only uses 1cpu / how much memory

import subprocess as sp
import numpy as np
import hyperopt
from hyperopt import hp, fmin, tpe, space_eval

from soft_patterns import *
import pdb


def train_search(train_data,
          dev_data,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          model_file_prefix,
          learning_rate,
          batch_size,
          run_scheduler=False,
          gpu=False,
          clip=None,
          max_len=-1,
          debug=0,
          dropout=0,
          word_dropout=0,
          patience=1000):
    """ Train a model on all the given docs """

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss(None, False)

    enable_gradient_clipping(model, clip)

    if dropout:
        dropout = torch.nn.Dropout(dropout)
    else:
        dropout = None

    debug_print = int(100 / batch_size) + 1

    writer = None

    # if model_save_dir is not None:
    #     writer = SummaryWriter(os.path.join(model_save_dir, "logs"))

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    start_time = monotonic()

    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        i = 0
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu), word_dropout, max_len)
            gold = [x[1] for x in batch]
            loss += torch.sum(
                train_batch(model, batch_obj, num_classes, gold, optimizer, loss_function, gpu, debug, dropout)
            )

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)
            i += 1
        #

        finish_iter_time = monotonic()
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)
        print("Epoch %d: %.4f" % (it,dev_acc))

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print(">>New best acc!")
    return best_dev_acc


def main_search(args):
    print(args)

    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.patterns.split("_")),
                                key=lambda t: t[0]))

    pre_computed_patterns = None

    if args.pre_computed_patterns is not None:
        pre_computed_patterns = read_patterns(args.pre_computed_patterns, pattern_specs)
        pattern_specs = OrderedDict(sorted(pattern_specs.items(), key=lambda t: t[0]))

    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))
    train_vocab = vocab_from_text(args.td)
    print("Train vocab size:", len(train_vocab))
    dev_vocab |= train_vocab

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, _ = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    np.random.shuffle(dev_data)
    num_iterations = args.num_iterations

    train_input, _ = read_docs(args.td, vocab, num_padding_tokens=num_padding_tokens)
    train_labels = read_labels(args.tl)

    num_classes = len(set(train_labels))

    # truncate data (to debug faster)
    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    if n is not None:
        train_data = train_data[:n]
        dev_data = dev_data[:n]

    if args.use_rnn:
        rnn = Rnn(word_dim,
                  args.hidden_dim,
                  cell_type=LSTM,
                  gpu=args.gpu)
    else:
        rnn = None

    semiring = \
        MaxPlusSemiring if args.maxplus else (
            LogSpaceMaxTimesSemiring if args.maxtimes else ProbSemiring
        )

    model = SoftPatternClassifier(pattern_specs,
                                  mlp_hidden_dim,
                                  num_mlp_layers,
                                  num_classes,
                                  embeddings,
                                  vocab,
                                  semiring,
                                  args.bias_scale_param,
                                  args.gpu,
                                  rnn,
                                  pre_computed_patterns,
                                  args.no_sl,
                                  args.shared_sl,
                                  args.no_eps,
                                  args.eps_scale,
                                  args.self_loop_scale)

    if args.gpu:
        model.to_cuda(model)

    model_file_prefix = 'model'
    model_save_dir = args.model_save_dir

    if model_save_dir is not None:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    acc  = train_search(
          train_data,
          dev_data,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          model_file_prefix,
          args.learning_rate,
          args.batch_size,
          args.scheduler,
          args.gpu,
          args.clip,
          args.max_doc_len,
          args.debug,
          args.dropout,
          args.word_dropout,
          args.patience)

    return acc



if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[soft_pattern_arg_parser(), training_arg_parser(), general_arg_parser()])
  sopa_args = parser.parse_args()

  
  # TODO: define ranges / params for hyperopt obj
  # patt spec
  # learning rate
  # dropout
  # mlp_hid_dim
  # batch_size
  
  def objective(opt_params):
    # exp_args = parser.parse_args()
    exp_args = sopa_args
    print(opt_params)

    exp_args.patterns = opt_params["pat"]
    exp_args.learning_rate = opt_params["lr"]
    exp_args.dropout = opt_params["dropout"]
    exp_args.mlp_hidden_dim = int(opt_params["mlp_hid_dim"])
    exp_args.batch_size = int(opt_params["batch_size"])

    return -main_search(exp_args)


  space = {
    'pat': hp.choice('pat', ['6-10_5-10_4-10_3-10_2-10',
                        '6-10_5-10_4-10']),
    'lr': hp.loguniform('lr', -9, -2),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'mlp_hid_dim': hp.quniform('mlp_hid_dim', low=100, high=300,q=10),
    'batch_size': hp.quniform('batch_size', low=10, high=64,q=10)
  }


  # fixed pattern spec
  # pat="6-10_5-10_4-10_3-10_2-10"

  best = fmin(objective, space, algo=tpe.suggest, max_evals=30)
  
  print(best)
  # -> {'a': 1, 'c2': 0.01420615366247227}
  print(space_eval(space, best))

  # main loop
  # for param in ...
  #   # get params
  #   lr = ...
  #   run_obj = sp.run(
  #     ["bash","run_code",
  #     pat, str(lr), str(mlp_hid_dim),
  #     str(dropout), 
  #     "1","2",
  #     str(batch_size),
  #     "0","1","2",
  #     "0","0","1",
  #     "42",
  #     ])
    
    
  # #

  # # analyze grid can be run separately afterwards