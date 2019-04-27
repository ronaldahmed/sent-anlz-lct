# bootstrapping proc
#  batch = 1/2 src, 1/2 tgt pred
#  1 epoch per iteration
#  5k unlabl inst
#  5 boots. iters
#       1000 per iter -> 30 per batch
#                     -> 300 steps


from soft_patterns import *
from collections import Counter
import pdb


def main_bootstrap(args):
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
    train_vocab = vocab_from_text(args.td) | vocab_from_text(args.tud)
    print("Train vocab size:", len(train_vocab))
    dev_vocab |= train_vocab

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, _ = read_docs(args.vd, vocab, num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    np.random.shuffle(dev_data)
    num_boots_iterations = args.boots_iterations
    num_iterations = args.num_iterations # just 1 epoch per boostrap

    train_input, _ = read_docs(args.td, vocab, num_padding_tokens=num_padding_tokens)
    train_labels = read_labels(args.tl)

    unl_train_input, _ = read_docs(args.tud, vocab, num_padding_tokens=num_padding_tokens)
    num_classes = len(set(train_labels))

    print("training src instances:", len(train_input))
    print("training tgt unl instances:", len(unl_train_input))
    print("num_classes:", num_classes)

    # truncate data (to debug faster)
    train_src_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_src_data)

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

    model_file_prefix = 'bst_model'
    # Loading model
    if args.input_model is not None:
        state_dict = torch.load(args.input_model)
        model.load_state_dict(state_dict)
    else:
        print("Specify model to load...")
        return None

    model_save_dir = args.model_save_dir

    if model_save_dir is not None:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    print("Training with", model_file_prefix)

    train_boostrap(
          train_src_data,
          dev_data,
          unl_train_input,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          num_boots_iterations,
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

    return 0


def train_boostrap(
          train_data,
          dev_data,
          unl_train_input,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          num_boots_iterations,
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

    if model_save_dir is not None:
        writer = SummaryWriter(os.path.join(model_save_dir, "logs"))

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    start_time = monotonic()
    np.random.shuffle(unl_train_input)
    n_unl_chunk = len(unl_train_input )+1 // num_boots_iterations

    for it_bst in range(num_boots_iterations):
      unl_train_input_chunk = unl_train_input[:(it_bst+1)*n_unl_chunk]
      unl_train_data = list(zip(unl_train_input, len(unl_train_input)*[0]))
      silver_data = predict_unlabeled(model,unl_train_data,batch_size,gpu,max_len,word_dropout)
      train_boost_data = train_data + silver_data
      print("Added %d inst to training data..." % len(silver_data))
      print("Label dist in silver data:",Counter([x[1] for x in silver_data]).most_common() )

      for it in range(num_iterations):
          np.random.shuffle(train_boost_data)

          loss = 0.0
          i = 0
          for batch in shuffled_chunked_sorted(train_boost_data, batch_size):
              batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu), word_dropout, max_len)
              gold = [x[1] for x in batch]
              loss += torch.sum(
                  train_batch(model, batch_obj, num_classes, gold, optimizer, loss_function, gpu, debug, dropout)
              )

              if i % debug_print == (debug_print - 1):
                  print(".", end="", flush=True)
              i += 1

          if writer is not None:
              for name, param in model.named_parameters():
                  writer.add_scalar("parameter_mean/" + name,
                                    param.data.mean(),
                                    it)
                  writer.add_scalar("parameter_std/" + name, param.data.std(), it)
                  if param.grad is not None:
                      writer.add_scalar("gradient_mean/" + name,
                                        param.grad.data.mean(),
                                        it)
                      writer.add_scalar("gradient_std/" + name,
                                        param.grad.data.std(),
                                        it)

              writer.add_scalar("loss/loss_train", loss, it)

          dev_loss = 0.0
          i = 0
          for batch in chunked_sorted(dev_data, batch_size):
              batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu))
              gold = [x[1] for x in batch]
              dev_loss += torch.sum(compute_loss(model, batch_obj, num_classes, gold, loss_function, gpu, debug).data)

              if i % debug_print == (debug_print - 1):
                  print(".", end="", flush=True)

              i += 1

          if writer is not None:
              writer.add_scalar("loss/loss_dev", dev_loss, it)
          print("\n")

          finish_iter_time = monotonic()
          train_acc = evaluate_accuracy(model, train_data[:1000], batch_size, gpu)
          dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)

          print(
              "iteration: {:>7,} train time: {:>9,.3f}m, eval time: {:>9,.3f}m "
              "train loss: {:>12,.3f} train_acc: {:>8,.3f}% "
              "dev loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                  it,
                  (finish_iter_time - start_time) / 60,
                  (monotonic() - finish_iter_time) / 60,
                  loss / len(train_data),
                  train_acc * 100,
                  dev_loss / len(dev_data),
                  dev_acc * 100
              )
          )

          if dev_loss < best_dev_loss:
              if dev_acc > best_dev_acc:
                  best_dev_acc = dev_acc
                  print("New best acc!")
              print("New best dev!")
              best_dev_loss = dev_loss
              best_dev_loss_index = 0
              if model_save_dir is not None:
                  model_save_file = os.path.join(model_save_dir, "{}_{}_{}.pth".format(model_file_prefix,it_bst,it))
                  print("saving model to", model_save_file)
                  torch.save(model.state_dict(), model_save_file)
          else:
              best_dev_loss_index += 1
              if best_dev_loss_index == patience:
                  print("Reached", patience, "iterations without improving dev loss. Breaking")
                  break

          if dev_acc > best_dev_acc:
              best_dev_acc = dev_acc
              print("New best acc!")
              if model_save_dir is not None:
                  model_save_file = os.path.join(model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                  print("saving model to", model_save_file)
                  torch.save(model.state_dict(), model_save_file)

          if run_scheduler:
              scheduler.step(dev_loss)

    return model


def predict_unlabeled(model,
          data,
          batch_size,
          gpu,
          max_len=-1,
          word_dropout=0):
    n = float(len(data))
    pred = []
    perc = 0.8
    top_n = int(perc * len(data))
    for batch in chunked_sorted(data, batch_size):
        batch_obj = Batch([x for x, y in batch], model.embeddings, to_cuda(gpu), word_dropout, max_len)
        #predicted = model.predict(batch_obj, debug)
        pred_lab_val = model.get_pred_scores(batch_obj)
        pred.extend(list(zip(pred_lab_val,batch)))
    pred.sort(key=lambda x: x[0][1], reverse=True)
    silver_data = [(x[1][0],x[0][0]) for x in pred[:top_n]]
    return silver_data


def boostrapping_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-bi", "--boots_iterations", help="Number of bootstrap iterations", type=int, default=5)
    p.add_argument("--tud", help="unlabeled training data - tgt domain", required=True)
    return p



if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[boostrapping_arg_parser(),soft_pattern_arg_parser(), training_arg_parser(), general_arg_parser()])
    sys.exit(main_bootstrap(parser.parse_args()))
