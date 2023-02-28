import argparse

def parse():

  parser = argparse.ArgumentParser(description='CS6910 - Assignment 1 :')

  # String type arguments
  parser.add_argument('-wp','--wandb_project', type=str, default = "myprojectname")
  parser.add_argument('-we','--wandb_entity', type=str, default = "myname")
  parser.add_argument('-d','--dataset', type=str, default = "fashion_mnist")
  parser.add_argument('-l','--loss', type=str, default = "mse")
  parser.add_argument('-o','--optimizer', type=str, default = "sgd")
  parser.add_argument('-w_i','--weight_init', type=str, default = "xavier")
  parser.add_argument('-a','--activation', type=str, default = "relu")
  parser.add_argument('-opt','--output', type=str, default = "softmax")

  # Integer type arguments
  parser.add_argument('-e','--epochs', type=int, default = 10)
  parser.add_argument('-nhl','--num_layers', type=int, default = 2)
  parser.add_argument('-sz','--hidden_size', type=int, default = 64)
  parser.add_argument('-b','--batch_size', type=int, default = 4)

  # Float type arguments
  parser.add_argument('-lr','--learning_rate',type=float, default = 1e-2)
  parser.add_argument('-m','--momentum',type=float, default = 0.6)
  parser.add_argument('-beta','--beta',type=float, default = 0.6)
  parser.add_argument('-beta1','--beta1',type=float, default = 0.5)
  parser.add_argument('-beta2','--beta2',type=float, default = 0.5)
  parser.add_argument('-eps','--epsilon',type=float, default = 0.000001)
  parser.add_argument('-w_d','--weight_decay',type=float, default =.0)

  args = parser.parse_args()

  return args

args = parse()

model_params = {

    "loss": args.loss,
    "weight_init":args.weight_init,
    "num_layers": args.num_layers,
    "hidden_size": args.hidden_size,
    "activation": args.activation,
    "output": args.output

  }

optimizer_params = {

    "optimizer": args.optimizer,
    "learning_rate": args.learning_rate,
    "momentum": args.momentum,
    "beta": args.beta,
    "beta1": args.beta1,
    "beta2": args.beta2,
    "epsilon": 	args.epsilon,
    "weight_decay": args.weight_decay

  }

training_params = {

    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "dataset":args.dataset

  }