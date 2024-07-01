def setup_parser(p):
    p.add_argument("--force_retrain", help="Re-train the SOM instead of using pre-pickled model",
                    action="store_true")

    p.add_argument("-m", "--somsize_m", type=int,
                        help="Width of the SOM map in grid units. If provided, forces retraining of the SOM.")

    p.add_argument("-n", "--somsize_n", type=int,
                        help="Height of the SOM map in grid units. If provided, forces retraining of the SOM.")

    p.add_argument("-lr", "--som_lr", type=float,
                        help="Learning rate for training the SOM. If provided, forces retraining of the SOM.")

    p.add_argument("-iters", "--som_iters", type=int,
                        help="Number of iterations for training. If provided, forces retraining of the SOM.")

    p.add_argument("-epochs", "--som_epochs", type=int,
                        help="Number of epochs for training. If provided, forces retraining of the SOM.")

    p.add_argument("-mode", "--somtrain_mode", choices=["batch", "random"],
                        help="Type of SOM training. Valid options are: batch, random. If provided, forces retraining of the SOM.")

    p.add_argument("-init", "--sominit", choices=["pca", "random"],
                        help="How to initialize the SOM. Valid options are: pca, random. If provided, forces retraining of the SOM.")

    return p