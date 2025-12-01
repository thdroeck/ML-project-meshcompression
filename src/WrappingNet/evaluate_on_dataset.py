from argparse import ArgumentParser
from WrappingNet.wrappingnet.models import get_model


def main(args):
    model = get_model(args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        type=str,
        required=True,
        help="path to the dataset",
    )

    parser.add_argument(
        "--model_checkpoint",
        dest="model_checkpoint",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )

    parser.add_argument(
        "--latent_dim",
        dest="latent_dim",
        type=int,
        required=True,
        help="dimension of the latent space",
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        required=True,
        help="name of the model architecture",
    )

    args = parser.parse_args()

    main(args)
