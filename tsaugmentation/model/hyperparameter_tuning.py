import math
import os
from skopt import gp_minimize
from skopt.space import Real, Integer
from tsaugmentation.model.create_dataset_versions_vae import (
    CreateTransformedVersionsVAE,
)
from tsaugmentation.utils.logger import Logger


# Define the search space for hyperparameters
space = [
    Real(0.0001, 0.1, name="learning_rate"),
    Integer(16, 128, name="batch_size"),
    Integer(4, 50, name="patience"),
    Integer(5, 20, name="window_size"),
    # Integer(10, 50, name="mv_normal_dim"),
]


def setup_hyperparameter_opt(dataset_name, freq):
    vae_model = CreateTransformedVersionsVAE(dataset_name=dataset_name, freq=freq)

    def train_evaluate_vae(params):
        (
            learning_rate,
            batch_size,
            patience,
            window_size,
            # mv_normal_dim,
        ) = params

        vae_model.window_size = int(window_size)
        vae_model.n_train = vae_model.n - vae_model.window_size + 1
        vae_model.preprocess_freq()

        # flatten_size = vae_model.get_flatten_size_encoder()
        # Update mv_normal_dim to be the GCD of the flattened tensor size and mv_normal_dim
        # mv_normal_dim = math.gcd(flatten_size, mv_normal_dim)

        model, _, es = vae_model.fit(
            # mv_normal_dim=mv_normal_dim,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # preds, z = vae_model.predict(vae=model)

        # dec_pred_hat = vae_model.generate_transformed_time_series(
        #    model, z, 0
        #)

        return es.best

    return train_evaluate_vae


def log_and_store_result(res, logger):
    logger.info(f"Current parameters: {res.x}, current loss: {res.fun}")


def optimize_hyperparameters(dataset_name, freq, n_calls):
    logger = Logger(
        "hyperparameter_tuning", dataset=f"{dataset_name}_hypertuning", to_file=True
    )
    train_evaluate_vae_with_vae_model = setup_hyperparameter_opt(dataset_name, freq)

    result = gp_minimize(
        func=train_evaluate_vae_with_vae_model,
        dimensions=space,
        n_calls=n_calls,
        n_random_starts=5,
        random_state=42,
        verbose=True,
        callback=[lambda res: log_and_store_result(res, logger)],
    )

    output_dir = "hyperparameter_tuning"
    os.makedirs(output_dir, exist_ok=True)
    best_params = dict(zip([d.name for d in space], result.x))
    best_loss = result.fun
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best loss: {best_loss}")

    with open(os.path.join(output_dir, f"best_params_and_loss_{dataset_name}.txt"), "w") as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best loss: {best_loss}")

    return result
