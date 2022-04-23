import os
import sys
import time
import logging
import inspect

import torch

def setup_logging():
    filename = inspect.stack()[2].filename
    filename = os.path.splitext(os.path.basename(filename))[0]
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    file_dir=f'experimentos/logs/{filename}_{timestamp}.log'

    file_handler = logging.FileHandler(file_dir)
    stream_handler = logging.StreamHandler(sys.stdout)

    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(format='%(message)s',
                        handlers=[file_handler, stream_handler],
                        level=logging.DEBUG)

    return logging.getLogger()


def ejecutar_experimento(n_reps, experimento, *exp_args,
                         model_save_dir=None, log_all_products=True,
                         **exp_kwargs):

    setup_logging()

    best_score = torch.inf

    for _ in range(n_reps):
        score, product = experimento(*exp_args, **exp_kwargs)
        if log_all_products:
            logging.info(f'Producto: {product}')
        logging.info(f'Puntaje: {score}')

        if score < best_score:
            best_score = score
            best_product = product

    logging.info(f'Mejor producto: {best_product}')
    logging.info(f'Mejor puntaje = {best_score}')

    anotaciones = input('Anotaciones:\n')
    logging.debug('Anotaciones:\n' + anotaciones)

    if model_save_dir is not None:
        torch.save(product, model_save_dir)