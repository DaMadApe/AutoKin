import os
import sys
import time
import logging
import inspect

import torch

def setup_logging(exec_call=False):
    filename = inspect.stack()[1+exec_call].filename
    filename = os.path.splitext(os.path.basename(filename))[0]
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    file_dir=f'experimentos/logs/{filename}_{timestamp}.log'

    file_handler = logging.FileHandler(file_dir)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    logger = logging.getLogger('experim')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def ejecutar_experimento(n_reps, experimento, *exp_args,
                         model_save_dir=None, log_all_products=True,
                         anotar=True,
                         **exp_kwargs):

    logger = logging.getLogger('experim')
    if not logger.hasHandlers():
        logger = setup_logging(exec_call=True)

    best_score = torch.inf

    for _ in range(n_reps):
        score, product = experimento(*exp_args, **exp_kwargs)
        if log_all_products:
            logger.info(f'\nProducto: {product}')
        logger.info(f'Puntaje: {score}')

        if score < best_score:
            best_score = score
            best_product = product

    logger.info(f'\nMejor producto: {best_product}')
    logger.info(f'Mejor puntaje = {best_score}')

    if anotar:
        anotaciones = input('Anotaciones:\n')
        logger.debug('\nAnotaciones:\n' + anotaciones)

    if model_save_dir is not None:
        torch.save(product, model_save_dir)