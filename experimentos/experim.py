from torch import inf

def repetir_experimento(n_reps, experimento, *exp_args, **exp_kwargs):
    best_score = inf

    for _ in range(n_reps):
        score, model = experimento(*exp_args, **exp_kwargs)

        if score < best_score:
            best_score = score
            best_model = model
    
    return best_score, best_model