def flatquant_fwrd(*args, **kwargs):
    from .train_utils import flatquant_fwrd as run

    return run(*args, **kwargs)


def save_hif4_flatquant_model(*args, **kwargs):
    from .flat_utils import save_hif4_flatquant_model as save

    return save(*args, **kwargs)


__all__ = ["flatquant_fwrd", "save_hif4_flatquant_model"]
