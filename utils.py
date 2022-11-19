import logging 

def setup_logging(log_file, **kwargs):
    base_kwargs = {"format":'%(asctime)s | %(levelname)s: %(message)s',
                   "level":logging.DEBUG, 
                   "filemode":"w",}
    new_kwargs = {**base_kwargs, **kwargs}
    logging.basicConfig(filename=log_file, **new_kwargs)