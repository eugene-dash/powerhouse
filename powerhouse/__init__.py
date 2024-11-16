from enum import Enum

from .powerhouse import Backend, init
backend, ph = init(Backend.AUTO, print_backend = True)
