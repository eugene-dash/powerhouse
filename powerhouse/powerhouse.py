from types import ModuleType
import copy
import numpy as np
from enum import Enum

import numpy as np
try:
  import cupy as cp
except:
  cp = None
try:
  import jax.numpy as jnp
except:
  jnp = None
if jnp is not None:
  jax_over_numpy = True
else:
  jax_over_numpy = False
if jnp is not None and cp is not None:
  jax_over_cupy = True
else:
  jax_over_cupy = False

class Backend(Enum):

  AUTO = 0,

  NUMPY = 2,

  CUPY = 7,

  JAX = 15,

  # JAX is implemented over another numpy api implementation
  # due to JAX providing an incomplete numpy api,
  JAX_OVER_NUMPY = 18,
  JAX_OVER_CUPY = 21,

def init(backend, print_backend = False):
  if print_backend:
    b, p = init(backend, print_backend = False)
    print("Powerhouse :: ", b, ' :: ', p)
    return (b, p)
  if not isinstance(backend, Backend):
    raise Exception("ERROR :: powerhouse.numpy_acc :: Expected Backend param")
  match backend:
    case Backend.AUTO:
      try:
        try:
          return init(Backend.JAX_OVER_CUPY),
        except:
          try:
            return init(Backend.JAX_OVER_NUMPY)
          except:
            raise Exception("INTERNAL")
      except:
        try:
          return init(Backend.CUPY)
        except:
          return init(Backend.NUMPY)
    case Backend.NUMPY:
      return (backend, np)
    case Backend.CUPY:
      if cp is None:
        raise Exception("ERROR :: powerhouse :: No Cupy")
      return (backend, cp)
    case Backend.JAX:
      try:
        return init(Backend.JAX_OVER_CUPY)
      except:
        return init(Backend.JAX_OVER_NUMPY)
    case Backend.JAX_OVER_NUMPY:
      if jax_over_numpy is None:
        raise Exception("ERROR :: powerhouse :: No Jax_over_numpy")
      return (backend, jax_over_numpy)
    case Backend.JAX_OVER_CUPY:
      if jax_over_cupy is None:
        raise Exception("ERROR :: powerhouse :: No Jax_over_numpy")
      return (backend, jax_over_cupy)
    case _:
      raise Exception("ERROR :: powerhouse.init :: backend not recognized")


def jax_over_function(other_api, func):
  def inner(*args, **kwargs):
    result = func(*args, **kwargs)
    if isinstance(result, other_api.ndarray):
      return jnp.asarray(result)
    else:
      return result
  return inner

def jax_over_module(other_api, other_module):
  new_module = other_module
  for name in dir(other_module):
    val = getattr(other_module, name)
    if name[0] == '_':
      continue
    if isinstance(val, ModuleType):
      # Needs a ton of work to make this function recursive
      continue
      #setattr(new_module, name, jax_over_module(other_api, val))
    elif callable(val):
      setattr(new_module, name, jax_over_function(other_api, val))
    elif isinstance(val, other_api.ndarray):
      if not other_api.ma.is_masked(val):
        setattr(new_module, name, jnp.asarray(val))
    else:
      continue
  return new_module

def jaxOver(new_api, other_api):
  for name in dir(jnp):
    if name[0] == '_':
      continue
    setattr(new_api, name, getattr(jnp, name))
  for name in dir(other_api):
    val = getattr(other_api, name)
    if name[0] == '_':
      continue
    if name in new_api.__dict__:
      continue
    elif isinstance(val, ModuleType):
      setattr(new_api, name, jax_over_module(other_api, val))
    elif callable(val):
      setattr(new_api, name, jax_over_function(other_api, val))
    elif isinstance(val, other_api.ndarray):
      if not other_api.ma.is_masked(val):
        setattr(new_api, name, jnp.asarray(val))
    else:
      continue

if jax_over_numpy:
  jax_over_numpy = ModuleType("jax_over_numpy")
  jaxOver(jax_over_numpy, np)
else:
  jax_over_numpy = None
if jax_over_cupy:
  jax_over_cupy = ModuleType("jax_over_cupy")
  jaxOver(jax_over_cupy, cp)
else:
  jax_over_cupy = None
