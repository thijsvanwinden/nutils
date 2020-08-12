# Copyright (c) 2020 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import typing
if typing.TYPE_CHECKING:
  from typing_extensions import Protocol
else:
  Protocol = object

from typing import Tuple, Union, Type, Callable, Sequence, Any, Optional, Iterator, Dict, Mapping, overload, List, Set
from . import evaluable, numeric, util, expression, types
from .transformseq import Transforms
import builtins, numpy, re, types as builtin_types, itertools, functools, operator, abc, numbers

IntoInt = Union['Int', int]
IntoArray = Union['Array', numpy.ndarray, IntoInt]
Shape = Sequence[IntoInt]
DType = Type[Union[bool, int, float]]

class Lowerable(Protocol):
  def prepare_eval(self, *, ndims: int, opposite: bool = False, replacements: Optional[Mapping[str, 'Array']] = None) -> evaluable.Array: ...

class Int(Lowerable):

  @classmethod
  def cast(cls, value: IntoInt) -> 'Int':
    if isinstance(value, numbers.Integral):
      return cls(_Constant(int(value)))
    elif isinstance(value, cls):
      return value
    else:
      raise ValueError

  def __init__(self, array: 'Array') -> None:
    if array.dtype != int:
      raise ValueError('Expected an `Array` with dtype `int` but got `{}`.'.format(array.dtype.__name__))
    if array.ndim != 0:
      raise ValueError('Expected an `Array` with dimension 0 buut got {}.'.format(array.ndim))
    self._array = array

  def __getnewargs__(self) -> Tuple['Array']:
    return self._array,

  def as_array(self) -> 'Array':
    return self._array

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return self._array.prepare_eval(**kwargs)

  @property
  def try_as_int(self) -> Union[int, 'Int']:
    if isinstance(self._array, _Constant):
      return int(self._array._value.value[()])
    else:
      return self

  def _binop(self, op: Callable[['Array', 'Array'], 'Array'], other_: IntoInt) -> Any:
    try:
      other = Int.cast(other_)
    except ValueError:
      return NotImplemented
    return Int(op(self._array, other._array))

  def _rbinop(self, op: Callable[['Array', 'Array'], 'Array'], other_: IntoInt) -> Any:
    try:
      other = Int.cast(other_)
    except ValueError:
      return NotImplemented
    return Int(op(other._array, self._array))

  def __add__(self, other: IntoInt) -> Any: return self._binop(add, other)
  def __radd__(self, other: IntoInt) -> Any: return self._rbinop(add, other)
  def __sub__(self, other: IntoInt) -> Any: return self._binop(subtract, other)
  def __rsub__(self, other: IntoInt) -> Any: return self._rbinop(subtract, other)
  def __mul__(self, other: IntoInt) -> Any: return self._binop(multiply, other)
  def __rmul__(self, other: IntoInt) -> Any: return self._rbinop(multiply, other)
  def __mod__(self, other: IntoInt) -> Any: return self._binop(mod, other)
  def __rmod__(self, other: IntoInt) -> Any: return self._rbinop(mod, other)
  def __pos__(self) -> 'Int': return self
  def __neg__(self) -> 'Int': return Int(negative(self._array))
  def __abs__(self) -> 'Int': return Int(abs(self._array))

def _containsarray(arg: Any) -> bool:
  return any(map(_containsarray, arg)) if isinstance(arg, (list, tuple)) else isinstance(arg, Array) or numeric.isnumber(arg)

class Array(Lowerable):

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def _check(self, value: evaluable.Array) -> evaluable.Array:
    assert isinstance(value, evaluable.Array)
    assert value.ndim == self.ndim
    for n, m in zip(value.shape, self.shape):
      if isinstance(m, int):
        assert n == m
    return value

  @classmethod
  def cast(cls, value: IntoArray) -> 'Array':
    if isinstance(value, Array):
      return value
    elif isinstance(value, Int):
      return value.as_array()
    elif numeric.isnumber(value) or numeric.isarray(value):
      return _Constant(value)
    elif isinstance(value, (list, tuple)):
      return stack(value, axis=0)
    else:
      raise ValueError

  def __init__(self, shape: Shape, dtype: DType) -> None:
    self.shape = tuple(n.try_as_int if isinstance(n, Int) else int(n) for n in shape)
    self.dtype = dtype

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def __getitem__(self, item: Any) -> 'Array':
    if not isinstance(item, tuple):
      item = item,
    iell = None
    nx = self.ndim - len(item)
    for i, it in enumerate(item):
      if it is ...:
        assert iell is None, 'at most one ellipsis allowed'
        iell = i
      elif it is numpy.newaxis:
        nx += 1
    array = self
    axis = 0
    for it in item + (slice(None),)*nx if iell is None else item[:iell] + (slice(None),)*(nx+1) + item[iell+1:]:
      if isinstance(it, numbers.Integral):
        array = get(array, axis, it)
      else:
        array = expand_dims(array, axis) if it is numpy.newaxis \
           else _takeslice(array, it, axis) if isinstance(it, slice) \
           else take(array, it, axis)
        axis += 1
    assert axis == array.ndim
    return array

  def __bool__(self) -> bool:
    return True

  def __len__(self) -> int:
    if self.ndim == 0:
      raise TypeError('len() of unsized object')
    elif not isinstance(self.shape[0], int):
      raise ValueError('unknown length')
    return self.shape[0]

  def __iter__(self) -> Iterator['Array']:
    if not self.shape:
      raise TypeError('iteration over a 0-d array')
    elif not isinstance(self.shape[0], int):
      raise ValueError('iteration over array with unknown length')
    return (self[i,...] for i in range(self.shape[0]))

  @property
  def size(self) -> Union[int, Int]:
    return util.product(self.shape, 1)

  @property
  def T(self) -> 'Array':
    return transpose(self)

  def _binop(self, op: Callable[['Array', 'Array'], 'Array'], other_: IntoArray) -> Any:
    try:
      other = Array.cast(other_)
    except ValueError:
      return NotImplemented
    return op(self, other)

  def _rbinop(self, op: Callable[['Array', 'Array'], 'Array'], other_: IntoArray) -> Any:
    try:
      other = Array.cast(other_)
    except ValueError:
      return NotImplemented
    return op(other, self)

  def __add__(self, other: IntoArray) -> Any: return self._binop(add, other)
  def __radd__(self, other: IntoArray) -> Any: return self._rbinop(add, other)
  def __sub__(self, other: IntoArray) -> Any: return self._binop(subtract, other)
  def __rsub__(self, other: IntoArray) -> Any: return self._rbinop(subtract, other)
  def __mul__(self, other: IntoArray) -> Any: return self._binop(multiply, other)
  def __rmul__(self, other: IntoArray) -> Any: return self._rbinop(multiply, other)
  def __truediv__(self, other: IntoArray) -> Any: return self._binop(divide, other)
  def __rtruediv__(self, other: IntoArray) -> Any: return self._rbinop(divide, other)
  def __pow__(self, other: IntoArray) -> Any: return self._binop(power, other)
  def __rpow__(self, other: IntoArray) -> Any: return self._rbinop(power, other)
  def __mod__(self, other: IntoArray) -> Any: return self._binop(mod, other)
  def __rmod__(self, other: IntoArray) -> Any: return self._rbinop(mod, other)
  def __pos__(self) -> 'Array': return self
  def __neg__(self) -> 'Array': return negative(self)
  def __abs__(self) -> 'Array': return abs(self)

  def sum(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'Array':
    return sum(self, axis)

  def prod(self, _axis: int) -> 'Array':
    return product(self, _axis)

  def dot(self, _other: IntoArray, axes: Optional[Union[int, Sequence[int]]] = None) -> 'Array':
    return dot(self, _other, axes)

  def normalized(self, _axis: int = -1) -> 'Array':
    return normalized(self, _axis)

  def normal(self, exterior: bool = False) -> 'Array':
    return normal(self, exterior)

  def curvature(self, ndims: int = -1) -> 'Array':
    return curvature(self, ndims)

  def swapaxes(self, _axis1: int, _axis2: int) -> 'Array':
    return swapaxes(self, _axis1, _axis2)

  def transpose(self, _axes: Optional[Sequence[int]]) -> 'Array':
    return transpose(self, _axes)

  def add_T(self, axes: Tuple[int, int]) -> 'Array':
    return add_T(self, axes)

  def grad(self, _geom: IntoArray, ndims: int = 0) -> 'Array':
    return grad(self, _geom, ndims)

  def laplace(self, _geom: IntoArray, ndims: int = 0) -> 'Array':
    return laplace(self, _geom, ndims)

  def symgrad(self, _geom: IntoArray, ndims: int = 0) -> 'Array':
    return symgrad(self, _geom, ndims)

  def div(self, _geom: IntoArray, ndims: int = 0) -> 'Array':
    return div(self, _geom, ndims)

  def dotnorm(self, _geom: IntoArray, axis: int = -1) -> 'Array':
    return dotnorm(self, _geom, axis)

  #tangent = lambda self, vec: tangent(self, vec)

  def ngrad(self, _geom: IntoArray, ndims: int = 0) -> 'Array':
    return ngrad(self, _geom, ndims)

  def nsymgrad(self, _geom: IntoArray, ndims: int = 0) -> 'Array':
    return nsymgrad(self, _geom, ndims)

  def choose(self, _choices: Sequence[IntoArray]) -> 'Array':
    return choose(self, _choices)

  def vector(self, ndims):
    if not self.ndim:
      raise Exception('a scalar function cannot be vectorized')
    return ravel(diagonalize(insertaxis(self, 1, ndims), 1), 0)

  def __repr__(self) -> str:
    return 'Array<{}>'.format(','.join('?' if isinstance(n, Int) else str(n) for n in self.shape))

  @property
  def simplified(self):
    return self

class _Wrapper(Array):

  def __init__(self, prepare_eval: Callable[..., evaluable.Array], *args: Lowerable, shape: Shape, dtype: DType) -> None:
    self._prepare_eval = prepare_eval
    self._args = args
    assert all(hasattr(arg, 'prepare_eval') for arg in self._args)
    super().__init__(shape, dtype)

  def __getnewargs__(self):
    return (self._prepare_eval, *self._args)

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return self._prepare_eval(*(arg.prepare_eval(**kwargs) for arg in self._args))

class _BroadcastedWrapper(_Wrapper):

  _dtypes = bool, int, float

  def __init__(self, prepare_eval: Callable[..., evaluable.Array], *args: IntoArray, min_dtype: Optional[DType] = None, force_dtype: Optional[DType] = None) -> None:
    broadcasted, shape, dtype = _broadcast(*args)
    assert not min_dtype or not force_dtype
    if min_dtype and (self._dtypes.index(dtype) < self._dtypes.index(min_dtype)):
      dtype = min_dtype
    if force_dtype:
      dtype = force_dtype
    self._min_dtype = min_dtype
    self._force_dtype = force_dtype
    super().__init__(prepare_eval, *broadcasted, shape=shape, dtype=dtype)

  def __getnewargs__(self):
    return (*super().__getnewargs__(), self._min_dtype, self._force_dtype)

class _Constant(Array):

  def __init__(self, value: Any) -> None:
    self._value = evaluable.Constant(value)
    super().__init__(self._value.shape, self._value.dtype)
    self._value

  def __getnewargs__(self):
    return self._value,

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return typing.cast(evaluable.Array, self._value)

class _Transpose(Array):

  @classmethod
  def _end(cls, array: Array, axes: Tuple[int, ...], invert: bool = False) -> Array:
    axes = tuple(numeric.normdim(array.ndim, axis) for axis in axes)
    if all(a == b for a, b in enumerate(axes, start=array.ndim-len(axes))):
      return array
    trans = [i for i in range(array.ndim) if i not in axes]
    trans.extend(axes)
    if len(trans) != array.ndim:
      raise Exception('duplicate axes')
    return cls(array, numpy.argsort(trans) if invert else trans)

  @classmethod
  def from_end(cls, array: Array, *axes: int) -> Array:
    return cls._end(array, axes, invert=True)

  @classmethod
  def to_end(cls, array: Array, *axes: int) -> Array:
    return cls._end(array, axes, invert=False)

  def __init__(self, arg: Array, axes: Tuple[int, ...]) -> None:
    self._arg = arg
    self._axes = axes
    super().__init__(tuple(arg.shape[axis] for axis in axes), arg.dtype)

  def __getnewargs__(self):
    return self._arg, self._axes

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return typing.cast(evaluable.Array, evaluable.Transpose(self._arg.prepare_eval(**kwargs), self._axes))

class _Opposite(Array):

  def __init__(self, arg: Array) -> None:
    self._arg = arg
    super().__init__(arg.shape, arg.dtype)

  def __getnewargs__(self):
    return self._arg,

  def prepare_eval(self, *, opposite: bool = False, **kwargs: Any) -> evaluable.Array:
    return self._arg.prepare_eval(opposite=not opposite, **kwargs)

class _Derivative(Array):

  def __init__(self, arg: Array, var: Array) -> None:
    self._arg = arg
    self._var = var
    super().__init__(arg.shape+var.shape, arg.dtype)

  def __getnewargs__(self):
    return self._arg, self._var

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    arg = self._arg.prepare_eval(**kwargs)
    var = self._var.prepare_eval(**kwargs)
    return evaluable.derivative(arg, var)

class Argument(Array):

  def __init__(self, name: str, shape: Shape) -> None:
    self.name = name
    super().__init__(shape, float)

  def __getnewargs__(self):
    return self.name, self.shape

  def prepare_eval(self, *, replacements: Optional[Mapping[str, Array]] = None, **kwargs: Any) -> evaluable.Array:
    if replacements and self.name in replacements:
      replacement = replacements[self.name]
      assert replacement.shape == self.shape
      # The `replacements` are deliberately omitted while lowering
      # `replacement`, as this could cause infinite recursions. Super
      # replacements are applied to the replacements themselves by
      # `_Replace.prepare_eval`.
      return replacement.prepare_eval(**kwargs)
    shape = tuple(n.prepare_eval(**kwargs) if isinstance(n, Int) else n for n in self.shape)
    return evaluable.Argument(self.name, shape)

class _Replace(Array):

  def __init__(self, arg: Array, replacements: Dict[str, Array]) -> None:
    self._arg = arg
    self._replacements = replacements
    super().__init__(arg.shape, arg.dtype)

  def __getnewargs__(self):
    return self._arg, self._replacements

  def prepare_eval(self, *, replacements: Optional[Mapping[str, Array]] = None, **kwargs: Any) -> evaluable.Array:
    if replacements:
      # Apply super `replacements` to `self._replacements`.
      #
      #   _Replace(_Replace(x, dict(x=y)), dict(y=z)) -> z
      replacements = dict(replacements)
      self_replacements = {k: _Replace(v, replacements) for k, v in self._replacements.items()}
      new_replacements = dict(replacements)
      # Merge with and overwrite super `replacements` with
      # `self._replacements`.
      #
      #   _Replace(_Replace(x, dict(x=y)), dict(x=z)) -> y
      new_replacements.update(self_replacements)
    else:
      new_replacements = dict(self._replacements)
    return self._arg.prepare_eval(replacements=new_replacements, **kwargs)

class _LocalCoords(Array):

  def __init__(self, ndims: int) -> None:
    super().__init__((ndims,), float)

  def __getnewargs__(self):
    return self.shape[0],

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return evaluable.LocalCoords(self.shape[0])

class _RootCoords(Array):

  def __init__(self, ndims: int) -> None:
    super().__init__((ndims,), float)

  def __getnewargs__(self):
    return self.shape[0],

  def prepare_eval(self, *, opposite: bool = False, **kwargs) -> evaluable.Array:
    return evaluable.rootcoords(self.shape[0], evaluable.SelectChain(int(opposite)))

class _Jacobian(Array):

  def __init__(self, geom: Array) -> None:
    assert geom.ndim == 1
    self._geom = geom
    super().__init__((), float)

  def __getnewargs__(self):
    return self._geom,

  def prepare_eval(self, *, ndims: int, **kwargs: Any) -> evaluable.Array:
    return evaluable.jacobian(self._geom.prepare_eval(ndims=ndims, **kwargs), ndims)

class _TransformsIndex(Array):

  def __init__(self, transforms: Transforms) -> None:
    self._transforms = transforms
    super().__init__((), int)

  def __getnewargs__(self):
    return self._transforms,

  def prepare_eval(self, *, opposite: bool = False, **kwargs: Any) -> evaluable.Array:
    index, tail = evaluable.TransformsIndexWithTail(self._transforms, evaluable.SelectChain(int(opposite)))
    return index

class _TransformsCoords(Array):

  def __init__(self, transforms: Transforms, dim: int) -> None:
    self._transforms = transforms
    super().__init__((dim,), int)

  def __getnewargs__(self):
    return self._transforms, self.shape[0]

  def prepare_eval(self, *, opposite: bool = False, **kwargs: Any) -> evaluable.Array:
    index, tail = evaluable.TransformsIndexWithTail(self._transforms, evaluable.SelectChain(int(opposite)))
    return evaluable.ApplyTransforms(tail)

class _Zeros(Array):

  def __getnewargs__(self):
    return self.shape, self.dtype

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return evaluable.Zeros(tuple(Int.cast(n).prepare_eval(**kwargs) for n in self.shape), self.dtype)

class _Ones(Array):

  def __getnewargs__(self):
    return self.shape, self.dtype

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return evaluable.ones(tuple(Int.cast(n).prepare_eval(**kwargs) for n in self.shape), self.dtype)

class _Normal(Array):

  def __init__(self, lgrad: Array) -> None:
    assert lgrad.ndim == 2 and lgrad.shape[0] == lgrad.shape[1]
    self._lgrad = lgrad
    super().__init__((lgrad.shape[0],), float)

  def __getnewargs__(self):
    return self._lgrad,

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return evaluable.Normal(self._lgrad.prepare_eval(**kwargs))

class _Elemwise(Array):

  def __init__(self, data: Tuple[types.frozenarray, ...], index: Int, dtype: DType) -> None:
    self._data = data
    self._index = index
    ndim = self._data[0].ndim if self._data else 0
    shape = tuple(Int(get([d.shape[i] for d in self._data], 0, index)) for i in range(ndim))
    super().__init__(shape, dtype)

  def __getnewargs__(self):
    return self._data, self._index, self.dtype

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    return evaluable.Elemwise(self._data, self._index.prepare_eval(**kwargs), self.dtype)

def _join_lengths(*lengths_: IntoInt) -> IntoInt:
  lengths = set(lengths_)
  if len(lengths) == 1:
    return next(iter(lengths))
  elif len(lengths - {1}) == 1:
    return next(iter(lengths - {1}))
  else:
    return Int(_Wrapper(evaluable.AssertEqual, *map(Int.cast, lengths), shape=(), dtype=int))

def _broadcast(*args_: IntoArray) -> Tuple[Tuple[Array, ...], Shape, DType]:
  args = tuple(map(Array.cast, args_))
  ndim = builtins.max(arg.ndim for arg in args)
  shape = tuple(_join_lengths(*(arg.shape[i+arg.ndim-ndim] for arg in args if i+arg.ndim-ndim >= 0)) for i in range(ndim))
  broadcasted = []
  for arg in args:
    for i, (n, m) in enumerate(zip(shape[ndim-arg.ndim:], arg.shape)):
      if m == 1 and n != m:
        arg = repeat(arg, n, i)
    arg = _prepend_axes(arg, shape[:ndim-arg.ndim])
    broadcasted.append(arg)
  return tuple(broadcasted), shape, evaluable._jointdtype(*(arg.dtype for arg in args))

# CONSTRUCTORS

def asarray(_arg: IntoArray) -> Array:
  return Array.cast(_arg)

def zeros(_shape: Shape, dtype: DType = float) -> Array:
  return _Zeros(_shape, dtype)

def ones(_shape: Shape, dtype: DType = float) -> Array:
  return _Ones(_shape, dtype)

def eye(n, dtype=float):
  return diagonalize(ones([n], dtype=dtype))

def levicivita(_n: int, dtype: DType = float) -> Array:
  'n-dimensional Levi-Civita symbol.'
  return _Constant(numeric.levicivita(_n))

# ARITHMETIC

def add(_left: IntoArray, _right: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.add, _left, _right)

def subtract(_left: IntoArray, _right: IntoArray) -> Array:
  return add(_left, negative(_right))

def negative(_arg: IntoArray) -> Array:
  return multiply(_arg, -1)

def multiply(_left: IntoArray, _right: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.multiply, _left, _right)

def divide(_left: IntoArray, _right: IntoArray) -> Array:
  return multiply(_left, reciprocal(_right))

def reciprocal(_arg: IntoArray) -> Array:
  return power(_arg, -1)

def power(_left: IntoArray, _right: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.power, _left, _right, min_dtype=float)

def sqrt(_arg: IntoArray) -> Array:
  return power(_arg, .5)

def abs(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return arg * sign(arg)

def sign(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Sign, _arg)

def mod(_a: IntoArray, _b: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Mod, _a, _b)

# TRIGONOMETRIC

def cos(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Cos, _arg, min_dtype=float)

def sin(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Sin, _arg, min_dtype=float)

def tan(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Tan, _arg, min_dtype=float)

def arccos(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.ArcCos, _arg, min_dtype=float)

def arcsin(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.ArcSin, _arg, min_dtype=float)

def arctan(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.ArcTan, _arg, min_dtype=float)

def arctan2(_a: IntoArray, _b: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.ArcTan2, _a, _b, min_dtype=float)

def cosh(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return .5 * (exp(arg) + exp(-arg))

def sinh(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return .5 * (exp(arg) - exp(-arg))

def tanh(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return 1 - 2. / (exp(2*arg) + 1)

def arctanh(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return .5 * (ln(1+arg) - ln(1-arg))

def exp(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Exp, _arg, min_dtype=float)

def log(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Log, _arg, min_dtype=float)

ln = log

def log2(_arg: IntoArray) -> Array:
  return log(_arg) / log(2)

def log10(_arg: IntoArray) -> Array:
  return log(_arg) / log(10)

# COMPARISON

def greater(_l: IntoArray, _r: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Greater, _l, _r, force_dtype=bool)

def equal(_l: IntoArray, _r: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Equal, _l, _r, force_dtype=bool)

def less(_l: IntoArray, _r: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Less, _l, _r, force_dtype=bool)

def min(_a: IntoArray, _b: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Minimum, _a, _b)

def max(_a: IntoArray, _b: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Maximum, _a, _b)

# OPPOSITE

def opposite(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return _Opposite(arg)

def mean(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return .5 * (arg + opposite(arg))

def jump(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return opposite(arg) - arg

# REDUCTION

def sum(_arg: IntoArray, axis: Optional[Union[int, Sequence[int]]] = None) -> Array:
  arg = Array.cast(_arg)
  if axis is None:
    if arg.ndim == 0:
      raise ValueError('Cannot sum last axis of 0-d array.')
    return _Wrapper(evaluable.Sum, arg, shape=arg.shape[:-1], dtype=arg.dtype)
  axes = typing.cast(Sequence[int], (axis,) if isinstance(axis, numbers.Integral) else axis)
  summed = _Transpose.to_end(arg, *axes)
  for i in range(len(axes)):
    summed = _Wrapper(evaluable.Sum, summed, shape=summed.shape[:-1], dtype=summed.dtype)
  return summed

def product(_arg: IntoArray, axis: int) -> Array:
  arg = Array.cast(_arg)
  transposed = _Transpose.to_end(arg, axis)
  return _Wrapper(evaluable.Product, transposed, shape=transposed.shape[:-1], dtype=transposed.dtype)

# LINEAR ALGEBRA

def dot(_a: IntoArray, _b: IntoArray, axes: Optional[Union[int, Sequence[int]]] = None) -> Array:
  a = Array.cast(_a)
  b = Array.cast(_b)
  if axes is None:
    assert b.ndim == 1 and b.shape[0] == a.shape[0]
    b = _append_axes(b, a.shape[1:])
    axes = 0,
  return sum(multiply(a, b), axes)

def trace(_arg: IntoArray, _n1: int = -2, _n2: int = -1) -> Array:
  return sum(_takediag(_arg, _n1, _n2), -1)

def norm2(_arg: IntoArray, _axis: Union[int, Sequence[int]] = -1) -> Array:
  arg = Array.cast(_arg)
  return sqrt(sum(multiply(arg, arg), _axis))

def normalized(_arg: IntoArray, _axis: int = -1) -> Array:
  arg = Array.cast(_arg)
  return divide(arg, insertaxis(norm2(arg, _axis), _axis, 1))

def matmat(_arg0: IntoArray, *args: IntoArray) -> Array:
  'helper function, contracts last axis of arg0 with first axis of arg1, etc'
  retval = Array.cast(_arg0)
  for arg in map(Array.cast, args):
    if retval.shape[-1] != arg.shape[0]:
      raise ValueError('incompatible shapes')
    retval = dot(retval[(...,)+(numpy.newaxis,)*(arg.ndim-1)], arg[(numpy.newaxis,)*(retval.ndim-1)], retval.ndim-1)
  return retval

def inverse(_arg: IntoArray, _axes: Tuple[int, int] = (-2,-1)) -> Array:
  transposed = _Transpose.to_end(Array.cast(_arg), *_axes)
  if transposed.shape[-2] != transposed.shape[-1]:
    raise ValueError('cannot compute the inverse along two axes with different lengths')
  inverted = _Wrapper(evaluable.Inverse, transposed, shape=transposed.shape, dtype=float)
  return _Transpose.from_end(inverted, *_axes)

def determinant(_arg: IntoArray, _axes: Tuple[int, int] = (-2,-1)) -> Array:
  transposed = _Transpose.to_end(Array.cast(_arg), *_axes)
  return _Wrapper(evaluable.Determinant, transposed, shape=transposed.shape[:-2], dtype=float)

def _eval_eigval(arg: evaluable.Array) -> evaluable.Array:
  return evaluable.Eig(arg, False)[0]

def _eval_eigval_symmetric(arg: evaluable.Array) -> evaluable.Array:
  return evaluable.Eig(arg, True)[0]

def _eval_eigvec(arg: evaluable.Array) -> evaluable.Array:
  return evaluable.Eig(arg, False)[1]

def _eval_eigvec_symmetric(arg: evaluable.Array) -> evaluable.Array:
  return evaluable.Eig(arg, True)[1]

def eig(_arg: IntoArray, _axes: Tuple[int, int] = (-2,-1), symmetric: bool = False) -> Tuple[Array, Array]:
  arg = Array.cast(_arg)
  transposed = _Transpose.to_end(arg, *_axes)
  # FIXME: symmetric is not lowerable
  eigval = _Wrapper(_eval_eigval_symmetric if symmetric else _eval_eigval, arg, shape=arg.shape[:-1], dtype=float)
  eigvec = _Wrapper(_eval_eigvec_symmetric if symmetric else _eval_eigvec, arg, shape=arg.shape, dtype=float)
  return diagonalize(eigval), eigvec

def _takediag(_arg: IntoArray, _axis1: int = -2, _axis2: int =-1) -> Array:
  arg = Array.cast(_arg)
  transposed = _Transpose.to_end(arg, _axis1, _axis2)
  if transposed.shape[-2] != transposed.shape[-1]:
    raise ValueError('cannot take the diagonal along two axes with different lengths')
  return _Wrapper(evaluable.TakeDiag, transposed, shape=transposed.shape[:-1], dtype=transposed.dtype)

def takediag(_arg: IntoArray, _axis: int = -2, _rmaxis: int = -1) -> Array:
  arg = Array.cast(_arg)
  axis = numeric.normdim(arg.ndim, _axis)
  rmaxis = numeric.normdim(arg.ndim, _rmaxis)
  assert axis < rmaxis
  return _Transpose.from_end(_takediag(arg, axis, rmaxis), axis)

def diagonalize(_arg: IntoArray, _axis: int = -1, _newaxis: int = -1) -> Array:
  arg = Array.cast(_arg)
  axis = numeric.normdim(arg.ndim, _axis)
  newaxis = numeric.normdim(arg.ndim+1, _newaxis)
  assert axis < newaxis
  transposed = _Transpose.to_end(arg, axis)
  diagonalized = _Wrapper(evaluable.Diagonalize, transposed, shape=(*transposed.shape, transposed.shape[-1]), dtype=transposed.dtype)
  return _Transpose.from_end(diagonalized, axis, newaxis)

def cross(_arg1: IntoArray, _arg2: IntoArray, axis: int) -> Array:
  (arg1, arg2), shape, dtype = _broadcast(_arg1, _arg2)
  axis = numeric.normdim(arg1.ndim, axis)
  assert arg1.shape[axis] == 3
  i = Array.cast(types.frozenarray([1, 2, 0]))
  j = Array.cast(types.frozenarray([2, 0, 1]))
  return _take(arg1, i, axis) * _take(arg2, j, axis) - _take(arg2, i, axis) * _take(arg1, j, axis)

def outer(arg1, arg2=None, axis=0):
  'outer product'

  if arg2 is None:
    arg2 = arg1
  elif arg1.ndim != arg2.ndim:
    raise ValueError('arg1 and arg2 have different dimensions')
  axis = numeric.normdim(arg1.ndim, axis)
  return expand_dims(arg1,axis+1) * expand_dims(arg2,axis)

# ARRAY OPS

def transpose(_arg: IntoArray, _axes: Optional[Sequence[int]] = None) -> Array:
  arg = Array.cast(_arg)
  if _axes is None:
    axes = tuple(reversed(range(arg.ndim)))
  else:
    axes = tuple(numeric.normdim(arg.ndim, axis) for axis in _axes)
  return _Transpose(arg, axes)

def _append_axes(_arg: IntoArray, _shape: Shape) -> Array:
  arg = Array.cast(_arg)
  for n in _shape:
    arg = _Wrapper(evaluable.InsertAxis, arg, Int.cast(n), shape=(*arg.shape, n), dtype=arg.dtype)
  return arg

def _prepend_axes(_arg: IntoArray, _shape: Shape) -> Array:
  arg = Array.cast(_arg)
  appended = _append_axes(arg, _shape)
  return _Transpose.from_end(appended, *range(len(_shape)))

def insertaxis(_arg: IntoArray, _axis: int, _length: IntoInt) -> Array:
  appended = _append_axes(_arg, (_length,))
  return _Transpose.from_end(appended, _axis)

def expand_dims(_arg: IntoArray, _axis: int) -> Array:
  return insertaxis(_arg, _axis, 1)

def repeat(_arg: IntoArray, _length: IntoInt, _axis: int) -> Array:
  arg = Array.cast(_arg)
  length = Int.cast(_length)
  assert arg.shape[_axis] == 1
  return insertaxis(get(arg, _axis, 0), _axis, length)

def swapaxes(_arg: IntoArray, _axis1: int, _axis2: int) -> Array:
  arg = Array.cast(_arg)
  trans = list(range(arg.ndim))
  trans[_axis1], trans[_axis2] = trans[_axis2], trans[_axis1]
  return transpose(arg, trans)

def ravel(_arg: IntoArray, axis: int) -> Array:
  arg = Array.cast(_arg)
  axis = numeric.normdim(arg.ndim-1, axis)
  transposed = _Transpose.to_end(arg, axis, axis+1)
  raveled = _Wrapper(evaluable.Ravel, transposed, shape=(*transposed.shape[:-2], transposed.shape[-2]*transposed.shape[-1]), dtype=transposed.dtype)
  return _Transpose.from_end(raveled, axis)

def unravel(_arg: IntoArray, _axis: int, _shape: Tuple[IntoInt, IntoInt]) -> Array:
  arg = Array.cast(_arg)
  axis = numeric.normdim(arg.ndim, _axis)
  assert len(_shape) == 2
  shape = tuple(map(Int.cast, _shape))
  transposed = _Transpose.to_end(arg, axis)
  unraveled = _Wrapper(evaluable.Unravel, transposed, *shape, shape=(*transposed.shape[:-1], *shape), dtype=transposed.dtype)
  return _Transpose.from_end(unraveled, axis, axis+1)

def _take(_arg: Array, _index: Union[Array, Int], _axis: int) -> Array:
  axis = numeric.normdim(_arg.ndim, _axis)
  index_shape = () if isinstance(_index, Int) else _index.shape
  transposed = _Transpose.to_end(_arg, axis)
  taken = _Wrapper(evaluable.Take, transposed, _index, shape=(*transposed.shape[:-1], *index_shape), dtype=_arg.dtype)
  return _Transpose.from_end(taken, *range(axis, axis+len(index_shape)))

def take(_arg: IntoArray, _index: IntoArray, axis: int) -> Array:
  arg = Array.cast(_arg)
  index = Array.cast(_index)
  assert index.ndim == 1
  length = arg.shape[axis]
  if index.dtype == bool:
    assert index.shape[0] == length
    index = find(index)
  return _take(arg, index, axis)

def get(_arg: IntoArray, _iax: int, _item: IntoInt) -> Array:
  arg = Array.cast(_arg)
  if isinstance(_item, numbers.Integral):
    length = arg.shape[_iax]
    if isinstance(length, numbers.Integral):
      _item = numeric.normdim(int(length), int(_item))
  return _take(arg, Int.cast(_item), _iax)

def _range(_length: IntoInt, _offset: IntoInt) -> Array:
  length = Int.cast(_length)
  offset = Int.cast(_offset)
  return _Wrapper(evaluable.Range, length, offset, shape=(length,), dtype=int)

def _takeslice(_arg: IntoArray, _s: slice, _axis: int) -> Array:
  arg = Array.cast(_arg)
  s = _s
  axis = _axis
  n = arg.shape[axis]
  if s.step == None or s.step == 1:
    start = 0 if s.start is None else s.start if s.start >= 0 else s.start + n
    stop = n if s.stop is None else s.stop if s.stop >= 0 else s.stop + n
    if start == 0 and stop == n:
      return arg
    index = _range(Int.cast(stop-start), Int.cast(start))
  elif isinstance(n, numbers.Integral):
    index = Array.cast(numpy.arange(*s.indices(int(n))))
  else:
    raise Exception('a non-unit slice requires a constant-length axis')
  return _take(arg, index, axis)

def _inflate(_arg: IntoArray, dofmap: IntoArray, length: IntoInt, axis: int) -> Array:
  arg = Array.cast(_arg)
  dofmap = Array.cast(dofmap)
  length = Int.cast(length)
  axis = numeric.normdim(arg.ndim+1-dofmap.ndim, axis)
  # TODO: assert dofmap.shape == arg.shape[axis:axis+dofmap.ndim]
  transposed = _Transpose.to_end(arg, *range(axis, axis+dofmap.ndim))
  inflated = _Wrapper(evaluable.Inflate, transposed, dofmap, length, shape=(*transposed.shape[:transposed.ndim-dofmap.ndim], length), dtype=transposed.dtype)
  return _Transpose.from_end(inflated, axis)

def concatenate(_args: Sequence[IntoArray], axis: int = 0) -> Array:
  args = tuple(map(Array.cast, _args))
  # TODO: broadcasting
  axis = numeric.normdim(args[0].ndim, axis)
  length = util.sum(arg.shape[axis] for arg in args)
  return util.sum(_inflate(arg, _range(Int.cast(arg.shape[axis]), Int.cast(offset)), length, axis)
    for arg, offset in zip(args, util.cumsum(arg.shape[axis] for arg in args)))

def stack(_args: Sequence[IntoArray], axis: int = 0) -> Array:
  aligned, shape, dtype = _broadcast(*_args)
  return util.sum(kronecker(arg, axis, len(aligned), i) for i, arg in enumerate(aligned))

def kronecker(_arg: IntoArray, axis: int, length: IntoInt, pos: IntoInt) -> Array:
  arg = Array.cast(_arg)
  length = Int.cast(length)
  pos = Int.cast(pos)
  inflated = _Wrapper(evaluable.Inflate, arg, pos, length, shape=(*arg.shape, length), dtype=arg.dtype)
  return _Transpose.from_end(inflated, axis)

def find(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  assert arg.ndim == 1 and arg.dtype == bool
  return _Wrapper(evaluable.Find, arg, shape=(Int(_array_int(arg).sum()),) ,dtype=int)

def replace_arguments(value: IntoArray, arguments: Mapping[str, IntoArray]) -> Array:
  return _Replace(Array.cast(value), {k: Array.cast(v) for k, v in arguments.items()})

# DERIVATIVES

def derivative(_arg: IntoArray, _var: IntoArray) -> Array:
  arg = Array.cast(_arg)
  var = Array.cast(_var)
  return _Derivative(arg, var)

def localgradient(_arg: IntoArray, _ndims: int) -> Array:
  return derivative(_arg, _LocalCoords(_ndims))

def grad(_func: IntoArray, _geom: IntoArray, ndims: int = 0) -> Array:
  func = Array.cast(_func)
  geom = Array.cast(_geom)
  if geom.ndim == 0:
    return grad(_func, _append_axes(_geom, (1,)))[...,0]
  elif geom.ndim > 1:
    sh = geom.shape[-2:]
    return unravel(grad(func, ravel(geom, geom.ndim-2), ndims), func.ndim+geom.ndim-2, sh)
  else:
    if ndims <= 0:
      ndims += geom.shape[0]
    J = localgradient(geom, ndims)
    if J.shape[0] == J.shape[1]:
      Jinv = inverse(J)
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      G = dot(J[:,:,numpy.newaxis], J[:,numpy.newaxis,:], 0)
      Ginv = inverse(G)
      Jinv = dot(J[numpy.newaxis,:,:], Ginv[:,numpy.newaxis,:], -1)
    else:
      raise Exception('cannot invert {}x{} jacobian'.format(J.shape[0], J.shape[1]))
    return dot(_append_axes(localgradient(func, ndims), Jinv.shape[-1:]), Jinv, -2)

def normal(_arg: IntoArray, exterior: bool = False) -> Array:
  arg = Array.cast(_arg)
  if arg.ndim == 0:
    return normal(insertaxis(arg, 0, 1), exterior)[...,0]
  elif arg.ndim > 1:
    sh = arg.shape[-2:]
    return unravel(normal(ravel(arg, arg.ndim-2), exterior), arg.ndim-2, sh)
  else:
    if not exterior:
      lgrad = localgradient(arg, len(arg))
      assert lgrad.ndim == 2 and lgrad.shape[0] == lgrad.shape[1]
      return _Wrapper(evaluable.Normal, lgrad, shape=(lgrad.shape[0],), dtype=float)
    lgrad = localgradient(arg, len(arg)-1)
    if len(arg) == 2:
      return Array.cast([lgrad[1,0], -lgrad[0,0]]).normalized()
    if len(arg) == 3:
      return cross(lgrad[:,0], lgrad[:,1], axis=0).normalized()
    raise NotImplementedError

def dotnorm(_arg: IntoArray, _geom: IntoArray, _axis: int = -1) -> Array:
  arg = Array.cast(_arg)
  geom = Array.cast(_geom)
  axis = numeric.normdim(arg.ndim, _axis)
  assert geom.ndim == 1 and geom.shape[0] == arg.shape[axis]
  return dot(arg, _append_axes(normal(geom), arg.shape[1:]), 0)

def jacobian(_geom: IntoArray, _ndims: Optional[int] = None) -> Array:
  geom = Array.cast(_geom)
  # TODO: check `_ndims` with `ndims` argument passed to `prepare_eval`.
  return _Jacobian(geom)

J = jacobian

def _d1(arg: IntoArray, var: IntoArray) -> Array:
  return derivative(arg, var) if isinstance(var, Argument) else grad(arg, var)

def d(arg: IntoArray, *vars: IntoArray) -> Array:
  'derivative of `arg` to `vars`'
  return functools.reduce(_d1, vars, Array.cast(arg))

def _surfgrad1(arg: IntoArray, geom: IntoArray) -> Array:
  geom = Array.cast(geom)
  return grad(arg, geom, len(geom)-1)

def surfgrad(arg: IntoArray, *vars: IntoArray) -> Array:
  'surface gradient of `arg` to `vars`'
  return functools.reduce(_surfgrad1, vars, arg)

def curvature(_geom: IntoArray, ndims: int = -1) -> Array:
  geom = Array.cast(_geom)
  return geom.normal().div(geom, ndims=ndims)

def div(_arg: IntoArray, _geom: IntoArray, ndims: int = 0) -> Array:
  return trace(grad(_arg, _geom, ndims))

def laplace(_arg: IntoArray, _geom: IntoArray, ndims: int = 0) -> Array:
  arg = Array.cast(_arg)
  geom = Array.cast(_geom)
  return arg.grad(geom, ndims).div(geom, ndims)

def symgrad(_arg: IntoArray, _geom: IntoArray, ndims: int = 0) -> Array:
  arg = Array.cast(_arg)
  geom = Array.cast(_geom)
  return multiply(.5, add_T(arg.grad(geom, ndims)))

def ngrad(_arg: IntoArray, _geom: IntoArray, ndims: int = 0) -> Array:
  arg = Array.cast(_arg)
  geom = Array.cast(_geom)
  return dotnorm(grad(arg, geom, ndims), geom)

def nsymgrad(_arg: IntoArray, _geom: IntoArray, ndims: int = 0) -> Array:
  arg = Array.cast(_arg)
  geom = Array.cast(_geom)
  return dotnorm(symgrad(arg, geom, ndims), geom)

# MISC

def isarray(_arg: Any) -> bool:
  return isinstance(_arg, Array)

def rootcoords(_ndims: int) -> Array:
  return _RootCoords(_ndims)

def transforms_index(transforms: Transforms) -> Int:
  return Int(_TransformsIndex(transforms))

def transforms_coords(transforms: Transforms, dim: int) -> Array:
  return _TransformsCoords(transforms, dim)

def Elemwise(_data: Sequence[numpy.ndarray], _index: IntoInt, dtype: DType) -> Array:
  data = tuple(map(types.frozenarray, _data))
  return _Elemwise(data, Int.cast(_index), dtype)

def Sampled(_points: IntoArray, expect: IntoArray) -> Array:
  points = Array.cast(_points)
  expect = Array.cast(expect)
  assert points.ndim == 1 and expect.ndim == 2 and expect.shape[1] == points.shape[0]
  return _Wrapper(evaluable.Sampled, points, expect, shape=expect.shape[1:], dtype=int)

def piecewise(level: IntoArray, intervals: Sequence[IntoArray], *funcs: IntoArray) -> Array:
  level = Array.cast(level)
  return util.sum(_array_int(greater(level, interval)) for interval in intervals).choose(funcs)

def partition(f: IntoArray, *levels: float) -> Sequence[Array]:
  '''Create a partition of unity for a scalar function f.

  When ``n`` levels are specified, ``n+1`` indicator functions are formed that
  evaluate to one if and only if the following condition holds::

      indicator 0: f < levels[0]
      indicator 1: levels[0] < f < levels[1]
      ...
      indicator n-1: levels[n-2] < f < levels[n-1]
      indicator n: f > levels[n-1]

  At the interval boundaries the indicators evaluate to one half, in the
  remainder of the domain they evaluate to zero such that the whole forms a
  partition of unity. The partitions can be used to create a piecewise
  continuous function by means of multiplication and addition.

  The following example creates a topology consiting of three elements, and a
  function ``f`` that is zero in the first element, parabolic in the second,
  and zero again in the third element.

  >>> from nutils import mesh
  >>> domain, x = mesh.rectilinear([3])
  >>> left, center, right = partition(x[0], 1, 2)
  >>> f = (1 - (2*x[0]-3)**2) * center

  Args
  ----
  f : :class:`Array`
      Scalar-valued function
  levels : scalar constants or :class:`Array`\\s
      The interval endpoints.

  Returns
  -------
  :class:`list` of scalar :class:`Array`\\s
      The indicator functions.
  '''

  f = Array.cast(f)
  signs = [sign(f - level) for level in levels]
  steps = map(subtract, signs[:-1], signs[1:])
  return [.5 - .5 * signs[0]] + [.5 * step for step in steps] + [.5 + .5 * signs[-1]]

def _eval_choose(_arg: evaluable.Array, *_choices: evaluable.Array) -> evaluable.Array:
  return evaluable.Choose(_arg, _choices)

def choose(_arg: IntoArray, _choices: Sequence[IntoArray]) -> Array:
  arg = Array.cast(_arg)
  choices, shape, dtype = _broadcast(*_choices)
  return _Wrapper(_eval_choose, arg, *choices, shape=shape, dtype=dtype)

def chain(_funcs: Sequence[IntoArray]) -> Sequence[Array]:
  'chain'

  funcs = tuple(map(Array.cast, _funcs))
  shapes = [func.shape[0] for func in funcs]
  return [concatenate([func if i==j else zeros((sh,) + func.shape[1:])
             for j, sh in enumerate(shapes)], axis=0)
               for i, func in enumerate(funcs)]

def vectorize(args: Sequence[IntoArray]) -> Array:
  '''
  Combine scalar-valued bases into a vector-valued basis.

  Args
  ----
  args : iterable of 1-dimensional :class:`nutils.function.Array` objects

  Returns
  -------
  :class:`Array`
  '''

  return concatenate([kronecker(arg, axis=-1, length=len(args), pos=iarg) for iarg, arg in enumerate(args)])

def simplified(_arg: IntoArray) -> Array:
  return Array.cast(_arg)

def iszero(_arg: IntoArray) -> bool:
  return False

def _array_int(_arg: IntoArray) -> Array:
  return _BroadcastedWrapper(evaluable.Int, _arg, force_dtype=int)

def add_T(_arg: IntoArray, axes: Tuple[int, int] = (-2,-1)) -> Array:
  arg = Array.cast(_arg)
  return swapaxes(arg, *axes) + arg

def RevolutionAngle() -> Array:
  return _Wrapper(evaluable.RevolutionAngle, shape=(), dtype=float)

def bifurcate1(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return _Wrapper(evaluable.bifurcate1, arg, shape=arg.shape, dtype=arg.dtype)

def bifurcate2(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return _Wrapper(evaluable.bifurcate2, arg, shape=arg.shape, dtype=arg.dtype)

def bifurcate(_arg1: IntoArray, _arg2: IntoArray) -> Tuple[Array, Array]:
  return bifurcate1(_arg1), bifurcate2(_arg2)

def trignormal(_angle: IntoArray) -> Array:
  angle = Array.cast(_angle)
  assert angle.ndim == 0
  return _Wrapper(evaluable.TrigNormal, angle, shape=(2,), dtype=float)

def trigtangent(_angle: IntoArray) -> Array:
  angle = Array.cast(_angle)
  assert angle.ndim == 0
  return _Wrapper(evaluable.TrigTangent, angle, shape=(2,), dtype=float)

def rotmat(_arg: IntoArray) -> Array:
  arg = Array.cast(_arg)
  return stack([trignormal(arg), trigtangent(arg)], 0)

# BASES

class Basis(Array):
  '''Abstract base class for bases.

  A basis is a sequence of elementwise polynomial functions.

  Parameters
  ----------
  ndofs : :class:`int`
      The number of functions in this basis.
  index : :class:`Int`
      The element index.
  coords : :class:`Array`
      The element local coordinates.

  Notes
  -----
  Subclasses must implement :meth:`get_dofs` and :meth:`get_coefficients` and
  if possible should redefine :meth:`get_support`.
  '''

  __slots__ = 'ndofs', 'nelems', 'index', 'coords'
  __cache__ = '_computed_support'

  def __init__(self, ndofs: int, nelems: int, index: Int, coords: Array) -> None:
    self.ndofs = ndofs
    self.nelems = nelems
    self.index = index
    self.coords = coords
    super().__init__((ndofs,), float)

  def prepare_eval(self, **kwargs: Any) -> evaluable.Array:
    index = self.index.prepare_eval(**kwargs)
    coords = self.coords.prepare_eval(**kwargs)
    return evaluable.Inflate(evaluable.Polyval(self.f_coefficients(index), coords), self.f_dofs(index), self.ndofs)

  @property
  def _computed_support(self) -> Tuple[types.frozenarray, ...]:
    support = [set() for i in range(self.ndofs)] # type: List[Set[int]]
    for ielem in range(self.nelems):
      for dof in self.get_dofs(ielem):
        support[dof].add(ielem)
    return tuple(types.frozenarray(numpy.fromiter(sorted(ielems), dtype=int), copy=False) for ielems in support)

  def get_support(self, dof: Union[numbers.Integral, numpy.ndarray]) -> numpy.ndarray:
    '''Return the support of basis function ``dof``.

    If ``dof`` is an :class:`int`, return the indices of elements that form the
    support of ``dof``.  If ``dof`` is an array, return the union of supports
    of the selected dofs as a unique array.  The returned array is always
    unique, i.e. strict monotonic increasing.

    Parameters
    ----------
    dof : :class:`int` or array of :class:`int` or :class:`bool`
        Index or indices of basis function or a mask.

    Returns
    -------
    support : sorted and unique :class:`numpy.ndarray`
        The elements (as indices) where function ``dof`` has support.
    '''

    if isinstance(dof, numbers.Integral):
      return self._computed_support[int(dof)]
    elif numeric.isintarray(dof):
      if dof.ndim != 1:
        raise IndexError('dof has invalid number of dimensions')
      if len(dof) == 0:
        return numpy.array([], dtype=int)
      dof = numpy.unique(dof)
      if dof[0] < 0 or dof[-1] >= self.ndofs:
        raise IndexError('dof out of bounds')
      if self.get_support == __class__.get_support.__get__(self, __class__):
        return numpy.unique([ielem for ielem in range(self.nelems) if numpy.in1d(self.get_dofs(ielem), dof, assume_unique=True).any()])
      else:
        return numpy.unique(numpy.fromiter(itertools.chain.from_iterable(map(self.get_support, dof)), dtype=int))
    elif numeric.isboolarray(dof):
      if dof.shape != (self.ndofs,):
        raise IndexError('dof has invalid shape')
      return self.get_support(numpy.where(dof)[0])
    else:
      raise IndexError('invalid dof')

  @abc.abstractmethod
  def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
    '''Return an array of indices of basis functions with support on element ``ielem``.

    If ``ielem`` is an :class:`int`, return the dofs on element ``ielem``
    matching the coefficients array as returned by :meth:`get_coefficients`.
    If ``ielem`` is an array, return the union of dofs on the selected elements
    as a unique array, i.e. a strict monotonic increasing array.

    Parameters
    ----------
    ielem : :class:`int` or array of :class:`int` or :class:`bool`
        Element number(s) or mask.

    Returns
    -------
    dofs : :class:`numpy.ndarray`
        A 1D Array of indices.
    '''

    if isinstance(ielem, numbers.Integral):
      raise NotImplementedError
    elif numeric.isintarray(ielem):
      if ielem.ndim != 1:
        raise IndexError('invalid ielem')
      if len(ielem) == 0:
        return numpy.array([], dtype=int)
      ielem = numpy.unique(ielem)
      if ielem[0] < 0 or ielem[-1] >= self.nelems:
        raise IndexError('ielem out of bounds')
      return numpy.unique(numpy.fromiter(itertools.chain.from_iterable(map(self.get_dofs, ielem)), dtype=int))
    elif numeric.isboolarray(ielem):
      if ielem.shape != (self.nelems,):
        raise IndexError('ielem has invalid shape')
      return self.get_dofs(numpy.where(ielem)[0])
    else:
      raise IndexError('invalid index')

  def get_ndofs(self, ielem: int) -> int:
    '''Return the number of basis functions with support on element ``ielem``.'''

    return len(self.get_dofs(ielem))

  @abc.abstractmethod
  def get_coefficients(self, ielem: int) -> types.frozenarray:
    '''Return an array of coefficients for all basis functions with support on element ``ielem``.

    Parameters
    ----------
    ielem : :class:`int`
        Element number.

    Returns
    -------
    coefficients : :class:`nutils.types.frozenarray`
        Array of coefficients with shape ``(nlocaldofs,)+(degree,)*ndims``,
        where the first axis corresponds to the dofs returned by
        :meth:`get_dofs`.
    '''

    raise NotImplementedError

  def get_coeffshape(self, ielem: int) -> numpy.ndarray:
    '''Return the shape of the array of coefficients for basis functions with support on element ``ielem``.'''

    return numpy.asarray(self.get_coefficients(ielem).shape[1:])

  def f_ndofs(self, index: evaluable.Array) -> evaluable.Array:
    return evaluable.ElemwiseFromCallable(self.get_ndofs, index, dtype=int, shape=())

  def f_dofs(self, index: evaluable.Array) -> evaluable.Array:
    return evaluable.ElemwiseFromCallable(self.get_dofs, index, dtype=int, shape=(self.f_ndofs(index),))

  def f_coefficients(self, index: evaluable.Array) -> evaluable.Array:
    coeffshape = evaluable.ElemwiseFromCallable(self.get_coeffshape, index, dtype=int, shape=self.coords.shape)
    return evaluable.ElemwiseFromCallable(self.get_coefficients, index, dtype=float, shape=(self.f_ndofs(index), *coeffshape))

  def __getitem__(self, index: Any) -> Array:
    if numeric.isintarray(index) and index.ndim == 1 and numpy.all(numpy.greater(numpy.diff(index), 0)):
      return MaskedBasis(self, index)
    elif numeric.isboolarray(index) and index.shape == (self.ndofs,):
      return MaskedBasis(self, numpy.where(index)[0])
    else:
      return super().__getitem__(index)

class PlainBasis(Basis):
  '''A general purpose implementation of a :class:`Basis`.

  Use this class only if there exists no specific implementation of
  :class:`Basis` for the basis at hand.

  Parameters
  ----------
  coefficients : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The coefficients of the basis functions per transform.  The order should
      match the ``transforms`` argument.
  dofs : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The dofs corresponding to the ``coefficients`` argument.
  ndofs : :class:`int`
      The number of basis functions.
  index : :class:`Int`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_coeffs', '_dofs'

  def __init__(self, coefficients: Sequence[numpy.ndarray], dofs: Sequence[numpy.ndarray], ndofs: int, index: Int, coords: Array) -> None:
    self._coeffs = tuple(map(types.frozenarray, coefficients))
    self._dofs = tuple(map(types.frozenarray, dofs))
    assert len(self._coeffs) == len(self._dofs)
    assert all(c.ndim == 1+coords.shape[0] for c in self._coeffs)
    assert all(len(c) == len(d) for c, d in zip(self._coeffs, self._dofs))
    super().__init__(ndofs, len(coefficients), index, coords)

  def __getnewargs__(self) -> Tuple[Tuple[types.frozenarray], Tuple[types.frozenarray], int, Int, Array]:
    return self._coeffs, self._dofs, self.ndofs, self.index, self.coords

  def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if not isinstance(ielem, numbers.Integral):
      return super().get_dofs(ielem)
    return self._dofs[ielem]

  def get_coefficients(self, ielem: int) -> numpy.ndarray:
    return self._coeffs[ielem]

  def f_ndofs(self, index: evaluable.Array) -> evaluable.Array:
    ndofs = numpy.fromiter(map(len, self._dofs), dtype=int, count=len(self._dofs))
    return evaluable.get(types.frozenarray(ndofs, copy=False), 0, index)

  def f_dofs(self, index: evaluable.Array) -> evaluable.Array:
    return evaluable.Elemwise(self._dofs, index, dtype=int)

  def f_coefficients(self, index: evaluable.Array) -> evaluable.Array:
    return evaluable.Elemwise(self._coeffs, index, dtype=float)

class DiscontBasis(Basis):
  '''A discontinuous basis with monotonic increasing dofs.

  Parameters
  ----------
  coefficients : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The coefficients of the basis functions per transform.  The order should
      match the ``transforms`` argument.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_coeffs', '_offsets'

  def __init__(self, coefficients: Sequence[numpy.ndarray], index: Int, coords: Array) -> None:
    self._coeffs = tuple(map(types.frozenarray, coefficients))
    assert all(c.ndim == 1+coords.shape[0] for c in self._coeffs)
    self._offsets = types.frozenarray(numpy.cumsum([0, *map(len, self._coeffs)]), copy=False)
    super().__init__(self._offsets[-1], len(coefficients), index, coords)

  def __getnewargs__(self) -> Tuple[Tuple[types.frozenarray], Int, Array]:
    return self._coeffs, self.index, self.coords

  def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if not isinstance(dof, numbers.Integral):
      return super().get_support(dof)
    ielem = numpy.searchsorted(self._offsets[:-1], numeric.normdim(self.ndofs, dof), side='right')-1
    return numpy.array([ielem], dtype=int)

  def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if not isinstance(ielem, numbers.Integral):
      return super().get_dofs(ielem)
    ielem = numeric.normdim(self.nelems, ielem)
    return numpy.arange(self._offsets[ielem], self._offsets[ielem+1])

  def get_ndofs(self, ielem: int) -> int:
    return self._offsets[ielem+1] - self._offsets[ielem]

  def get_coefficients(self, ielem: int) -> types.frozenarray:
    return self._coeffs[ielem]

  def f_ndofs(self, index: evaluable.Array) -> evaluable.Array:
    ndofs = numpy.diff(self._offsets)
    return evaluable.get(types.frozenarray(ndofs, copy=False), 0, index)

  def f_dofs(self, index: evaluable.Array) -> evaluable.Array:
    return evaluable.Range(self.f_ndofs(index), offset=evaluable.get(self._offsets, 0, index))

  def f_coefficients(self, index: evaluable.Array) -> evaluable.Array:
    return evaluable.Elemwise(self._coeffs, index, dtype=float)

class MaskedBasis(Basis):
  '''An order preserving subset of another :class:`Basis`.

  Parameters
  ----------
  parent : :class:`Basis`
      The basis to mask.
  indices : array of :class:`int`\\s
      The strict monotonic increasing indices of ``parent`` basis functions to
      keep.
  '''

  __slots__ = '_parent', '_indices'

  def __init__(self, parent: Basis, indices: numpy.ndarray) -> None:
    indices = types.frozenarray(indices)
    if indices.ndim != 1:
      raise ValueError('`indices` should have one dimension but got {}'.format(indices.ndim))
    if len(indices) and not numpy.all(numpy.greater(numpy.diff(indices), 0)):
      raise ValueError('`indices` should be strictly monotonic increasing')
    if len(indices) and (indices[0] < 0 or indices[-1] >= len(parent)):
      raise ValueError('`indices` out of range \x5b0,{}\x29'.format(len(parent)))
    self._parent = parent
    self._indices = indices
    super().__init__(len(self._indices), parent.nelems, parent.index, parent.coords)

  def __getnewargs__(self) -> Tuple[Basis, types.frozenarray]:
    return self._parent, self._indices

  def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
    return numeric.sorted_index(self._indices, self._parent.get_dofs(ielem), missing='mask')

  def get_coeffshape(self, ielem: int) -> numpy.ndarray:
    return self._parent.get_coeffshape(ielem)

  def get_coefficients(self, ielem: int) -> numpy.ndarray:
    mask = numeric.sorted_contains(self._indices, self._parent.get_dofs(ielem))
    return self._parent.get_coefficients(ielem)[mask]

  def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return self._parent.get_support(self._indices[dof])

class StructuredBasis(Basis):
  '''A basis for class:`nutils.transformseq.StructuredTransforms`.

  Parameters
  ----------
  coeffs : :class:`tuple` of :class:`tuple`\\s of arrays
      Per dimension the coefficients of the basis functions per transform.
  start_dofs : :class:`tuple` of arrays of :class:`int`\\s
      Per dimension the dof of the first entry in ``coeffs`` per transform.
  stop_dofs : :class:`tuple` of arrays of :class:`int`\\s
      Per dimension one plus the dof of the last entry  in ``coeffs`` per
      transform.
  dofs_shape : :class:`tuple` of :class:`int`\\s
      The tensor shape of the dofs.
  transforms_shape : :class:`tuple` of :class:`int`\\s
      The tensor shape of the transforms.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_coeffs', '_start_dofs', '_stop_dofs', '_dofs_shape', '_transforms_shape'

  def __init__(self, coeffs: Sequence[Sequence[numpy.ndarray]], start_dofs: Sequence[numpy.ndarray], stop_dofs: Sequence[numpy.ndarray], dofs_shape: Sequence[int], transforms_shape: Sequence[int], index: Int, coords: Array) -> None:
    self._coeffs = tuple(tuple(types.frozenarray(a) for a in b) for b in coeffs)
    self._start_dofs = tuple(map(types.frozenarray, start_dofs))
    self._stop_dofs = tuple(map(types.frozenarray, stop_dofs))
    self._dofs_shape = tuple(map(int, dofs_shape))
    self._transforms_shape = tuple(map(int, transforms_shape))
    super().__init__(util.product(dofs_shape), util.product(transforms_shape), index, coords)

  def __getnewargs__(self) -> Tuple[Tuple[Tuple[types.frozenarray, ...], ...], Tuple[types.frozenarray, ...], Tuple[types.frozenarray, ...], Tuple[int, ...], Tuple[int, ...], Int, Array]:
    return self._coeffs, self._start_dofs, self._stop_dofs, self._dofs_shape, self._transforms_shape, self.index, self.coords

  def _get_indices(self, ielem: int) -> Tuple[int, ...]:
    ielem = numeric.normdim(self.nelems, ielem)
    indices = [] # type: List[int]
    for n in reversed(self._transforms_shape):
      ielem, index = divmod(ielem, n)
      indices.insert(0, index)
    if ielem != 0:
      raise IndexError
    return tuple(indices)

  def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if not isinstance(ielem, numbers.Integral):
      return super().get_dofs(ielem)
    indices = self._get_indices(ielem)
    dofs = numpy.array(0)
    for start_dofs_i, stop_dofs_i, ndofs_i, index_i in zip(self._start_dofs, self._stop_dofs, self._dofs_shape, indices):
      dofs_i = numpy.arange(start_dofs_i[index_i], stop_dofs_i[index_i], dtype=int) % ndofs_i
      dofs = numpy.add.outer(dofs*ndofs_i, dofs_i)
    return types.frozenarray(dofs.ravel(), dtype=types.strictint, copy=False)

  def get_ndofs(self, ielem: int) -> int:
    indices = self._get_indices(ielem)
    ndofs = 1
    for start_dofs_i, stop_dofs_i, index_i in zip(self._start_dofs, self._stop_dofs, indices):
      ndofs *= stop_dofs_i[index_i] - start_dofs_i[index_i]
    return ndofs

  def get_coefficients(self, ielem: int) -> types.frozenarray:
    return functools.reduce(numeric.poly_outer_product, map(operator.getitem, self._coeffs, self._get_indices(ielem)))

  def f_coefficients(self, index: evaluable.Array) -> evaluable.Array:
    coeffs = []
    for coeffs_i in self._coeffs:
      if any(coeffs_ij != coeffs_i[0] for coeffs_ij in coeffs_i[1:]):
        return super().f_coefficients(index)
      coeffs.append(coeffs_i[0])
    return evaluable.Constant(functools.reduce(numeric.poly_outer_product, coeffs))

  def f_ndofs(self, index: evaluable.Array) -> evaluable.Array:
    ndofs = 1
    for start_dofs_i, stop_dofs_i in zip(self._start_dofs, self._stop_dofs):
      ndofs_i = stop_dofs_i - start_dofs_i
      if any(ndofs_ij != ndofs_i[0] for ndofs_ij in ndofs_i[1:]):
        return super().f_ndofs(index)
      ndofs *= ndofs_i[0]
    return evaluable.Constant(ndofs)

  def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if not isinstance(dof, numbers.Integral):
      return super().get_support(dof)
    dof = numeric.normdim(self.ndofs, dof)
    ndofs = 1
    ntrans = 1
    supports = []
    for start_dofs_i, stop_dofs_i, ndofs_i, ntrans_i in zip(reversed(self._start_dofs), reversed(self._stop_dofs), reversed(self._dofs_shape), reversed(self._transforms_shape)):
      dof, dof_i = divmod(dof, ndofs_i)
      supports_i = []
      while dof_i < stop_dofs_i[-1]:
        stop_ielem = numpy.searchsorted(start_dofs_i, dof_i, side='right')
        start_ielem = numpy.searchsorted(stop_dofs_i, dof_i, side='right')
        supports_i.append(numpy.arange(start_ielem, stop_ielem, dtype=int))
        dof_i += ndofs_i
      supports.append(numpy.unique(numpy.concatenate(supports_i)) * ntrans)
      ndofs *= ndofs_i
      ntrans *= ntrans_i
    assert dof == 0
    return types.frozenarray(functools.reduce(numpy.add.outer, reversed(supports)).ravel(), copy=False, dtype=types.strictint)

class PrunedBasis(Basis):
  '''A subset of another :class:`Basis`.

  Parameters
  ----------
  parent : :class:`Basis`
      The basis to prune.
  transmap : one-dimensional array of :class:`int`\\s
      The indices of transforms in ``parent`` that form this subset.
  index : :class:`Array`
      The element index.
  coords : :class:`Array`
      The element local coordinates.
  '''

  __slots__ = '_parent', '_transmap', '_dofmap'

  def __init__(self, parent: Basis, transmap: numpy.ndarray, index: Int, coords: Array) -> None:
    self._parent = parent
    self._transmap = types.frozenarray(transmap)
    self._dofmap = parent.get_dofs(self._transmap)
    super().__init__(len(self._dofmap), len(transmap), index, coords)

  def __getnewargs__(self) -> Tuple[Basis, types.frozenarray, Int, Array]:
    return self._parent, self._transmap, self.index, self.coords

  def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if numeric.isintarray(ielem) and ielem.ndim == 1 and numpy.any(numpy.less(ielem, 0)):
      raise IndexError('dof out of bounds')
    return types.frozenarray(numpy.searchsorted(self._dofmap, self._parent.get_dofs(self._transmap[ielem])), copy=False)

  def get_coefficients(self, ielem: int) -> types.frozenarray:
    return self._parent.get_coefficients(self._transmap[ielem])

  def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return numeric.sorted_index(self._transmap, self._parent.get_support(self._dofmap[dof]), missing='mask')

  def f_ndofs(self, index: evaluable.Array) -> evaluable.Array:
    return self._parent.f_ndofs(evaluable.get(self._transmap, 0, index))

  def f_coefficients(self, index: evaluable.Array) -> evaluable.Array:
    return self._parent.f_coefficients(evaluable.get(self._transmap, 0, index))

# NAMESPACE

def _eval_ast(ast, functions):
  '''evaluate ``ast`` generated by :func:`nutils.expression.parse`'''

  op, *args = ast
  if op is None:
    value, = args
    return value

  args = (_eval_ast(arg, functions) for arg in args)
  if op == 'group':
    array, = args
    return array
  elif op == 'arg':
    name, *shape = args
    return Argument(name, shape)
  elif op == 'substitute':
    array, *arg_value_pairs = args
    subs = {}
    assert len(arg_value_pairs) % 2 == 0
    for arg, value in zip(arg_value_pairs[0::2], arg_value_pairs[1::2]):
      assert arg.name not in subs
      subs[arg.name] = value
    return replace_arguments(array, subs)
  elif op == 'call':
    func, generates, consumes, *args = args
    args = tuple(map(Array.cast, args))
    kwargs = {}
    if generates:
      kwargs['generates'] = generates
    if consumes:
      kwargs['consumes'] = consumes
    result = functions[func](*args, **kwargs)
    shape = builtins.sum((arg.shape[:arg.ndim-consumes] for arg in args), ())
    if result.ndim != len(shape) + generates or result.shape[:len(shape)] != shape:
      raise ValueError('expected an array with shape {} and {} additional axes when calling {} but got {}'.format(shape, generates, func, result.shape))
    return result
  elif op == 'jacobian':
    geom, ndims = args
    return J(geom, ndims)
  elif op == 'eye':
    length, = args
    return eye(length)
  elif op == 'normal':
    geom, = args
    return normal(geom)
  elif op == 'getitem':
    array, dim, index = args
    return get(array, dim, index)
  elif op == 'trace':
    array, n1, n2 = args
    return trace(array, n1, n2)
  elif op == 'sum':
    array, axis = args
    return sum(array, axis)
  elif op == 'concatenate':
    return concatenate(args, axis=0)
  elif op == 'grad':
    array, geom = args
    return grad(array, geom)
  elif op == 'surfgrad':
    array, geom = args
    return grad(array, geom, len(geom)-1)
  elif op == 'derivative':
    func, target = args
    return derivative(func, target)
  elif op == 'append_axis':
    array, length = args
    return insertaxis(array, -1, length)
  elif op == 'transpose':
    array, trans = args
    return transpose(array, trans)
  elif op == 'jump':
    array, = args
    return jump(array)
  elif op == 'mean':
    array, = args
    return mean(array)
  elif op == 'neg':
    array, = args
    return -Array.cast(array)
  elif op in ('add', 'sub', 'mul', 'truediv', 'pow'):
    left, right = args
    return getattr(operator, '__{}__'.format(op))(Array.cast(left), Array.cast(right))
  else:
    raise ValueError('unknown opcode: {!r}'.format(op))

def _sum_expr(arg: Array, *, consumes:int = 0) -> Array:
  if consumes == 0:
    raise ValueError('sum must consume at least one axis but got zero')
  return sum(arg, range(arg.ndim-consumes, arg.ndim))

def _norm2_expr(arg: Array, *, consumes: int = 0) -> Array:
  if consumes == 0:
    raise ValueError('sum must consume at least one axis but got zero')
  return norm2(arg, range(arg.ndim-consumes, arg.ndim))

def _J_expr(geom: Array, *, consumes: int = 0) -> Array:
  if consumes != 1:
    raise ValueError('J consumes exactly one axis but got {}'.format(consumes))
  return J(geom)

def _arctan2_expr(_a: Array, _b: Array) -> Array:
  a = Array.cast(_a)
  b = Array.cast(_b)
  return arctan2(_append_axes(a, b.shape), _prepend_axes(b, a.shape))

class Namespace:
  '''Namespace for :class:`Array` objects supporting assignments with tensor expressions.

  The :class:`Namespace` object is used to store :class:`Array` objects.

  >>> from nutils import function
  >>> ns = function.Namespace()
  >>> ns.A = function.zeros([3, 3])
  >>> ns.x = function.zeros([3])
  >>> ns.c = 2

  In addition to the assignment of :class:`Array` objects, it is also possible
  to specify an array using a tensor expression string — see
  :func:`nutils.expression.parse` for the syntax.  All attributes defined in
  this namespace are available as variables in the expression.  If the array
  defined by the expression has one or more dimensions the indices of the axes
  should be appended to the attribute name.  Examples:

  >>> ns.cAx_i = 'c A_ij x_j'
  >>> ns.xAx = 'x_i A_ij x_j'

  It is also possible to simply evaluate an expression without storing its
  value in the namespace by passing the expression to the method ``eval_``
  suffixed with appropriate indices:

  >>> ns.eval_('2 c')
  Array<>
  >>> ns.eval_i('c A_ij x_j')
  Array<3>
  >>> ns.eval_ij('A_ij + A_ji')
  Array<3,3>

  For zero and one dimensional expressions the following shorthand can be used:

  >>> '2 c' @ ns
  Array<>
  >>> 'A_ij x_j' @ ns
  Array<3>

  Sometimes the dimension of an expression cannot be determined, e.g. when
  evaluating the identity array:

  >>> ns.eval_ij('δ_ij')
  Traceback (most recent call last):
  ...
  nutils.expression.ExpressionSyntaxError: Length of axis cannot be determined from the expression.
  δ_ij
    ^

  There are two ways to inform the namespace of the correct lengths.  The first is to
  assign fixed lengths to certain indices via keyword argument ``length_<indices>``:

  >>> ns_fixed = function.Namespace(length_ij=2)
  >>> ns_fixed.eval_ij('δ_ij')
  Array<2,2>

  Note that evaluating an expression with an incompatible length raises an
  exception:

  >>> ns = function.Namespace(length_i=2)
  >>> ns.a = numpy.array([1,2,3])
  >>> 'a_i' @ ns
  Traceback (most recent call last):
  ...
  nutils.expression.ExpressionSyntaxError: Length of index i is fixed at 2 but the expression has length 3.
  a_i
    ^

  The second is to define a fallback length via the ``fallback_length`` argument:

  >>> ns_fallback = function.Namespace(fallback_length=2)
  >>> ns_fallback.eval_ij('δ_ij')
  Array<2,2>

  When evaluating an expression through this namespace the following functions
  are available: ``opposite``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan2``, ``arctanh``, ``exp``, ``abs``,
  ``ln``, ``log``, ``log2``, ``log10``, ``sqrt`` and ``sign``.

  Additional pointwise functions can be passed to argument ``functions``. All
  functions should take :class:`Array` objects as arguments and must return an
  :class:`Array` with as shape the sum of all shapes of the arguments.

  >>> def sqr(a):
  ...   return a**2
  >>> def mul(a, b):
  ...   return a[(...,)+(None,)*b.ndim] * b[(None,)*a.ndim]
  >>> ns_funcs = function.Namespace(functions=dict(sqr=sqr, mul=mul))
  >>> ns_funcs.a = numpy.array([1,2,3])
  >>> ns_funcs.b = numpy.array([4,5])
  >>> 'sqr(a_i)' @ ns_funcs # same as 'a_i^2'
  Array<3>
  >>> ns_funcs.eval_ij('mul(a_i, b_j)') # same as 'a_i b_j'
  Array<3,2>
  >>> 'mul(a_i, a_i)' @ ns_funcs # same as 'a_i a_i'
  Array<>

  Args
  ----
  default_geometry_name : :class:`str`
      The name of the default geometry.  This argument is passed to
      :func:`nutils.expression.parse`.  Default: ``'x'``.
  fallback_length : :class:`int`, optional
      The fallback length of an axis if the length cannot be determined from
      the expression.
  length_<indices> : :class:`int`
      The fixed length of ``<indices>``.  All axes in the expression marked
      with one of the ``<indices>`` are asserted to have the specified length.
  functions : :class:`dict`, optional
      Pointwise functions that should be available in the namespace,
      supplementing the default functions listed above. All functions should
      return arrays with as shape the sum of all shapes of the arguments.

  Attributes
  ----------
  arg_shapes : :class:`dict`
      A readonly map of argument names and shapes.
  default_geometry_name : :class:`str`
      The name of the default geometry.  See argument with the same name.
  '''

  __slots__ = '_attributes', '_arg_shapes', 'default_geometry_name', '_fixed_lengths', '_fallback_length', '_functions'

  _re_assign = re.compile('^([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-z]+)?$')

  _default_functions = dict(
    opposite=opposite, sin=sin, cos=cos, tan=tan, sinh=sinh, cosh=cosh,
    tanh=tanh, arcsin=arcsin, arccos=arccos, arctan=arctan, arctan2=_arctan2_expr, arctanh=arctanh,
    exp=exp, abs=abs, ln=ln, log=ln, log2=log2, log10=log10, sqrt=sqrt,
    sign=sign, d=d, surfgrad=surfgrad, n=normal,
    sum=_sum_expr, norm2=_norm2_expr, J=_J_expr,
  )

  def __init__(self, *, default_geometry_name: str = 'x', fallback_length: Optional[int] = None, functions: Optional[Mapping[str, Callable]] = None, **kwargs: Any) -> None:
    if not isinstance(default_geometry_name, str):
      raise ValueError('default_geometry_name: Expected a str, got {!r}.'.format(default_geometry_name))
    if '_' in default_geometry_name or not self._re_assign.match(default_geometry_name):
      raise ValueError('default_geometry_name: Invalid variable name: {!r}.'.format(default_geometry_name))
    fixed_lengths = {}
    for name, value in kwargs.items():
      if not name.startswith('length_'):
        raise TypeError('__init__() got an unexpected keyword argument {!r}'.format(name))
      for index in name[7:]:
        if index in fixed_lengths:
          raise ValueError('length of index {} specified more than once'.format(index))
        fixed_lengths[index] = value
    super().__setattr__('_attributes', {})
    super().__setattr__('_arg_shapes', {})
    super().__setattr__('_fixed_lengths', types.frozendict({i: l for indices, l in fixed_lengths.items() for i in indices} if fixed_lengths else {}))
    super().__setattr__('_fallback_length', fallback_length)
    super().__setattr__('default_geometry_name', default_geometry_name)
    super().__setattr__('_functions', dict(itertools.chain(self._default_functions.items(), () if functions is None else functions.items())))
    super().__init__()

  def __getstate__(self) -> Dict[str, Any]:
    'Pickle instructions'
    attrs = '_arg_shapes', '_attributes', 'default_geometry_name', '_fixed_lengths', '_fallback_length', '_functions'
    return {k: getattr(self, k) for k in attrs}

  def __setstate__(self, d: Mapping[str, Any]) -> None:
    'Unpickle instructions'
    for k, v in d.items(): super().__setattr__(k, v)

  @property
  def arg_shapes(self) -> Mapping[str, Shape]:
    return builtin_types.MappingProxyType(self._arg_shapes)

  @property
  def default_geometry(self) -> str:
    ''':class:`nutils.function.Array`: The default geometry, shorthand for ``getattr(ns, ns.default_geometry_name)``.'''
    return getattr(self, self.default_geometry_name)

  def __call__(*args, **subs: IntoArray) -> 'Namespace':
    '''Return a copy with arguments replaced by ``subs``.

    Return a copy of this namespace with :class:`Argument` objects replaced
    according to ``subs``.

    Args
    ----
    **subs : :class:`dict` of :class:`str` and :class:`nutils.function.Array` objects
        Replacements of the :class:`Argument` objects, identified by their names.

    Returns
    -------
    ns : :class:`Namespace`
        The copy of this namespace with replaced :class:`Argument` objects.
    '''

    if len(args) != 1:
      raise TypeError('{} instance takes 1 positional argument but {} were given'.format(type(args[0]).__name__, len(args)))
    self, = args
    ns = Namespace(default_geometry_name=self.default_geometry_name)
    for k, v in self._attributes.items():
      setattr(ns, k, replace_arguments(v, subs))
    return ns

  def copy_(self, *, default_geometry_name: Optional[str] = None) -> 'Namespace':
    '''Return a copy of this namespace.'''

    if default_geometry_name is None:
      default_geometry_name = self.default_geometry_name
    ns = Namespace(default_geometry_name=default_geometry_name, fallback_length=self._fallback_length, functions=self._functions, **{'length_{i}': l for i, l in self._fixed_lengths.items()})
    for k, v in self._attributes.items():
      setattr(ns, k, v)
    return ns

  def __getattr__(self, name: str) -> Any:
    '''Get attribute ``name``.'''

    if name.startswith('eval_'):
      return lambda expr: _eval_ast(expression.parse(expr, variables=self._attributes, indices=name[5:], arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0], self._functions)
    try:
      return self._attributes[name]
    except KeyError:
      pass
    raise AttributeError(name)

  def __setattr__(self, name: str, value: Any) -> Any:
    '''Set attribute ``name`` to ``value``.'''

    if name in self.__slots__:
      raise AttributeError('readonly')
    m = self._re_assign.match(name)
    if not m or m.group(2) and len(set(m.group(2))) != len(m.group(2)):
      raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))
    else:
      name, indices = m.groups()
      indices = indices[1:] if indices else None
      if isinstance(value, str):
        ast, arg_shapes = expression.parse(value, variables=self._attributes, indices=indices, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)
        value = _eval_ast(ast, self._functions)
        self._arg_shapes.update(arg_shapes)
      else:
        assert not indices
      self._attributes[name] = Array.cast(value)

  def __delattr__(self, name: str) -> None:
    '''Delete attribute ``name``.'''

    if name in self.__slots__:
      raise AttributeError('readonly')
    elif name in self._attributes:
      del self._attributes[name]
    else:
      raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))

  @overload
  def __rmatmul__(self, expr: str) -> Array: ...
  @overload
  def __rmatmul__(self, expr: Union[Tuple[str, ...], List[str]]) -> Tuple[Array, ...]: ...
  def __rmatmul__(self, expr: Union[str, Tuple[str, ...], List[str]]) -> Union[Array, Tuple[Array, ...]]:
    '''Evaluate zero or one dimensional ``expr`` or a list of expressions.'''

    if isinstance(expr, (tuple, list)):
      return tuple(map(self.__rmatmul__, expr))
    if not isinstance(expr, str):
      return NotImplemented
    try:
      ast = expression.parse(expr, variables=self._attributes, indices=None, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0]
    except expression.AmbiguousAlignmentError:
      raise ValueError('`expression @ Namespace` cannot be used because the expression has more than one dimension.  Use `Namespace.eval_...(expression)` instead')
    return _eval_ast(ast, self._functions)
