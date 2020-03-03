"""Provides Numba type, FunctionType, that makes functions as
instances of a first-class function type.
"""

# import types as pytypes
import numba
from numba import types as nbtypes
from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (
    FunctionType, FunctionPrototype, WrapperAddressProtocol)
from numba.core.dispatcher import Dispatcher


@typeof_impl.register(WrapperAddressProtocol)
@typeof_impl.register(CFunc)
@typeof_impl.register(Dispatcher)
def typeof_function_type(val, c):
    return FunctionType.fromobject(val)


# TODO: typeof_impl for types.FunctionType, ctypes.CFUNCTYPE


@register_model(FunctionPrototype)
class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """
    def __init__(self, dmm, fe_type):
        if isinstance(fe_type, FunctionType):
            ftype = fe_type.ftype
        elif isinstance(fe_type, FunctionPrototype):
            ftype = fe_type
        else:
            raise NotImplementedError((type(fe_type)))
        retty = dmm.lookup(ftype.rtype).get_value_type()
        args = [dmm.lookup(t).get_value_type() for t in ftype.atypes]
        be_type = ir.PointerType(ir.FunctionType(retty, args))
        super(FunctionProtoModel, self).__init__(dmm, fe_type, be_type)


@register_model(FunctionType)
class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """
    def __init__(self, dmm, fe_type):
        members = [
            # address of cfunc wrapper function:
            ('addr', nbtypes.voidptr),
            # address of PyObject* referencing the Python function
            # object:
            ('pyaddr', nbtypes.voidptr),
        ]
        super(FunctionModel, self).__init__(dmm, fe_type, members)


@lower_constant(FunctionType)
def lower_constant_function_type(context, builder, typ, pyval):
    if isinstance(pyval, CFunc):
        addr = pyval._wrapper_address
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        sfunc.addr = context.add_dynamic_addr(builder, addr,
                                              info=str(typ))
        sfunc.pyaddr = context.add_dynamic_addr(builder, id(pyval),
                                                info=type(pyval).__name__)
        return sfunc._getvalue()
    if isinstance(pyval, Dispatcher):
        cres = pyval.get_compile_result(typ.signature(), compile=True)
        if cres is None:
            # TODO: raise exception as compilation failed (unless
            # compile is disabled). Set compile=False to reproduce.
            addr = -1
        else:
            wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
            addr = cres.library.get_pointer_to_function(wrapper_name)
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        sfunc.addr = context.add_dynamic_addr(builder, addr,
                                              info=str(typ))
        sfunc.pyaddr = context.add_dynamic_addr(builder, id(pyval),
                                                info=type(pyval).__name__)
        return sfunc._getvalue()
    # TODO: implement support for WrapperAddressProtocol,
    # and types.FunctionType, ctypes.CFUNCTYPE

    raise NotImplementedError(
        'lower_constant_struct_function_type({}, {}, {}, {})'
        .format(context, builder, typ, pyval))


def _get_wrapper_address(func, sig):
    """Return the address of a compiled function that implements `func`.

    Warning: The compiled function must be compatible with the given
    signature `sig`. If it is not, then result of calling the compiled
    function is undefined. The compatibility is ensured when passing
    in a first-class function to a Numba njit compiled function either
    as an argument or via namespace scoping.

    Parameters
    ----------
    func : object
      A function object that has been numba.cfunc decorated or an
      object that implements the wrapper address protocol (see note
      below).  numba.cfunc(sig) is applied to pure Python function
      inputs and the source of numba.cfunc decorated functions in case
      the signature of `func` and `sig` do not match.
    sig : Signature
      The function signature.

    Returns
    -------
    addr : int
      An address in memory (pointer value) of the compiled function
      corresponding to the specified signature.

    Note: wrapper address protocol
    ------------------------------

    An object implements the wrapper address protocol iff the object
    provides a callable attribute named __wrapper_address__ that takes
    a Signature instance as the argument, and returns an integer
    representing the address or pointer value of a compiled function
    for the given signature.

    """
    print(f'_get_wrapper_address[{func}]({sig=})')
    if sig.return_type == types.undefined:
        # addr==-1 indicates that no implementation is available for
        # cases where automatic type-inference was unsuccesful. For
        # example, the type of unused jit-decorated function arguments
        # is unknown but also unneeded.
        addr = -1
    elif hasattr(func, '__wrapper_address__'):
        # func can be any object that implements the
        # __wrapper_address__ protocol.
        addr = func.__wrapper_address__()
    elif isinstance(func, types.FunctionType):
        cfunc = numba.cfunc(sig)(func)
        addr = cfunc._wrapper_address
    elif isinstance(func, CFunc):
        if sig == func._sig:
            addr = func._wrapper_address
        else:
            # TODO: remove?
            cfunc = numba.cfunc(sig)(func._pyfunc)
            addr = cfunc._wrapper_address
    elif isinstance(func, Dispatcher):
        cres = func.get_compile_result(sig, compile=True)
        if cres is None:
            addr = 0
        else:
            wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
            addr = cres.library.get_pointer_to_function(wrapper_name)
            print(f'{wrapper_name=}')
    else:
        raise NotImplementedError(
            f'get wrapper address of {type(func)} instance with {sig!r}')
    if not isinstance(addr, int):
        raise TypeError(
            f'wrapper address must be integer, got {type(addr)} instance')
    if addr <= 0 and addr != -1:
        raise ValueError(f'wrapper address of {type(func)} instance must be'
                         f' a positive integer but got {addr}')
    print(f'{addr=}')
    #raise
    return addr


@unbox(FunctionType)
def unbox_function_type(typ, obj, c):
    sig = typ.signature()
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # Get the obj wrapper address. The code below trusts that the
    # function numba.function._get_wrapper_address exists and can be
    # called with two arguments. However, if an exception is raised in
    # the function, then it will be caught and propagated to the
    # caller.
    modname = c.context.insert_const_string(c.builder.module, __name__)
    numba_mod = c.pyapi.import_module_noblock(modname)
    numba_func = c.pyapi.object_getattr_string(
        numba_mod, '_get_wrapper_address')
    c.pyapi.decref(numba_mod)

    sig_obj = c.pyapi.unserialize(c.pyapi.serialize_object(sig))
    addr = c.pyapi.call_function_objargs(numba_func, (obj, sig_obj))

    with c.builder.if_else(cgutils.is_null(c.builder, addr),
                           likely=False) as (then, orelse):
        with then:
            # propagate any errors in getting pyaddr to the caller
            c.builder.ret(c.pyapi.get_null_object())
        with orelse:
            # _get_wrapper_address checks that pyaddr is int and
            # nonzero, so no need to check it here. But it will be
            # impossible to tell if the addr value actually
            # corresponds to a memory location of a valid function.
            sfunc.addr = c.pyapi.long_as_voidptr(addr)
            c.pyapi.decref(addr)

            llty = c.context.get_value_type(nbtypes.voidptr)
            sfunc.pyaddr = c.builder.ptrtoint(obj, llty)

    return NativeValue(sfunc._getvalue())


@box(FunctionType)
def box_function_type(typ, val, c):
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    # TODO: reconsider the boxing model, see the limitations in the
    # unbox function above.
    pyaddr_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    raw_ptr = c.builder.inttoptr(sfunc.pyaddr, c.pyapi.pyobj)
    with c.builder.if_then(cgutils.is_null(c.builder, raw_ptr),
                           likely=False):
        cstr = f"first-class function {typ} parent object not set"
        c.pyapi.err_set_string("PyExc_MemoryError", cstr)
        c.builder.ret(c.pyapi.get_null_object())
    c.builder.store(raw_ptr, pyaddr_ptr)
    cfunc = c.builder.load(pyaddr_ptr)
    c.pyapi.incref(cfunc)
    return cfunc
