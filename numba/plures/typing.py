from xnd import xnd
from ndtypes import ndt
from llvmlite import ir

from ..types import Type, int64
from ..decorators import jit
from ..extending import typeof_impl
from ..extending import models, register_model
from ..extending import unbox, box, NativeValue, lower_getattr
from ..typing.templates import AttributeTemplate, infer_getattr


from .llvm import xnd_t, ptr, i8, i32, i64, context, index, ndt_t


class NDTType(Type):
    def __init__(self):
        super().__init__(name="ndt")


class XndType(Type):
    def __init__(self):
        super().__init__(name="xnd")


xnd_type = XndType()
ndt_type = NDTType()


@infer_getattr
class XndAttribute(AttributeTemplate):
    key = XndType

    def resolve_is_valid(self, ty):
        return int64

    def resolve_err_occurred(self, ty):
        return int64

    def resolve_type(self, ty):
        return ndt_type


@typeof_impl.register(xnd)
def typeof_xnd(val, c):
    return xnd_type


@typeof_impl.register(ndt)
def typeof_ndt(val, c):
    return ndt_type


@register_model(NDTType)
class NDTModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ptr(ndt_t)
        super().__init__(dmm, fe_type, be_type)


@register_model(XndType)
class XndModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ptr(xnd_t)
        super().__init__(dmm, fe_type, be_type)


@lower_getattr(XndType, "is_valid")
def xnd_is_valid_value(context, builder, ty, val):
    xnd_ndim = builder.module.get_or_insert_function(
        ir.FunctionType(i64, [ptr(xnd_t)]), name="xnd_is_valid"
    )
    return builder.call(xnd_ndim, [val])


@lower_getattr(XndType, "err_occurred")
def xnd_err_occurred_value(context, builder, ty, val):
    xnd_ndim = builder.module.get_or_insert_function(
        ir.FunctionType(i64, [ptr(xnd_t)]), name="xnd_err_occurred"
    )
    return builder.call(xnd_ndim, [val])


@lower_getattr(XndType, "type")
def xnd_type_value(context, builder, ty, val):
    return builder.load(builder.gep(val, [index(0), index(2)], True))


@unbox(XndType)
def unbox_xnd(typ, obj, c):
    """
    Convert a xnd object to a native xnd_t ptr.
    """
    const_xnd = pycapsule_import(
        c,
        "xnd._xnd._API",
        2,
        ir.FunctionType(ptr(xnd_t), [c.pyapi.pyobj]),
        name="const_xnd",
    )

    return NativeValue(c.builder.call(const_xnd, [obj]))


@unbox(NDTType)
def unbox_ndt(typ, obj, c):
    """
    Convert a ndt object to a native ndt_t ptr.
    """
    const_ndt = pycapsule_import(
        c,
        "ndtypes._ndtypes._API",
        2,
        ir.FunctionType(ptr(ndt_t), [c.pyapi.pyobj]),
        name="const_ndt",
    )

    return NativeValue(c.builder.call(const_ndt, [obj]))


@box(XndType)
def box_xnd(typ, val, c):
    """
    Convert a native ptr(xnd_t) structure to a xnd object.
    """
    xnd_from_xnd = pycapsule_import(
        c,
        "xnd._xnd._API",
        5,
        ir.FunctionType(c.pyapi.pyobj, [c.pyapi.pyobj, ptr(xnd_t)]),
        name="xnd_from_xnd",
    )

    xnd_type = c.pyapi.unserialize(c.pyapi.serialize_object(xnd))
    res = c.builder.call(xnd_from_xnd, [xnd_type, val])
    c.pyapi.decref(xnd_type)
    return res


@box(NDTType)
def box_ndt(typ, val, c):
    """
    Convert a native ptr(ndt_t) structure to a ndt object.
    """
    ndt_from_type = pycapsule_import(
        c,
        "ndtypes._ndtypes._API",
        6,
        ir.FunctionType(c.pyapi.pyobj, [ptr(ndt_t)]),
        name="ndt_from_type",
    )

    res = c.builder.call(ndt_from_type, [val])
    return res


def pycapsule_import(c, path, i: int, fntype, name=None):
    """
    Get's a function stored in the PyCapsule at path `path`, at index `i` with type `fntype`.

    This is based on the LLVM outputted by clang for doing this:
    https://gist.github.com/saulshanabrook/8467b10c97cc0a76ae0bcdf95a8ca478
    """
    builder = c.builder
    capsule_import = c.pyapi._get_function(
        ir.FunctionType(ptr(i8), [c.pyapi.cstring, i32]), name="PyCapsule_Import"
    )
    api_string = c.pyapi.context.insert_const_string(c.pyapi.module, path)
    xnd_api = builder.call(capsule_import, [api_string, ir.Constant(i32, 0)])

    return builder.load(
        builder.bitcast(builder.gep(xnd_api, [index(i * 8)], True), ptr(ptr(fntype))),
        name=name,
    )

