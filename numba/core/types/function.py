
__all__ = ['FunctionType', 'FunctionPrototype', 'WrapperAddressProtocol']

from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, utils


class FunctionTypeImplBase:
    """Base class for function type implementations.
    """

    def supports(self, signature):
        """Check if the given signature can be supported by the first-class
        function type support.
        """
        print(f'{type(self).__name__}:supports({signature=})')
        #return False
        sig_types = list(signature.args)
        rtype = signature.return_type
        if rtype is not None:
            sig_types.append(rtype) 
        
        for typ in sig_types:
            if isinstance(typ, types.Optional):
                return False
            #if isinstance(typ, types.FunctionType):
            #    return False
            print(typ, type(typ))
            #if not isinstance(typ, (types.Number, types.FunctionType, types.Array, types.NoneType, types.Tuple)):
            #    return False
        return True

    def unwrap(self):
        """Return type that disables first-class function type support.
        """
        raise NotImplementedError(f'{type(self).__name__}.unwrap()')

    def signature(self):
        """Return the signature of the function type.
        """
        raise NotImplementedError(f'{type(self).__name__}.signature')

    def signatures(self):
        """Generator of the function type signatures.
        """
        yield self.signature

    def check_signature(self, sig, compile=False):
        """Check if the given signature belongs to the function type.
        """
        for signature in self.signatures():
            if sig == signature:
                return True
            print('AAAAAA', sig, signature)
        return False

    def get_call_type(self, context, args, kws):
        from numba.core import typing
        # use when a new compilation is not feasible
        for signature in self.signatures():
            if len(args) != len(signature.args):
                continue
            for atype, sig_atype in zip(args, signature.args):
                atype = types.unliteral(atype)
                # Get the casting score
                conv_score = context.context.can_convert(
                    fromty=atype, toty=sig_atype
                )
                # Allow safe casts
                if conv_score > typing.context.Conversion.safe:
                    break
            else:
                # TODO: there may be better matches
                return signature
        raise ValueError(
            f'{self} argument types do not match with {args}')


class FunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, signature):
        self.signature = types.unliteral(signature)

    def has_signatures(self):
        return True

    @property
    def name(self):
        return f'{type(self).__name__}'

    def __repr__(self):
        return f'{self.name}[{self.signature}]'


class CFuncFunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, cfunc):
        self.cfunc = cfunc
        self.signature = types.unliteral(self.cfunc._sig)

    def has_signatures(self):
        return True

    @property
    def name(self):
        return f'{type(self).__name__}({self.cfunc._pyfunc.__name__})'

    def __repr__(self):
        return f'{self.name}[{self.signature}]'


class DispatcherFunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        assert len(self.dispatcher.overloads) <= 1, (self.dispatcher.py_func.__name__, len(self.dispatcher.overloads),)

    @property
    def signature(self):
        # return the signature of the last compile result
        for cres in reversed(self.dispatcher.overloads.values()):
            return types.unliteral(cres.signature)
        # return something that is a signature
        mn, mx = utils.get_nargs_range(self.dispatcher.py_func)
        #return types.unknown(*((types.unknown,) * mn))
        #return types.undefined
        return types.undefined(*((types.undefined,) * mn))

    def has_signatures(self):
        return len(self.dispatcher.overloads) > 0

    def signatures(self):
        try:
            cres_list = list(self.dispatcher.overloads.values())
        except KeyError as msg:
            print(f'BBBBBB: {msg}')
            cres_list = []

            print((self.dispatcher.overloads.copy()))
            
        for cres in cres_list:
            yield types.unliteral(cres.signature)

    @property
    def name(self):
        return f'{type(self).__name__}({self.dispatcher.py_func.__name__})'

    def __repr__(self):
        return f'{self.name}[{", ".join(map(str, self.signatures()))}]'

    def get_call_type(self, context, args, kws):
        args = tuple(map(types.unliteral, args))
        cres = self.dispatcher.get_compile_result(args, compile=False)
        if cres is not None:
            return types.unliteral(cres.signature)
        template, pysig, args, kws = self.dispatcher.get_call_template(args,
                                                                       kws)
        sig = template(context.context).apply(args, kws)
        # return sig  # uncomment to reproduce `a() + b() -> 2 * a()` issue
        return types.unliteral(sig)

    def check_signature(self, sig, compile=False):
        print(f'check_signature: {sig=}')
        if super(DispatcherFunctionTypeImpl, self).check_signature(
                sig, compile=compile):
            return True
        if compile:
            try:
                self.dispatcher.compile(sig.args)
                return True
            except Exception as msg:
                print(f'FAILED TO COMPILE: {msg}')
                pass
        return False

    def unwrap(self):
        return types.Dispatcher(self.dispatcher)


class FunctionType(Type):
    """
    First-class function type.
    """

    cconv = None
    _hash = None
    _key = None

    def __new__(cls, impl):
        print('NEW', cls, impl)
        obj = Type.__new__(cls)
        obj.__init__(impl)
        if obj.determined:
            return cls._intern(obj)
        print(obj.__dict__)
        return obj

    def __getattr__(self, name):
        if name == '_code':
            if not self.determined:
                assert 0, (self, )
                return -1
            inst = type(self)._intern(self)
            self._code = inst._code
            print('GETATTR', name, ' -> ', inst._code)
            return self._code
        if name == 'key':
            if not self.determined:
                return types.undefined.key
            return self.ftype.name
        if name == 'name':
            if not self.determined:
                return types.undefined.name
            return self.ftype.name
        if name == 'ftype':
            sig = self.signature()
            return FunctionPrototype(sig.return_type, sig.args)
        raise AttributeError(name)

    def unify(self, context, other):
        if not self.determined:
            if other.determined:
                print('UNIFY', self, other, self.impl)
                self.impl.check_signature(other.signature(), compile=True)
                print(self)
                return other

    def __hash__(self):
        key = self.key
        h = hash(key)
        print(f'HASH[{id(self)}]: {key=} -> {h=}')
        #assert self.has_signatures()
        if self._hash is None:
            self._key = key
            self._hash = h
            #if not hasattr(self, '_code'):
            #    # intern the function type
            #    inst = type(self)._intern(self)
            #    self._code = inst._code
            #    #assert inst is self, (inst, self)
        else:
            assert self._hash == h, ((self._hash, self._key), (h, self.key))
        return h

    @property
    def determined(self):
        return self.has_signatures()

    def __init__(self, impl):
        self.impl = impl

    @staticmethod
    def fromobject(obj):
        import inspect
        from numba.core.dispatcher import Dispatcher
        from numba.core.ccallback import CFunc
        from numba.core.typing.templates import Signature
        if isinstance(obj, CFunc):
            impl = CFuncFunctionTypeImpl(obj)
        elif isinstance(obj, Signature):
            impl = FunctionTypeImpl(obj)
        elif isinstance(obj, Dispatcher):
            print('fromobject:', obj.targetoptions, inspect.isgeneratorfunction(obj.py_func), obj.py_func, len(obj.overloads), obj._cache)
            if isinstance(obj._type, FunctionType):
                return obj._type
            if obj.targetoptions.get('no_cfunc_wrapper'):
                return types.Dispatcher(obj)
            if obj._cache.cache_path:
                obj.targetoptions['no_cfunc_wrapper'] = True
                return types.Dispatcher(obj)
            if 0 and obj.overloads:
                return types.Dispatcher(obj)
            if 0 and obj.overloads:
                for cres in obj.overloads.values():
                    if cres.library.has_dynamic_globals:
                        return types.Dispatcher(obj)
            if inspect.isgeneratorfunction(obj.py_func):
                obj.targetoptions['no_cfunc_wrapper'] = True
                return types.Dispatcher(obj)
            if obj._impl_kind == 'generated':
                obj.targetoptions['no_cfunc_wrapper'] = True
                return types.Dispatcher(obj)
            if obj._compiling_counter:
                obj.targetoptions['no_cfunc_wrapper'] = True
                return types.Dispatcher(obj)
            if 0 and obj.targetoptions.get('no_cfunc_wrapper', True):
                # first-class function is disabled for the given
                # dispatcher instance, fallback to default:
                return types.Dispatcher(obj)
            impl = DispatcherFunctionTypeImpl(obj)
            typ = FunctionType(impl)
            obj._type = typ
            return typ
        elif isinstance(obj, WrapperAddressProtocol):
            impl = FunctionTypeImpl(obj.signature())
        else:
            raise NotImplementedError(
                f'function type from {type(obj).__name__}')
        return FunctionType(impl)

    @property
    def name(self):
        return self.ftype.name

    @property
    def key(self):
        return self.ftype.name

    def has_signatures(self):
        """Return True if function type contains known signatures.
        """
        return self.impl.has_signatures()

    def signature(self):
        return self.impl.signature

    @property
    def ftype(self):
        sig = self.signature()
        return FunctionPrototype(sig.return_type, sig.args)

    def get_call_type(self, context, args, kws):
        return self.impl.get_call_type(context, args, kws)

    def get_ftype(self, sig=None):
        """Return function prorotype of the given or default signature.
        """
        if sig is None:
            sig = self.signature()
        else:
            # Check that the first-class function type implementation
            # supports the given signature.
            if not self.impl.check_signature(sig):
                raise ValueError(f'[{type(self.impl).__name__}]{repr(self)} does not support {sig}')
        return FunctionPrototype(sig.return_type, sig.args)

    def check_signature(self, sig, compile=False):
        """Check if the given signature belongs to the function type.
        Try to compile if enabled.
        """
        return self.impl.check_signature(sig, compile=compile)

    def matches(self, other, compile=False):
        """Return True if other's signatures matches exactly with self's
        signatures. If other does not have known signatures, try to
        compile (if enabled).
        """
        if self.has_signatures():
            if other.has_signatures():
                return self.signature() == other.signature()
            if compile:
                return other.check_signature(self.signature(), compile=compile)
        return False

    def supports(self, signature):
        """Check if function type can be a first-class function for the given signature.
        """
        return self.impl.supports(signature)

    def unwrap(self):
        return self.impl.unwrap()

class FunctionPrototype(Type):
    """
    Represents the prototype of a first-class function type.
    Used internally.
    """
    mutable = True
    cconv = None

    def __init__(self, rtype, atypes):
        self.rtype = rtype
        self.atypes = tuple(atypes)

        assert isinstance(rtype, Type), (rtype)
        lst = []
        for atype in self.atypes:
            assert isinstance(atype, Type), (atype)
            lst.append(atype.name)
        name = '%s(%s)' % (rtype, ', '.join(lst))

        super(FunctionPrototype, self).__init__(name)

    @property
    def key(self):
        return self.name


class WrapperAddressProtocol(ABC):
    """Base class for Wrapper Address Protocol.

    Objects that inherit from the WrapperAddressProtocol can be passed
    as arguments to Numba jit compiled functions where it can be used
    as first-class functions. As a minimum, the derived types must
    implement two methods ``__wrapper_address__`` and ``signature``.
    """

    @abstractmethod
    def __wrapper_address__(self):
        """Return the address of a first-class function.

        Returns
        -------
        addr : int
        """

    @abstractmethod
    def signature(self):
        """Return the signature of a first-class function.

        Returns
        -------
        sig : Signature
          The returned Signature instance represents the type of a
          first-class function that the given WrapperAddressProtocol
          instance represents.
        """
